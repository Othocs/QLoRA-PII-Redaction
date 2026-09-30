"""Wrappers that turn existing PII systems into `Detector`s with our label set.

Heavy imports (presidio, gliner, transformers) happen inside constructors so the
package imports without the `baselines` extra.
"""

from __future__ import annotations

from pii_gateway.detectors.base import detect_chunked, drop_contained, trim_span
from pii_gateway.spans import OTHER, Span


class PresidioDetector:
    def __init__(
        self,
        label_map: dict[str, str],
        spacy_model: str = "en_core_web_lg",
        score_threshold: float = 0.35,
        language: str = "en",
        name: str = "presidio",
    ):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        nlp = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": language, "model_name": spacy_model}],
            }
        ).create_engine()
        self.engine = AnalyzerEngine(nlp_engine=nlp, supported_languages=[language])
        self.label_map = label_map
        self.score_threshold = score_threshold
        self.language = language
        self.name = name

    def detect(self, text: str) -> list[Span]:
        results = self.engine.analyze(
            text=text, language=self.language, score_threshold=self.score_threshold
        )
        spans = []
        for r in results:
            s, e = trim_span(text, r.start, r.end)
            if e > s:
                label = self.label_map.get(r.entity_type, OTHER)
                spans.append(Span(s, e, label, text[s:e], self.name, float(r.score)))
        return drop_contained(spans)


class GlinerDetector:
    """Zero-shot GLiNER span model; `prompts` maps prompt text -> our label."""

    def __init__(
        self,
        model_id: str,
        prompts: dict[str, str],
        threshold: float = 0.3,
        name: str = "gliner",
        device: str = "cpu",
        max_chars: int = 1500,
        overlap: int = 200,
    ):
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        self.prompts = prompts
        self.threshold = threshold
        self.name = name
        self.max_chars, self.overlap = max_chars, overlap

    def _detect_chunk(self, chunk: str) -> list[Span]:
        ents = self.model.predict_entities(
            chunk, list(self.prompts), threshold=self.threshold, flat_ner=True
        )
        spans = []
        for e in ents:
            s, t = trim_span(chunk, e["start"], e["end"])
            if t > s:
                label = self.prompts.get(e["label"], OTHER)
                spans.append(Span(s, t, label, source=self.name, score=float(e["score"])))
        return spans

    def detect(self, text: str) -> list[Span]:
        return detect_chunked(text, self._detect_chunk, self.max_chars, self.overlap)


class OpenMedDetector:
    """OpenMed privacy filter: token classifier with BIOES tags, decoded here.

    Not in the local default set (1.4B-parameter MoE on an 8 GB laptop); run it on a GPU box.
    """

    def __init__(
        self,
        model_id: str,
        label_map: dict[str, str],
        name: str = "openmed",
        device: str | int = -1,
        max_chars: int = 2000,
        overlap: int = 200,
    ):
        from transformers import pipeline

        self.pipe = pipeline(
            "token-classification",
            model=model_id,
            trust_remote_code=True,
            aggregation_strategy="none",
            device=device,
        )
        self.label_map = label_map
        self.name = name
        self.max_chars, self.overlap = max_chars, overlap

    def _detect_chunk(self, chunk: str) -> list[Span]:
        toks = self.pipe(chunk)  # "O" tokens are already filtered out by the pipeline
        spans: list[Span] = []
        cur: list | None = None  # [start, end, category, scores]
        for t in sorted(toks, key=lambda t: t["start"]):
            tag = t["entity"]
            prefix, _, cat = tag.partition("-") if "-" in tag else ("S", "", tag)
            gap = chunk[cur[1] : t["start"]] if cur else ""
            contiguous = cur is not None and cur[2] == cat and gap.strip() == ""
            if prefix in ("I", "E") and contiguous:
                cur[1] = t["end"]
                cur[3].append(t["score"])
            else:
                if cur:
                    spans.append(self._close(chunk, cur))
                cur = [t["start"], t["end"], cat, [t["score"]]]
            if prefix in ("E", "S") and cur:
                spans.append(self._close(chunk, cur))
                cur = None
        if cur:
            spans.append(self._close(chunk, cur))
        return [s for s in spans if s is not None]

    def _close(self, chunk: str, cur: list) -> Span | None:
        s, e = trim_span(chunk, cur[0], cur[1])
        if e <= s:
            return None
        label = self.label_map.get(cur[2], OTHER)
        return Span(s, e, label, source=self.name, score=float(sum(cur[3]) / len(cur[3])))

    def detect(self, text: str) -> list[Span]:
        return detect_chunked(text, self._detect_chunk, self.max_chars, self.overlap)
