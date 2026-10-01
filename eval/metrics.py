"""Redaction metrics. Pure Python + numpy; no model dependencies.

Headline metric is leakage (share of gold PII characters left unmasked), because in
redaction a miss is a data breach and an extra mask is a nuisance. Leakage and
over-redaction ignore labels on purpose: they answer "was it masked?", which also
makes systems with different label sets comparable. Strict F1 needs exact
boundaries *and* label; partial F1 needs any overlap and ignores the label.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from pii_gateway.spans import Example, Span, dedupe_spans


def _div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def _prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    # No predictions -> nothing wrongly predicted (P=1); no gold -> nothing to miss (R=1).
    p = _div(tp, n_pred, 1.0)
    r = _div(tp, n_gold, 1.0)
    return {"precision": p, "recall": r, "f1": _div(2 * p * r, p + r)}


def _mask(n: int, spans: Iterable[Span]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s in spans:
        m[max(0, s.start) : min(n, s.end)] = True
    return m


@dataclass
class _Acc:
    """Running totals for one slice (overall, a label or a region)."""

    docs: int = 0
    docs_leaking: int = 0
    gold_chars: int = 0
    leaked_chars: int = 0
    pred_chars: int = 0
    over_chars: int = 0
    strict_tp: int = 0
    n_gold: int = 0
    n_pred: int = 0
    partial_gold_hit: int = 0
    partial_pred_hit: int = 0

    def result(self) -> dict:
        strict = _prf(self.strict_tp, self.n_pred, self.n_gold)
        pp = _div(self.partial_pred_hit, self.n_pred, 1.0)
        pr = _div(self.partial_gold_hit, self.n_gold, 1.0)
        return {
            "docs": self.docs,
            "n_gold_spans": self.n_gold,
            "n_pred_spans": self.n_pred,
            "leakage_chars": _div(self.leaked_chars, self.gold_chars),
            "leakage_docs": _div(self.docs_leaking, self.docs),
            "over_redaction": _div(self.over_chars, self.pred_chars),
            "strict": strict,
            "partial": {"precision": pp, "recall": pr, "f1": _div(2 * pp * pr, pp + pr)},
        }


@dataclass
class DocScore:
    gold_chars: int
    leaked_chars: int
    pred_chars: int
    over_chars: int
    leaks: bool
    strict_tp: int
    n_gold: int
    n_pred: int
    partial_gold_hit: int
    partial_pred_hit: int
    # per-label pieces for aggregation
    by_label: dict[str, dict[str, int]] = field(default_factory=dict)


IGNORE = "IGNORE"


def score_doc(text: str, gold: list[Span], pred: list[Span]) -> DocScore:
    """Score one document.

    Gold spans labelled IGNORE mark regions outside the evaluation scope (e.g. a
    dataset's "company" label): they are not counted as leaks, masking them is not
    over-redaction, and predictions lying entirely inside them are not counted.
    """
    ignore = [s for s in gold if s.label == IGNORE]
    gold = dedupe_spans(s for s in gold if s.label != IGNORE)
    n = len(text)
    gm = _mask(n, gold)
    im = _mask(n, ignore) & ~gm  # real gold wins where the two overlap
    pred = [s for s in dedupe_spans(pred) if not im[max(0, s.start) : min(n, s.end)].all()]
    pm = _mask(n, pred) & ~im
    tp_keys = {(s.start, s.end, s.label) for s in gold} & {(s.start, s.end, s.label) for s in pred}

    def overlaps(s: Span, m: np.ndarray) -> bool:
        return bool(m[s.start : s.end].any())

    by_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in gold:
        seg = pm[s.start : s.end]
        d = by_label[s.label]
        d["n_gold"] += 1
        d["gold_chars"] += s.end - s.start
        d["leaked_chars"] += int((~seg).sum())
        d["partial_gold_hit"] += int(seg.any())
        d["strict_tp"] += int((s.start, s.end, s.label) in tp_keys)
    for s in pred:
        d = by_label[s.label]
        d["n_pred"] += 1
        d["partial_pred_hit"] += int(overlaps(s, gm))

    return DocScore(
        gold_chars=int(gm.sum()),
        leaked_chars=int((gm & ~pm).sum()),
        pred_chars=int(pm.sum()),
        over_chars=int((pm & ~gm).sum()),
        leaks=any(not pm[s.start : s.end].all() for s in gold),
        strict_tp=len(tp_keys),
        n_gold=len(gold),
        n_pred=len(pred),
        partial_gold_hit=sum(overlaps(s, pm) for s in gold),
        partial_pred_hit=sum(overlaps(s, gm) for s in pred),
        by_label={k: dict(v) for k, v in by_label.items()},
    )


def _add(acc: _Acc, d: DocScore) -> None:
    acc.docs += 1
    acc.docs_leaking += int(d.leaks)
    acc.gold_chars += d.gold_chars
    acc.leaked_chars += d.leaked_chars
    acc.pred_chars += d.pred_chars
    acc.over_chars += d.over_chars
    acc.strict_tp += d.strict_tp
    acc.n_gold += d.n_gold
    acc.n_pred += d.n_pred
    acc.partial_gold_hit += d.partial_gold_hit
    acc.partial_pred_hit += d.partial_pred_hit


def aggregate(
    examples: Iterable[Example],
    predictions: Mapping[str, list[Span]],
    group_keys: Sequence[str] = ("region", "language"),
) -> dict:
    """Micro-averaged metrics overall, per gold/pred label and per `meta[key]` for each
    key in `group_keys` that at least one example has.

    Examples with no entry in `predictions` count as "predicted nothing".
    Per-label leakage uses gold spans of that label; per-label strict P/R/F1 use
    spans of that label on both sides, so a baseline label we can't map (NAME,
    OTHER) shows up with precision 0 and no gold.
    """
    overall = _Acc()
    groups: dict[str, dict[str, _Acc]] = {k: defaultdict(_Acc) for k in group_keys}
    labels: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for ex in examples:
        d = score_doc(ex.text, ex.spans, predictions.get(ex.id, []))
        _add(overall, d)
        for key in group_keys:
            if ex.meta.get(key) is not None:
                _add(groups[key][str(ex.meta[key])], d)
        for lab, counts in d.by_label.items():
            for k, v in counts.items():
                labels[lab][k] += v

    per_label = {}
    for lab in sorted(labels):
        c = labels[lab]
        per_label[lab] = {
            "n_gold": c["n_gold"],
            "n_pred": c["n_pred"],
            "leakage_chars": _div(c["leaked_chars"], c["gold_chars"]),
            "partial_recall": _div(c["partial_gold_hit"], c["n_gold"], 1.0),
            "strict": _prf(c["strict_tp"], c["n_pred"], c["n_gold"]),
        }

    out = overall.result()
    out["per_label"] = per_label
    keep = ("docs", "leakage_chars", "leakage_docs", "over_redaction", "strict", "partial")
    for key, accs in groups.items():
        if accs:
            out[f"per_{key}"] = {
                k: {kk: vv for kk, vv in acc.result().items() if kk in keep}
                for k, acc in sorted(accs.items())
            }
    return out
