"""System name -> Detector, built from configs/labels/*.yaml."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import yaml

from pii_gateway.detectors.base import Detector

CONFIG_DIR = Path(os.environ.get("PII_CONFIG_DIR", Path(__file__).parents[3] / "configs"))


def _labels_cfg(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / "labels" / f"{name}.yaml").read_text())


def _presidio(**kw) -> Detector:
    from pii_gateway.detectors.baselines import PresidioDetector

    cfg = _labels_cfg("presidio")
    return PresidioDetector(
        label_map=cfg["map"],
        spacy_model=kw.get("spacy_model") or os.environ.get("PII_SPACY_MODEL", cfg["spacy_model"]),
        score_threshold=cfg["score_threshold"],
        language=cfg["language"],
    )


def _gliner(key: str) -> Callable[..., Detector]:
    def build(**kw) -> Detector:
        from pii_gateway.detectors.baselines import GlinerDetector

        cfg = _labels_cfg("gliner")["models"][key]
        return GlinerDetector(
            model_id=cfg["model_id"],
            prompts=cfg["prompts"],
            threshold=cfg["threshold"],
            name=key,
            device=kw.get("device") or "cpu",
        )

    return build


def _openmed(**kw) -> Detector:
    from pii_gateway.detectors.baselines import OpenMedDetector

    cfg = _labels_cfg("openmed")
    return OpenMedDetector(
        model_id=cfg["model_id"], label_map=cfg["map"], device=kw.get("device", -1)
    )


DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"


def _llm(with_adapter: bool) -> Callable[..., Detector]:
    def build(**kw) -> Detector:
        from pii_gateway.detectors.llm import LLMDetector

        adapter = kw.get("adapter")
        if with_adapter and not adapter:
            raise ValueError("system 'lora' needs --adapter <path to the trained adapter>")
        return LLMDetector(
            base_model=kw.get("base_model") or DEFAULT_BASE_MODEL,
            adapter=adapter if with_adapter else None,
            backend=kw.get("backend") or "vllm",
            name="lora" if with_adapter else "base_llm",
        )

    return build


SYSTEMS: dict[str, Callable[..., Detector]] = {
    "presidio": _presidio,
    "gliner_knowledgator": _gliner("gliner_knowledgator"),
    "gliner_nvidia": _gliner("gliner_nvidia"),
    "openmed": _openmed,
    "base_llm": _llm(with_adapter=False),
    "lora": _llm(with_adapter=True),
}


def build(name: str, **kw) -> Detector:
    if name not in SYSTEMS:
        raise KeyError(f"unknown system {name!r}; known: {', '.join(SYSTEMS)}")
    return SYSTEMS[name](**kw)
