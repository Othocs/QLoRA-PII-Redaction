"""End-to-end: Presidio (small spaCy model) on the 20-doc fixture -> results JSON + summary."""

import json

import pytest

pytest.importorskip("presidio_analyzer")
pytest.importorskip("en_core_web_sm")


@pytest.mark.slow
def test_run_eval_presidio_fixture(tmp_path):
    from eval.run_eval import main

    written = main(
        [
            "--systems", "presidio",
            "--testsets", "fixture",
            "--out", str(tmp_path),
            "--spacy-model", "en_core_web_sm",
        ]
    )  # fmt: skip
    assert [p.name for p in written] == ["presidio__fixture.json"]
    r = json.loads(written[0].read_text())
    assert r["n_examples"] == 20
    m = r["metrics"]
    for key in ("leakage_chars", "leakage_docs", "over_redaction"):
        assert 0 <= m[key] <= 1
    assert set(m["strict"]) == {"precision", "recall", "f1"}
    assert "per_label" in m and "per_region" in m
    assert r["latency"]["p95_ms_per_1k_chars"] > 0
    assert r["latency"]["n_errors"] == 0
    assert (tmp_path / "runs" / "presidio__fixture.jsonl").exists()
    assert "Presidio" in (tmp_path / "SUMMARY.md").read_text()


def test_registry_unknown_system():
    from pii_gateway.detectors.registry import build

    with pytest.raises(KeyError):
        build("nope")
