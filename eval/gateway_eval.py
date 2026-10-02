"""Score the full gateway (model + validators) from a model's saved predictions. No GPU.

    python -m eval.gateway_eval --tags m5_targeted,m5_s42 --testsets tab,support_desk_300,test_id

For each tag and test set it reads results/runs/<tag>__<set>.jsonl (the model's per-document
spans) and writes two result sets, both scored with the gateway label scope (IBAN and
IPADDRESS gold counted, unlike the model-only default):
  <tag>_gl   the model alone            (like-for-like baseline for the gateway row)
  <tag>_gw   model + validators, exactly as pii_gateway.api combines them: validators on the
             normalised text, spans mapped back, recall-first union with the model's spans
Each gets results/<name>__<set>.json and results/runs/<name>__<set>.jsonl, so eval.bootstrap
(--gateway-labels) and eval.summarize pick them up.

One approximation: the live gateway also feeds the model the normalised text; here the
model's predictions come from the raw text (they were made by eval.run_eval).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.metrics import aggregate
from eval.run_eval import load_testset
from pii_gateway.detectors.validators import ValidatorDetector
from pii_gateway.merge import recall_first_union
from pii_gateway.normalize import normalize, spans_to_original
from pii_gateway.spans import Span


def model_predictions(tag: str, testset: str, results: Path) -> dict[str, list[Span]]:
    preds = {}
    with open(results / "runs" / f"{tag}__{testset}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            preds[r["id"]] = [Span(p["start"], p["end"], p["label"], p.get("text", ""),
                                   p.get("source", "llm")) for p in r["pred"]]  # fmt: skip
    return preds


def gateway_spans(text: str, model: list[Span], validators: ValidatorDetector) -> list[Span]:
    norm = normalize(text)
    found = spans_to_original(norm, text, validators.detect(norm.text))
    model = [Span(s.start, s.end, s.label, text[s.start : s.end], "llm") for s in model]
    return recall_first_union(text, found, model)


def write(results: Path, name: str, testset: str, examples, preds, source_tag: str) -> dict:
    metrics = aggregate(examples, preds)
    out = {"system": name, "detector": "gateway" if name.endswith("_gw") else "model",
           "source_predictions": source_tag, "testset": testset, "n_examples": len(examples),
           "gateway_labels": True, "metrics": metrics}  # fmt: skip
    (results / f"{name}__{testset}.json").write_text(json.dumps(out, indent=2) + "\n")
    with open(results / "runs" / f"{name}__{testset}.jsonl", "w", encoding="utf-8") as f:
        for ex in examples:
            spans = [s.to_dict(full=True) for s in preds.get(ex.id, [])]
            f.write(json.dumps({"id": ex.id, "pred": spans}, ensure_ascii=False) + "\n")
    return metrics


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--testsets", required=True)
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    results = Path(args.results)
    validators = ValidatorDetector()
    rows = []
    for testset in [t for t in args.testsets.split(",") if t]:
        _, examples = load_testset(testset, None, gateway_labels=True)
        for tag in [t for t in args.tags.split(",") if t]:
            model = model_predictions(tag, testset, results)
            gw = {ex.id: gateway_spans(ex.text, model.get(ex.id, []), validators)
                  for ex in examples}  # fmt: skip
            m_gl = write(results, f"{tag}_gl", testset, examples, model, tag)
            m_gw = write(results, f"{tag}_gw", testset, examples, gw, tag)
            row = {
                "tag": tag,
                "testset": testset,
                "model_leak": 100 * m_gl["leakage_chars"],
                "gateway_leak": 100 * m_gw["leakage_chars"],
                "model_over": 100 * m_gl["over_redaction"],
                "gateway_over": 100 * m_gw["over_redaction"],
            }
            rows.append(row)
            print(f"{tag:14s} {testset:18s} leak model {row['model_leak']:6.2f}  gateway "
                  f"{row['gateway_leak']:6.2f} | over model {row['model_over']:6.2f}  gateway "
                  f"{row['gateway_over']:6.2f}", file=sys.stderr)  # fmt: skip
    return rows


if __name__ == "__main__":
    main()
