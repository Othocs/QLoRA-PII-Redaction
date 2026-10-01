"""Re-score logged raw LLM outputs with strict vs fuzzy span alignment (no model needed).

    python -m eval.realign results/runs/<tag>__<testset>.raw.jsonl --testset nemotron

The raw log (written when PII_LLM_LOG_RAW=1) holds every chunk's model output, so
alignment changes can be measured offline. Prints leakage/F1 for both modes plus
the most common reasons values are dropped, with examples.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from eval.metrics import aggregate
from eval.run_eval import load_testset
from pii_gateway.detectors.base import merge_overlapping
from pii_gateway.detectors.llm import align_spans, parse_output
from pii_gateway.spans import Span


def realign(raw_path: str, testset: str, fuzzy: bool) -> tuple[dict, list[dict], dict]:
    _, examples = load_testset(testset, None)
    by_id = {ex.id: ex for ex in examples}
    per_doc: dict[str, list[Span]] = defaultdict(list)
    drops: list[dict] = []
    fmt = Counter()
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            ex = by_id[e["id"]]
            chunk = ex.text[e["offset"] : e["offset"] + e["chunk_len"]]
            parsed = parse_output(e["raw"])
            fmt["chunks"] += 1
            fmt["valid_json"] += parsed.valid_json
            fmt["truncated"] += e.get("truncated", False)
            fmt["items"] += len(parsed.items)
            d: list[dict] = []
            spans, _ = align_spans(chunk, parsed.items, fuzzy=fuzzy, drops=d)
            for x in d:
                x["id"] = e["id"]
            drops.extend(d)
            per_doc[e["id"]].extend(
                Span(s.start + e["offset"], s.end + e["offset"], s.label) for s in spans
            )
    preds = {k: merge_overlapping(v) for k, v in per_doc.items()}
    return aggregate([by_id[k] for k in by_id], preds), drops, dict(fmt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--testset", required=True)
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args()
    for mode in ("strict", "fuzzy"):
        m, drops, fmt = realign(args.raw, args.testset, fuzzy=mode == "fuzzy")
        n = fmt["items"]
        print(
            f"{mode:6s} leakage {100 * m['leakage_chars']:5.1f}%  docs leaking "
            f"{100 * m['leakage_docs']:5.1f}%  over {100 * m['over_redaction']:5.1f}%  "
            f"strict F1 {m['strict']['f1']:.3f}  partial F1 {m['partial']['f1']:.3f}  "
            f"dropped {len(drops)}/{n} ({100 * len(drops) / max(n, 1):.1f}%)"
        )
    print(
        f"chunks {fmt['chunks']}, valid JSON {fmt['valid_json']}, "
        f"hit token limit {fmt['truncated']}"
    )
    print("still dropped after fuzzy alignment, by reason/label:")
    print(Counter((d["reason"], d["label"]) for d in drops).most_common(12))
    for d in drops[: args.examples]:
        print("  ", d["label"], repr(d["text"])[:100])


if __name__ == "__main__":
    main()
