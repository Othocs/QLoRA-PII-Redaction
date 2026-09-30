"""Draw the hand-audit sample from the training pool and write the review sheets.

    python data/audit/sample_audit.py [--n 200] [--seed 7]

Writes, next to this file:
  audit_sample.jsonl   the sampled documents (our Example format)
  audit_spans.csv      one row per gold span: fill `verdict` (+ `correct_label` / `note`)
  audit_docs.csv       one row per document: fill `missed_pii` with PII the gold labels missed

verdict values: ok | wrong_label | not_pii | bad_boundary
missed_pii format: "LABEL: value; LABEL: value" (empty if nothing was missed)

Refuses to overwrite sheets that already contain answers.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

from pii_gateway.spans import read_examples, write_examples

HERE = Path(__file__).parent
CONTEXT = 60


def inline_masked(text: str, spans) -> str:
    """Text with each gold span shown as [LABEL: value], so a reviewer reads it in one pass."""
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s.start):
        if s.start < pos:
            continue
        out.append(text[pos : s.start])
        out.append(f"[{s.label}: {text[s.start : s.end]}]")
        pos = s.end
    out.append(text[pos:])
    return "".join(out)


def context(text: str, start: int, end: int) -> str:
    left = text[max(0, start - CONTEXT) : start].replace("\n", " ")
    right = text[end : end + CONTEXT].replace("\n", " ")
    return f"…{left}«{text[start:end]}»{right}…"


def has_answers(path: Path, column: str) -> bool:
    if not path.exists():
        return False
    with open(path, newline="", encoding="utf-8") as f:
        return any((row.get(column) or "").strip() for row in csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/processed/train_50k.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    spans_csv, docs_csv = HERE / "audit_spans.csv", HERE / "audit_docs.csv"
    if has_answers(spans_csv, "verdict") or has_answers(docs_csv, "missed_pii"):
        sys.exit("audit sheets already have answers; move them away to resample")

    by_region = defaultdict(list)
    for ex in read_examples(args.pool):
        by_region[ex.meta.get("region")].append(ex)
    rng = random.Random(args.seed)
    regions = sorted(by_region)
    sample = []
    for i, region in enumerate(regions):  # equal share per region, remainder to the first ones
        k = args.n // len(regions) + (i < args.n % len(regions))
        sample.extend(rng.sample(by_region[region], k))
    rng.shuffle(sample)

    write_examples(HERE / "audit_sample.jsonl", sample)
    with open(spans_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["doc_id", "span_idx", "label", "value", "context", "verdict", "correct_label", "note"]
        )
        for ex in sample:
            for i, s in enumerate(ex.spans):
                w.writerow(
                    [
                        ex.id,
                        i,
                        s.label,
                        ex.text[s.start : s.end],
                        context(ex.text, s.start, s.end),
                        "",
                        "",
                        "",
                    ]
                )
    with open(docs_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "region", "n_spans", "annotated_text", "missed_pii", "note"])
        for ex in sample:
            w.writerow(
                [
                    ex.id,
                    ex.meta.get("region"),
                    len(ex.spans),
                    inline_masked(ex.text, ex.spans),
                    "",
                    "",
                ]
            )
    n_spans = sum(len(ex.spans) for ex in sample)
    print(f"{len(sample)} docs, {n_spans} gold spans -> {spans_csv.name}, {docs_csv.name}")


if __name__ == "__main__":
    main()
