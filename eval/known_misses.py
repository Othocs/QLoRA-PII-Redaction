"""Which gold spans that a baseline missed entirely does a candidate now cover?

    python -m eval.known_misses --baseline p2_m4_lr4e-4 --candidate m5_targeted [--testset ...]

Milestone 1 diagnostic: the targeted data was designed around M4's misses on
support_desk_hard, so this set can only confirm the fixes landed; the promotion gate is the
unseen support_desk_fresh. A span counts as covered when any predicted span overlaps it.
Reads results/runs/<tag>__<testset>.jsonl.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.run_eval import load_testset


def predictions(tag: str, testset: str, results: Path) -> dict[str, list[tuple[int, int]]]:
    out = {}
    with open(results / "runs" / f"{tag}__{testset}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = [(p["start"], p["end"]) for p in r["pred"]]
    return out


def missed(spans, preds) -> list:
    return [s for s in spans if s.label != "IGNORE"
            and not any(a < s.end and s.start < b for a, b in preds)]  # fmt: skip


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--testset", default="support_desk_hard")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    results = Path(args.results)
    _, examples = load_testset(args.testset, None)
    base = predictions(args.baseline, args.testset, results)
    cand = predictions(args.candidate, args.testset, results)
    rows, fixed, new_misses = [], Counter(), Counter()
    for ex in examples:
        b_miss = missed(ex.spans, base.get(ex.id, []))
        c_miss = {(s.start, s.end) for s in missed(ex.spans, cand.get(ex.id, []))}
        for s in b_miss:
            ok = (s.start, s.end) not in c_miss
            fixed[s.label, ok] += 1
            rows.append({"id": ex.id, "label": s.label, "text": ex.text[s.start : s.end],
                         "recovered": ok})  # fmt: skip
        b_keys = {(s.start, s.end) for s in b_miss}
        for s in missed(ex.spans, cand.get(ex.id, [])):
            if (s.start, s.end) not in b_keys:
                new_misses[s.label] += 1
    for r in rows:
        mark = "FIXED " if r["recovered"] else "missed"
        print(f"{mark}  {r['id']}  {r['label']:<16} {r['text']!r}")
    out = {
        "testset": args.testset,
        "baseline_misses": len(rows),
        "recovered": sum(r["recovered"] for r in rows),
        "new_misses_by_label": dict(new_misses),
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
