"""95% bootstrap confidence intervals over documents, from saved predictions.

    python -m eval.bootstrap --testset tab --tags m3_seed1,m3_seed2,m3_seed3 [--n 2000]
    python -m eval.bootstrap --testset tab --tags m3_s1,m4_s1 --paired   # difference A - B

Reads results/runs/<tag>__<testset>.jsonl (per-example predictions written by
eval/run_eval.py) and the test set itself, scores every document once with
eval.metrics.score_doc, then resamples documents with replacement. Leakage and
over-redaction are ratios of sums, so each resample only re-adds per-document counts.

With several tags (seeds of one model) the per-document counts are averaged across
seeds first, so the interval covers document sampling around the seed mean; the
spread between seeds is reported next to it. --paired takes exactly two tags and gives
the interval of their leakage difference on the same resampled documents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.metrics import score_doc
from eval.run_eval import load_testset
from pii_gateway.spans import Span


def doc_counts(tag: str, testset: str, results: Path) -> np.ndarray:
    """Per-document [gold_chars, leaked_chars, pred_chars, over_chars], in test-set order."""
    _, examples = load_testset(testset, None)
    preds = {}
    with open(results / "runs" / f"{tag}__{testset}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            preds[r["id"]] = [Span(p["start"], p["end"], p["label"]) for p in r["pred"]]
    rows = []
    for ex in examples:
        d = score_doc(ex.text, ex.spans, preds.get(ex.id, []))
        rows.append([d.gold_chars, d.leaked_chars, d.pred_chars, d.over_chars])
    return np.asarray(rows, dtype=float)


def ratios(c: np.ndarray) -> tuple[float, float]:
    s = c.sum(axis=0)
    leak = s[1] / s[0] if s[0] else 0.0
    over = s[3] / s[2] if s[2] else 0.0
    return leak, over


def bootstrap(counts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """(n, 2) array of resampled (leakage, over-redaction)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), size=(n, len(counts)))
    out = np.empty((n, 2))
    for i in range(n):
        out[i] = ratios(counts[idx[i]])
    return out


def ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--tags", required=True, help="comma-separated: seeds of one model")
    ap.add_argument("--paired", action="store_true", help="two tags: CI of leakage A - B")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    results = Path(args.results)
    tags = [t for t in args.tags.split(",") if t]
    per_tag = {t: doc_counts(t, args.testset, results) for t in tags}

    if args.paired:
        if len(tags) != 2:
            ap.error("--paired needs exactly two tags")
        a, b = per_tag[tags[0]], per_tag[tags[1]]
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(a), size=(args.n, len(a)))
        diffs = np.array([ratios(a[i])[0] - ratios(b[i])[0] for i in idx])
        lo, hi = ci(diffs)
        point = ratios(a)[0] - ratios(b)[0]
        out = {"testset": args.testset, "a": tags[0], "b": tags[1],
               "leakage_diff_pct": 100 * point, "ci95_pct": [100 * lo, 100 * hi],
               "a_better_share": float((diffs < 0).mean())}  # fmt: skip
    else:
        mean_counts = np.mean([per_tag[t] for t in tags], axis=0)
        samples = bootstrap(mean_counts, args.n)
        leak, over = ratios(mean_counts)
        seed_leaks = [100 * ratios(per_tag[t])[0] for t in tags]
        out = {
            "testset": args.testset,
            "tags": tags,
            "docs": len(mean_counts),
            "leakage_pct": 100 * leak,
            "leakage_ci95_pct": [100 * x for x in ci(samples[:, 0])],
            "over_redaction_pct": 100 * over,
            "over_redaction_ci95_pct": [100 * x for x in ci(samples[:, 1])],
            "seed_leakage_pct": seed_leaks,
            "seed_spread_pct": max(seed_leaks) - min(seed_leaks),
        }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
