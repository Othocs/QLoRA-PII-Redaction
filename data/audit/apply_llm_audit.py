"""Write the LLM-assisted audit (llm_audit.tsv) into the review sheets; draw a spot-check sample.

    python data/audit/apply_llm_audit.py [--spotcheck 50] [--seed 11]

Every gold span gets a verdict: the one in llm_audit.tsv if listed, else "ok". Notes
are prefixed "[llm]" so human edits stay distinguishable. Also writes
audit_spotcheck.csv: half flagged spans, half "ok" spans, for a human to confirm
(`human_agrees` = y/n, plus `human_verdict` when n).
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from pii_gateway.spans import read_examples

HERE = Path(__file__).parent


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_tsv(path: Path) -> tuple[dict[tuple[int, int], dict], dict[int, list[str]]]:
    spans: dict[tuple[int, int], dict] = {}
    missed: dict[int, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        cols += [""] * (5 - len(cols))
        doc, span, verdict, correct, note = cols[:5]
        if span == "MISSED":
            missed.setdefault(int(doc), []).append(correct or note or verdict)
            continue
        spans[(int(doc), int(span))] = {"verdict": verdict, "correct_label": correct, "note": note}
    return spans, missed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spotcheck", type=int, default=50)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    verdicts, missed = load_tsv(HERE / "llm_audit.tsv")
    docs = list(read_examples(HERE / "audit_sample.jsonl"))
    by_id = {ex.id: i for i, ex in enumerate(docs)}

    rows = read_csv(HERE / "audit_spans.csv")
    for r in rows:
        v = verdicts.get((by_id[r["doc_id"]], int(r["span_idx"])), {"verdict": "ok"})
        r["verdict"] = v["verdict"]
        r["correct_label"] = v.get("correct_label", "")
        r["note"] = f"[llm] {v['note']}" if v.get("note") else ""
    with open(HERE / "audit_spans.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    drows = read_csv(HERE / "audit_docs.csv")
    for r in drows:
        m = missed.get(by_id[r["doc_id"]], [])
        r["missed_pii"] = "; ".join(m) if m else "none"
        r["note"] = "[llm] reviewed"
    with open(HERE / "audit_docs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(drows[0]))
        w.writeheader()
        w.writerows(drows)

    rng = random.Random(args.seed)
    flagged = [r for r in rows if r["verdict"] != "ok"]
    unflagged = [r for r in rows if r["verdict"] == "ok"]
    half = args.spotcheck // 2
    sample = rng.sample(flagged, min(half, len(flagged)))
    sample += rng.sample(unflagged, args.spotcheck - len(sample))
    rng.shuffle(sample)
    with open(HERE / "audit_spotcheck.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "span_idx", "label", "value", "context", "llm_verdict",
                    "llm_note", "human_agrees", "human_verdict", "human_note"])  # fmt: skip
        for r in sample:
            w.writerow([r["doc_id"], r["span_idx"], r["label"], r["value"], r["context"],
                        r["verdict"], r["note"], "", "", ""])  # fmt: skip
    n_flag = sum(r["verdict"] != "ok" for r in rows)
    print(f"{len(rows)} spans: {n_flag} flagged; {sum(map(len, missed.values()))} missed values;")
    print(f"spot-check sheet: {len(sample)} rows -> audit_spotcheck.csv")


if __name__ == "__main__":
    main()
