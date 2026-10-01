"""Score the filled audit sheets and update the results block in AUDIT.md.

    python data/audit/score_audit.py

Per label: share of gold spans judged wrong (wrong_label | not_pii | bad_boundary)
with a 95% Wilson interval, over the rows reviewed so far. Also counts PII the
gold labels missed (audit_docs.csv `missed_pii`). Only the text between the
RESULTS markers in AUDIT.md is rewritten; the hand-written notes are kept.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
VERDICTS = {"ok", "wrong_label", "not_pii", "bad_boundary"}
TAGS = ("event_date", "place", "injected")
BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def score() -> str:
    per_label: dict[str, Counter] = defaultdict(Counter)
    tags: dict[str, Counter] = defaultdict(Counter)  # note tag -> label counts
    unknown = Counter()
    with open(HERE / "audit_spans.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row["verdict"] or "").strip().lower()
            if not v:
                continue
            if v not in VERDICTS:
                unknown[v] += 1
                continue
            per_label[row["label"]][v] += 1
            tag = (row.get("note") or "").removeprefix("[llm]").strip().split(":")[0].split(" ")[0]
            if v == "ok" and tag in TAGS:
                tags[tag][row["label"]] += 1

    missed: Counter = Counter()
    docs_reviewed = 0
    with open(HERE / "audit_docs.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = (row["missed_pii"] or "").strip()
            note = (row["note"] or "").strip()
            if m or note:
                docs_reviewed += 1
            for item in filter(None, (x.strip() for x in m.split(";"))):
                if item.lower() in ("none", "-"):
                    continue
                missed[item.split(":", 1)[0].strip().upper()] += 1

    lines = [
        "| Label | Reviewed | Wrong | Error rate | 95% CI | wrong_label | not_pii | bad_boundary |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    tot = Counter()
    for label in sorted(per_label, key=lambda k: -sum(per_label[k].values())):
        c = per_label[label]
        n = sum(c.values())
        k = n - c["ok"]
        tot.update(c)
        lo, hi = wilson(k, n)
        lines.append(
            f"| {label} | {n} | {k} | {k / n:.1%} | {lo:.1%}–{hi:.1%} | "
            f"{c['wrong_label']} | {c['not_pii']} | {c['bad_boundary']} |"
        )
    n, k = sum(tot.values()), sum(tot.values()) - tot["ok"]
    if n:
        lo, hi = wilson(k, n)
        lines.append(
            f"| **All** | {n} | {k} | {k / n:.1%} | {lo:.1%}–{hi:.1%} | "
            f"{tot['wrong_label']} | {tot['not_pii']} | {tot['bad_boundary']} |"
        )
    out = [f"Spans reviewed: {n}. Documents with a missed-PII or note entry: {docs_reviewed}.", ""]
    out += lines if n else ["No verdicts filled in yet."]
    if tags:
        out += ["", "Spans judged correct by the dataset's convention but tagged in notes:"]
        for tag in TAGS:
            if tags[tag]:
                detail = ", ".join(f"{lab} {c}" for lab, c in tags[tag].most_common())
                out.append(f"- `{tag}`: {sum(tags[tag].values())} ({detail})")
    if missed:
        out += [
            "",
            "PII missed by the gold labels, by label: "
            + ", ".join(f"{k} {v}" for k, v in missed.most_common()),
        ]
    if unknown:
        out += ["", f"Unrecognised verdicts (ignored): {dict(unknown)}"]
    return "\n".join(out)


def main() -> None:
    block = score()
    path = HERE / "AUDIT.md"
    doc = path.read_text(encoding="utf-8")
    new = re.sub(
        re.escape(BEGIN) + r".*?" + re.escape(END),
        f"{BEGIN}\n{block}\n{END}",
        doc,
        flags=re.S,
    )
    path.write_text(new, encoding="utf-8")
    print(block)


if __name__ == "__main__":
    main()
