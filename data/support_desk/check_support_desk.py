"""Validate the hand-written support-desk set, and optionally fill in span offsets.

    python data/support_desk/check_support_desk.py            # validate
    python data/support_desk/check_support_desk.py --fill     # add missing start/end, then validate

When writing a message you may give spans as {"label", "text"} only; --fill finds
each one in the message as a whole word, left to right, and adds start/end. Checks:
  - ids are unique, labels are from our label set
  - every span's offsets match its text, sit on word boundaries, and don't overlap
  - every `meta.keep` string occurs in the text and is not inside a span
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from pii_gateway.spans import LABELS

PATH = Path(__file__).parent / "support_desk_300.jsonl"


def _find_word(text: str, value: str, pos: int) -> int:
    """First occurrence of value at or after pos that doesn't start or end inside a word,
    so a house number "9" is never matched inside "ORD-51290"."""
    m = re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)").search(text, pos)
    return m.start() if m else -1


def fill_offsets(rec: dict) -> None:
    text, pos = rec["text"], 0
    for sp in rec["spans"]:
        if "start" in sp and "end" in sp:
            pos = sp["end"]
            continue
        i = _find_word(text, sp["text"], pos)
        if i == -1:
            i = _find_word(text, sp["text"], 0)  # out of order: fall back to first occurrence
        if i == -1:
            raise ValueError(f"{rec['id']}: span text {sp['text']!r} not found as a whole word")
        sp["start"], sp["end"] = i, i + len(sp["text"])
        pos = sp["end"]
    rec["spans"].sort(key=lambda s: s["start"])


def check(rec: dict) -> list[str]:
    errs, text = [], rec["text"]
    prev_end = -1
    for sp in rec["spans"]:
        if sp["label"] not in LABELS:
            errs.append(f"unknown label {sp['label']}")
        if "start" not in sp:
            errs.append(f"span {sp.get('text')!r} has no offsets (run with --fill)")
            continue
        if text[sp["start"] : sp["end"]] != sp.get("text", text[sp["start"] : sp["end"]]):
            errs.append(f"offsets of {sp['text']!r} point at {text[sp['start'] : sp['end']]!r}")
        a, b = sp["start"], sp["end"]
        if (a > 0 and text[a - 1].isalnum()) or (b < len(text) and text[b].isalnum()):
            errs.append(f"span {sp.get('text')!r} starts or ends inside a word")
        if sp["start"] < prev_end:
            errs.append(f"span {sp.get('text')!r} overlaps the previous one")
        prev_end = sp["end"]
    for keep in rec.get("meta", {}).get("keep", []):
        i = text.find(keep)
        if i == -1:
            errs.append(f"keep string {keep!r} not in text")
        elif any(
            sp.get("start", 0) < i + len(keep) and i < sp.get("end", 0) for sp in rec["spans"]
        ):
            errs.append(f"keep string {keep!r} overlaps a PII span")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(PATH))
    ap.add_argument("--fill", action="store_true")
    args = ap.parse_args()
    path = Path(args.path)
    recs = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if args.fill:
        for r in recs:
            fill_offsets(r)
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8"
        )
    bad, seen = 0, set()
    for r in recs:
        errs = check(r)
        if r["id"] in seen:
            errs.append("duplicate id")
        seen.add(r["id"])
        for e in errs:
            print(f"{r['id']}: {e}", file=sys.stderr)
        bad += bool(errs)
    n_spans = sum(len(r["spans"]) for r in recs)
    print(f"{len(recs)} messages, {n_spans} spans, {bad} with problems (target: 300 messages)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
