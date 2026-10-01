"""Rule-based filters for OpenPII label noise found by the audit (data/audit/AUDIT.md).

    python data/clean_labels.py --check-audit     # precision/recall vs the audited spans
    python data/clean_labels.py                    # write train_clean_{5k,10k}.jsonl

Each rule looks at a short context window around one gold span and says whether
the span is probably *not* a person's data (a money amount tagged
CREDITCARDNUMBER, a percentage or group statistic tagged AGE, ...). Training
examples with any flagged span are dropped, not relabelled: a wrong rule then
only costs data, whereas relabelling would teach the model to leak.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from pii_gateway.spans import Example, Span, read_examples, write_examples

AUDIT_DIR = Path("data/audit")
PROCESSED = Path("data/processed")


def _before(text: str, s: Span, n: int = 30) -> str:
    return text[max(0, s.start - n) : s.start].lower()


def _after(text: str, s: Span, n: int = 30) -> str:
    return text[s.end : s.end + n].lower()


MONEY_BEFORE = re.compile(
    r"[$£€]\s*$|(\b(funds?|fund of|funding of|rent of|deposit of|value of|up to|over|balance|"
    r"contribution|credit line of|reimbursement of|fee|allocated|covers|total|amount|cost|"
    r"price|salary|budget|pledge|grant)\b[^.\n]{0,20})$"
)
MONEY_AFTER = re.compile(r"^\s*(per (month|year|week)|fee|balance|million|thousand)\b")


# "fee payable via 4111…", "charged to the card ending in …": a real card after a money word
CARD_CUE_BEFORE = re.compile(
    r"\b(via|card|ending in|ending|charged to|number|no\.?|using|with)\s*:?\s*$"
)


def money_as_card(text: str, s: Span) -> bool:
    if s.label != "CREDITCARDNUMBER":
        return False
    before = _before(text, s)
    if CARD_CUE_BEFORE.search(before):
        return False
    return bool(MONEY_BEFORE.search(before) or MONEY_AFTER.search(_after(text, s)))


AGE_AFTER = re.compile(
    r"^\s*(%|percent|\s*%|\+?\s*years? of experience|-?\s*year[- ]milestone|minutes|hours|"
    r"million|out of|°|x\b|cm\b|kg\b)"
)
GROUP_NOUNS = (
    r"(participants|members|residents|employees|patients|users|individuals|adults|children|"
    r"students|people|travell?ers|guests|volunteers|founders|consumers|engineers|men|women|"
    r"females|males|seniors|those|anyone|staff|contributors|applicants|customers|clients)"
)
AGE_BEFORE = re.compile(
    r"(\baverage( age)?( of)?|\b" + GROUP_NOUNS + r"\s+aged|\bunder|\bover|\bat least|"
    r"\branging( in age)? from|\bfrom ages?|\bages|\brate of|"
    r"\bscore of|\bnumber of [a-z]+:?|\bincreased by|\bdecreased by|\bdropped to|\bby roughly|"
    r"\bages?\s+between|\bexperience \(years\):?\s*_*|\bboth\b[^.]{0,10})\s*\(?$"
)
AGE_RANGE = re.compile(r"^\s*([–-]|to)\s*\d|^\s*and (older|above|over)\b")  # "29–16", "38 to 15"
AGE_RANGE_END = re.compile(r"\d\s*([–-]|to)\s*$")  # the second number of a range


def age_not_a_person(text: str, s: Span) -> bool:
    if s.label != "AGE":
        return False
    before, after = _before(text, s), _after(text, s)
    return bool(
        AGE_AFTER.search(after)
        or AGE_BEFORE.search(before)
        or AGE_RANGE.search(after)
        or AGE_RANGE_END.search(before)
    )


GROUP_BEFORE = re.compile(
    r"\b(ratio|distribution|split|all|categories of|irrespective of|regardless of|source of|"
    r"demographics?|identities|analysts?)\s*\(?$"
)
GROUP_AFTER = re.compile(r"^\s*(analysts|identities|demographics|data|out of|existence)\b")


def group_attribute(text: str, s: Span) -> bool:
    if s.label not in ("GENDER", "SEX"):
        return False
    return bool(GROUP_BEFORE.search(_before(text, s, 25)) or GROUP_AFTER.search(_after(text, s)))


REF_AFTER = re.compile(
    r"^\s*(regulations?|compliance|requirements?|fee structure|entries|fields)\b"
)


def reference_not_id(text: str, s: Span) -> bool:
    return s.label in ("TAXNUM", "SOCIALNUM", "ZIPCODE", "CREDITCARDNUMBER") and bool(
        REF_AFTER.search(_after(text, s))
    )


RULES: dict[str, Callable[[str, Span], bool]] = {
    "money_as_card": money_as_card,
    "age_not_a_person": age_not_a_person,
    "group_attribute": group_attribute,
    "reference_not_id": reference_not_id,
}


def flags(ex: Example) -> list[tuple[str, Span]]:
    return [(name, s) for s in ex.spans for name, rule in RULES.items() if rule(ex.text, s)]


# ---------------------------------------------------------------- audit check


def check_audit() -> dict:
    """Compare rule hits with the audit verdicts on the 1,490 audited spans."""
    docs = {ex.id: ex for ex in read_examples(AUDIT_DIR / "audit_sample.jsonl")}
    with open(AUDIT_DIR / "audit_spans.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tp = fp = fn = 0
    per_rule = Counter()
    fps, fns = [], []
    for r in rows:
        ex = docs[r["doc_id"]]
        s = ex.spans[int(r["span_idx"])]
        hit = [name for name, rule in RULES.items() if rule(ex.text, s)]
        noisy = r["verdict"] == "not_pii"
        for name in hit:
            per_rule[(name, "tp" if noisy else "fp")] += 1
        if hit and noisy:
            tp += 1
        elif hit:
            fp += 1
            fps.append((s.label, ex.text[max(0, s.start - 40) : s.end + 25].replace("\n", " ")))
        elif noisy and s.label in ("AGE", "CREDITCARDNUMBER", "GENDER", "SEX", "TAXNUM"):
            fn += 1
            fns.append((s.label, ex.text[max(0, s.start - 40) : s.end + 25].replace("\n", " ")))
    out = {
        "flagged": tp + fp,
        "precision": tp / max(1, tp + fp),
        "recall_on_target_labels": tp / max(1, tp + fn),
        "per_rule": {f"{k[0]}:{k[1]}": v for k, v in sorted(per_rule.items())},
    }
    print(json.dumps(out, indent=2))
    print("\nfalse positives (flagged, audit says ok):")
    for lab, ctx in fps:
        print(f"  {lab:16s} …{ctx}…")
    print("\nmissed (audit not_pii on target labels, not flagged):")
    for lab, ctx in fns[:40]:
        print(f"  {lab:16s} …{ctx}…")
    return out


# ---------------------------------------------------------------- build


def build(sizes: tuple[int, ...] = (5000, 10000)) -> dict:
    """Filter train_50k in its stratified order; the first n kept examples form each set."""
    pool = list(read_examples(PROCESSED / "train_50k.jsonl"))
    kept, dropped = [], Counter()
    for ex in pool:
        f = flags(ex)
        if f:
            dropped.update(name for name, _ in f)
            dropped["_examples"] += 1
        else:
            kept.append(ex)
    stats = {
        "pool": len(pool),
        "kept": len(kept),
        "dropped_examples": dropped.pop("_examples", 0),
        "flags_by_rule": dict(dropped.most_common()),
    }
    for n in sizes:
        name = f"train_clean_{n // 1000}k"
        write_examples(PROCESSED / f"{name}.jsonl", kept[:n])
        stats[name] = len(kept[:n])
    print(json.dumps(stats, indent=2), file=sys.stderr)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check-audit", action="store_true")
    args = ap.parse_args()
    if args.check_audit:
        check_audit()
        return
    stats = build()
    path = PROCESSED / "stats_train.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old["clean_openpii"] = stats
    path.write_text(json.dumps(old, indent=2) + "\n")


if __name__ == "__main__":
    main()
