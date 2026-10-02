"""Build the training mixes and the dev sets that come with them.

    python data/prepare_train_mix.py            # week 3 part B: Nemotron, mixes 10k/20k, gretel_dev
    python data/prepare_train_mix.py phase2     # HPO plan phase 2: Gretel EN, mix 30k, nemotron_dev
    python data/prepare_train_mix.py targeted   # milestone 1: mix 30k + targeted messages

Needs data/processed/train_clean_{5k,10k}.jsonl (data/clean_labels.py). Writes:
  train_nemo_{5k,10k}.jsonl   Nemotron-PII *train* split, as <=1,200-char windows
  train_mix_10k.jsonl          clean OpenPII 5k + Nemotron 5k, shuffled
  train_mix_20k.jsonl          clean OpenPII 10k + Nemotron 10k, shuffled
  gretel_dev.jsonl             500 EN + 500 non-EN docs from Gretel's *train* split
                               (model selection without touching the Gretel test set)
phase2:
  train_gretel_10k.jsonl       Gretel *train* split, English only, <=1,200-char windows,
                               gretel_dev documents excluded
  train_mix_30k.jsonl          clean OpenPII 10k + Nemotron 10k + Gretel EN 10k, shuffled
  nemotron_dev.jsonl           1,000 windows of Nemotron train docs NOT used for training,
                               eval label map (out-of-scope labels = IGNORE)
targeted:
  train_mix_32k.jsonl          train_mix_30k + data/synthetic/targeted_2k.jsonl, shuffled
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import yaml
from prepare_eval_sets import GRETEL_LANG, label_map, make_example

from pii_gateway.detectors.base import chunk_text
from pii_gateway.spans import Example, Span, read_examples, write_examples

PROCESSED = Path("data/processed")
RAW = Path("data/raw/eval")
SEED = 31
WINDOW = 1200  # = inference chunk size = longest OpenPII training document


def fetch(repo_id: str, file: str) -> Path:
    from huggingface_hub import hf_hub_download

    local = RAW / repo_id.split("/")[1]
    return Path(hf_hub_download(repo_id, file, repo_type="dataset", local_dir=local))


def windows(ex: Example, max_chars: int = WINDOW) -> list[Example]:
    """Split into <=max_chars windows; keep windows that no span crosses."""
    out = []
    for k, (off, chunk) in enumerate(chunk_text(ex.text, max_chars, 0)):
        end = off + len(chunk)
        inside, crossing = [], False
        for s in ex.spans:
            if s.start >= off and s.end <= end:
                inside.append(Span(s.start - off, s.end - off, s.label, s.text))
            elif s.start < end and s.end > off:
                crossing = True
                break
        if not crossing:
            out.append(Example(f"{ex.id}-w{k}", chunk, inside, dict(ex.meta)))
    return out


def build_nemotron_train(rng: random.Random, n: int) -> tuple[list[Example], dict]:
    import pyarrow.parquet as pq

    cfg = yaml.safe_load(Path("configs/labels/train/nemotron.yaml").read_text())
    lmap, ok_labels = cfg["map"], set(cfg["map"]) | set(cfg["not_personal"])
    identity = {lab: lab for lab in lmap.values()}  # spans below already carry our labels
    rows = pq.read_table(
        fetch("nvidia/Nemotron-PII", "data/train-00000-of-00001.parquet"),
        columns=["uid", "domain", "locale", "text", "spans"],
    ).to_pylist()
    stats = Counter(rows=len(rows))
    blocking = Counter()
    wins: list[Example] = []
    for r in rows:
        raw = ast.literal_eval(r["spans"])
        bad = {s["label"] for s in raw} - ok_labels
        if bad:
            stats["docs_unnameable_pii"] += 1
            blocking.update(bad)
            continue
        kept = [(s["start"], s["end"], lmap[s["label"]]) for s in raw if s["label"] in lmap]
        meta = {"source": "nemotron_train", "language": "en", "region": r["locale"]}
        ex = make_example(f"nemotrain-{r['uid']}", r["text"], kept, identity, meta)
        if ex is None:
            stats["docs_bad_offsets"] += 1
            continue
        stats["docs_used"] += 1
        w = windows(ex)
        stats["windows"] += len(w)
        wins.extend(w)
    rng.shuffle(wins)
    stats["windows_without_pii"] = sum(not w.spans for w in wins[:n])
    return wins[:n], {**stats, "top_blocking_labels": dict(blocking.most_common(10))}


def build_gretel_dev(rng: random.Random, n_en: int = 500, n_xx: int = 500) -> list[Example]:
    import pyarrow.parquet as pq

    rows = pq.read_table(
        fetch("gretelai/synthetic_pii_finance_multilingual", "data/train-00000-of-00001.parquet")
    ).to_pylist()
    lmap, en, xx = label_map("gretel"), [], []
    for r in rows:
        lang = GRETEL_LANG.get(r["language"], r["language"])
        raw = [(s["start"], s["end"], s["label"]) for s in json.loads(r["pii_spans"])]
        meta = {"source": "gretel_train", "language": lang}
        ex = make_example(f"greteltrain-{r['index']}", r["generated_text"], raw, lmap, meta)
        if ex is not None:
            (en if lang == "en" else xx).append(ex)
    return rng.sample(en, n_en) + rng.sample(xx, n_xx)


HONORIFICS = {"mr", "mrs", "ms", "miss", "mx", "dr", "prof", "sir", "madam", "mister", "madame"}
PARTICLES = {"van", "von", "de", "der", "den", "da", "di", "du", "la", "le", "del", "della",
             "ten", "ter", "bin", "al", "el", "dos", "das", "uit"}  # fmt: skip
PLACEHOLDER = re.compile(
    r"[\d_@]|^(partner|party|client|customer|buyer|seller|employee|user|tenant|landlord|"
    r"borrower|lender|company|recipient|sender|applicant|member|director|manager)\b",
    re.IGNORECASE,
)


def split_name(text: str, start: int, end: int) -> list[tuple[int, int, str]] | None:
    """A full-name span -> TITLE / GIVENNAME / SURNAME spans; None for a placeholder.

    Leading honorifics -> TITLE; the last token (with lowercase particles such as "van",
    "de" in front of it) -> SURNAME; the tokens before -> GIVENNAME. A single name after a
    title is a SURNAME ("Dr. Patel"), alone a GIVENNAME ("Nancy").
    """
    value = text[start:end]
    toks = [(m.start() + start, m.end() + start, m.group()) for m in re.finditer(r"\S+", value)]
    if not toks or PLACEHOLDER.search(value) or (len(toks) > 1 and len(toks[-1][2]) == 1):
        return None  # empty, digits/underscores, a role ("Partner A")
    title = []
    while toks and toks[0][2].rstrip(".").lower() in HONORIFICS:
        title.append(toks.pop(0))
    out = [(title[0][0], title[-1][1], "TITLE")] if title else []
    if not toks:
        return None
    if len(toks) == 1:
        return out + [(toks[0][0], toks[0][1], "SURNAME" if title else "GIVENNAME")]
    j = len(toks) - 1
    while j > 1 and toks[j - 1][2].lower() in PARTICLES:
        j -= 1
    return out + [(toks[0][0], toks[j - 1][1], "GIVENNAME"), (toks[j][0], toks[-1][1], "SURNAME")]


def build_gretel_train(
    rng: random.Random, n: int, exclude_ids: set[str]
) -> tuple[list[Example], dict]:
    import pyarrow.parquet as pq

    cfg = yaml.safe_load(Path("configs/labels/train/gretel.yaml").read_text())
    lmap, split = cfg["map"], set(cfg["split"])
    ok_labels = set(lmap) | split | set(cfg["not_personal"])
    identity = {lab: lab for lab in [*lmap.values(), "TITLE", "GIVENNAME", "SURNAME"]}
    rows = pq.read_table(
        fetch("gretelai/synthetic_pii_finance_multilingual", "data/train-00000-of-00001.parquet")
    ).to_pylist()
    stats, blocking = Counter(), Counter()
    wins: list[Example] = []
    for r in rows:
        if r["language"] != "English":
            continue
        stats["english_rows"] += 1
        doc_id = f"greteltrain-{r['index']}"
        if doc_id in exclude_ids:
            stats["docs_in_gretel_dev"] += 1
            continue
        raw = json.loads(r["pii_spans"])
        bad = {s["label"] for s in raw} - ok_labels
        if bad:
            stats["docs_unnameable_pii"] += 1
            blocking.update(bad)
            continue
        text, kept, placeholder = r["generated_text"], [], False
        for s in raw:
            if s["label"] in split:
                parts = split_name(text, s["start"], s["end"]) if s["end"] <= len(text) else None
                if parts is None:
                    placeholder = True
                    break
                kept.extend(parts)
            elif s["label"] in lmap:
                kept.append((s["start"], s["end"], lmap[s["label"]]))
        if placeholder:
            stats["docs_placeholder_name"] += 1
            continue
        meta = {"source": "gretel_train", "language": "en"}
        ex = make_example(doc_id, text, kept, identity, meta)
        if ex is None:
            stats["docs_bad_offsets"] += 1
            continue
        stats["docs_used"] += 1
        w = windows(ex)
        stats["windows"] += len(w)
        wins.extend(w)
    rng.shuffle(wins)
    stats["windows_without_pii"] = sum(not w.spans for w in wins[:n])
    return wins[:n], {**stats, "top_blocking_labels": dict(blocking.most_common(10))}


def build_nemotron_dev(rng: random.Random, n: int, used_uids: set[str]) -> list[Example]:
    """One window per Nemotron train document never used for training; eval label map."""
    import pyarrow.parquet as pq

    lmap = label_map("nemotron")
    rows = pq.read_table(
        fetch("nvidia/Nemotron-PII", "data/train-00000-of-00001.parquet"),
        columns=["uid", "locale", "text", "spans"],
    ).to_pylist()
    rows = [r for r in rows if r["uid"] not in used_uids]
    rng.shuffle(rows)
    out: list[Example] = []
    for r in rows:
        if len(out) >= n:
            break
        raw = [(s["start"], s["end"], s["label"]) for s in ast.literal_eval(r["spans"])]
        meta = {"source": "nemotron_train_dev", "language": "en", "region": r["locale"]}
        ex = make_example(f"nemodev-{r['uid']}-{r['locale']}", r["text"], raw, lmap, meta)
        w = [x for x in windows(ex) if any(s.label != "IGNORE" for s in x.spans)] if ex else []
        if w:
            out.append(rng.choice(w))
    return out


def phase2(stats: dict) -> None:
    gretel_dev_ids = {ex.id for ex in read_examples(PROCESSED / "gretel_dev.jsonl")}
    gretel, gstats = build_gretel_train(random.Random(f"{SEED}-gretel"), 10000, gretel_dev_ids)
    write_examples(PROCESSED / "train_gretel_10k.jsonl", gretel)
    stats["gretel_train"] = gstats

    mix = (
        list(read_examples(PROCESSED / "train_clean_10k.jsonl"))
        + list(read_examples(PROCESSED / "train_nemo_10k.jsonl"))
        + gretel
    )
    random.Random(f"{SEED}-train_mix_30k").shuffle(mix)
    write_examples(PROCESSED / "train_mix_30k.jsonl", mix)
    stats["train_mix_30k"] = dict(Counter(ex.meta.get("source") for ex in mix))

    used = {
        ex.id.removeprefix("nemotrain-").rsplit("-w", 1)[0]
        for ex in read_examples(PROCESSED / "train_nemo_10k.jsonl")
    }
    dev = build_nemotron_dev(random.Random(f"{SEED}-nemotron-dev"), 1000, used)
    write_examples(PROCESSED / "nemotron_dev.jsonl", dev)
    stats["nemotron_dev"] = {"rows": len(dev), "excluded_training_docs": len(used)}


def targeted(stats: dict) -> None:
    """Milestone 1: train_mix_32k = train_mix_30k + the DeepSeek-written targeted messages."""
    extra = list(read_examples(Path("data/synthetic/targeted_2k.jsonl")))
    mix = list(read_examples(PROCESSED / "train_mix_30k.jsonl")) + extra
    random.Random(f"{SEED}-train_mix_32k").shuffle(mix)
    write_examples(PROCESSED / "train_mix_32k.jsonl", mix)
    stats["train_mix_32k"] = dict(Counter(ex.meta.get("source") for ex in mix))
    stats["targeted"] = {
        "rows": len(extra),
        "patterns": dict(Counter(ex.meta.get("pattern") for ex in extra)),
        "labels": dict(Counter(s.label for ex in extra for s in ex.spans)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("step", nargs="?", default="base", choices=["base", "phase2", "targeted"])
    args = ap.parse_args()
    stats_path = PROCESSED / "stats_train.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    if args.step == "targeted":
        targeted(stats)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps({k: stats[k] for k in ("train_mix_32k", "targeted")}, indent=2),
              file=sys.stderr)  # fmt: skip
        return
    if args.step == "phase2":
        phase2(stats)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps({k: stats[k] for k in ("gretel_train", "train_mix_30k", "nemotron_dev")},
                         indent=2), file=sys.stderr)  # fmt: skip
        return

    nemo, nstats = build_nemotron_train(random.Random(f"{SEED}-nemo"), 10000)
    write_examples(PROCESSED / "train_nemo_10k.jsonl", nemo)
    write_examples(PROCESSED / "train_nemo_5k.jsonl", nemo[:5000])
    stats["nemotron_train"] = nstats

    for name, clean, k in [("train_mix_10k", "train_clean_5k", 5000),
                           ("train_mix_20k", "train_clean_10k", 10000)]:  # fmt: skip
        mix = list(read_examples(PROCESSED / f"{clean}.jsonl")) + nemo[:k]
        random.Random(f"{SEED}-{name}").shuffle(mix)
        write_examples(PROCESSED / f"{name}.jsonl", mix)
        stats[name] = dict(Counter(ex.meta.get("source") for ex in mix))

    dev = build_gretel_dev(random.Random(f"{SEED}-gretel-dev"))
    write_examples(PROCESSED / "gretel_dev.jsonl", dev)
    stats["gretel_dev"] = dict(Counter(ex.meta["language"] for ex in dev))

    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "clean_openpii"}, indent=2),
          file=sys.stderr)  # fmt: skip


if __name__ == "__main__":
    main()
