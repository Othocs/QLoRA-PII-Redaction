"""Build the week 3 part B training mixes and the Gretel OOD dev set.

    python data/prepare_train_mix.py

Needs data/processed/train_clean_{5k,10k}.jsonl (data/clean_labels.py). Writes:
  train_nemo_{5k,10k}.jsonl   Nemotron-PII *train* split, as <=1,200-char windows
  train_mix_10k.jsonl          clean OpenPII 5k + Nemotron 5k, shuffled
  train_mix_20k.jsonl          clean OpenPII 10k + Nemotron 10k, shuffled
  gretel_dev.jsonl             500 EN + 500 non-EN docs from Gretel's *train* split
                               (model selection without touching the Gretel test set)
"""

from __future__ import annotations

import ast
import json
import random
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


def main() -> None:
    stats_path = PROCESSED / "stats_train.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

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
