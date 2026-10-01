"""Build the out-of-distribution test sets (week 3).

    python data/prepare_eval_sets.py [nemotron tab gretel openpii_xx]   # default: all

Writes data/processed/{nemotron,tab,gretel_en,gretel_xx,openpii_xx}.jsonl and
data/processed/stats_eval.json. Labels are mapped with configs/labels/eval/*.yaml:
labels outside our scope become IGNORE spans (see eval/metrics.py).
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import yaml
from prepare_openpii import (
    _minhash,
    convert_row,
    iter_raw,
    mask_text,
    stratified_order,
    take_deduped,
)

from pii_gateway.spans import Example, Span, validate_spans, write_examples

IGNORE = "IGNORE"
LABEL_DIR = Path("configs/labels/eval")
GRETEL_LANG = {
    "English": "en",
    "German": "de",
    "Dutch": "nl",
    "Spanish": "es",
    "Italian": "it",
    "Swedish": "sv",
    "France": "fr",
}


def label_map(name: str) -> dict[str, str]:
    return yaml.safe_load((LABEL_DIR / f"{name}.yaml").read_text())


def fetch(cfg: dict, key: str) -> Path:
    from huggingface_hub import hf_hub_download

    sub = cfg[key]
    local = Path(cfg["raw_dir"]) / sub["repo_id"].split("/")[1]
    return Path(hf_hub_download(sub["repo_id"], sub["file"], repo_type="dataset", local_dir=local))


def make_example(
    id_: str, text: str, raw_spans: list[tuple[int, int, str]], lmap: dict, meta: dict
) -> Example | None:
    """Map labels, take span text from the source (datasets disagree on its case),
    and drop the document if any span is out of range."""
    spans = []
    for start, end, label in raw_spans:
        if not (0 <= start < end <= len(text)):
            return None
        spans.append(Span(start, end, lmap.get(label, IGNORE), text[start:end]))
    spans.sort(key=lambda s: (s.start, s.end))
    assert not validate_spans(text, spans)
    return Example(id_, text, spans, meta)


def stats_for(exs: list[Example], dropped: int) -> dict:
    labels = Counter(s.label for ex in exs for s in ex.spans)
    return {
        "rows": len(exs),
        "dropped_bad_offsets": dropped,
        "chars": sum(len(ex.text) for ex in exs),
        "per_language": dict(Counter(ex.meta.get("language") for ex in exs).most_common()),
        "spans_per_label": dict(labels.most_common()),
    }


def build_nemotron(cfg: dict, rng: random.Random) -> dict[str, list[Example]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(
        fetch(cfg, "nemotron"), columns=["uid", "domain", "locale", "text", "spans"]
    ).to_pylist()
    rows = rng.sample(rows, cfg["nemotron"]["n"])
    lmap, out, dropped = label_map("nemotron"), [], 0
    for r in rows:
        raw = [(s["start"], s["end"], s["label"]) for s in ast.literal_eval(r["spans"])]
        meta = {
            "source": "nemotron",
            "language": "en",
            "region": r["locale"],
            "domain": r["domain"],
        }
        # uid is shared by a document's "us" and "intl" versions: uid + locale is the key
        ex = make_example(f"nemotron-{r['uid']}-{r['locale']}", r["text"], raw, lmap, meta)
        if ex is None:
            dropped += 1
        else:
            out.append(ex)
    return {"nemotron": out}, {"nemotron": dropped}


def build_tab(cfg: dict, rng: random.Random) -> dict[str, list[Example]]:
    z = zipfile.ZipFile(fetch(cfg, "tab"))
    docs = json.loads(z.read("echr_test.json"))
    # One annotation per court case: prefer a quality-checked one, else the first.
    chosen: dict[str, dict] = {}
    for d in docs:
        cur = chosen.get(d["doc_id"])
        if cur is None or (d.get("quality_checked") and not cur.get("quality_checked")):
            chosen[d["doc_id"]] = d
    lmap, out, dropped = label_map("tab"), [], 0
    for doc_id, d in sorted(chosen.items()):
        raw = []
        for m in d["entity_mentions"]:
            masked = m["identifier_type"] in ("DIRECT", "QUASI")
            label = m["entity_type"] if masked and m["entity_type"] in lmap else "__ignore__"
            raw.append((m["start_offset"], m["end_offset"], label))
        meta = {"source": "tab", "language": "en", "annotator": d["annotator_id"]}
        ex = make_example(f"tab-{doc_id}", d["text"], raw, lmap, meta)
        if ex is None:
            dropped += 1
        else:
            out.append(ex)
    return {"tab": out}, {"tab": dropped}


def build_gretel(cfg: dict, rng: random.Random) -> dict[str, list[Example]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(fetch(cfg, "gretel")).to_pylist()
    lmap, en, xx = label_map("gretel"), [], []
    dropped = Counter()
    for r in rows:
        lang = GRETEL_LANG.get(r["language"], r["language"])
        raw = [(s["start"], s["end"], s["label"]) for s in json.loads(r["pii_spans"])]
        meta = {"source": "gretel", "language": lang, "document_type": r["document_type"]}
        ex = make_example(f"gretel-{r['index']}", r["generated_text"], raw, lmap, meta)
        if ex is None:
            dropped["gretel_en" if lang == "en" else "gretel_xx"] += 1
            continue
        (en if lang == "en" else xx).append(ex)
    en = rng.sample(en, min(cfg["gretel"]["n_english"], len(en)))
    return {"gretel_en": en, "gretel_xx": xx}, dict(dropped)


def build_openpii_xx(cfg: dict, rng: random.Random) -> dict[str, list[Example]]:
    sub = cfg["openpii_xx"]
    raw_val = Path("data/raw/validation.jsonl")
    out = []
    for lang in sub["languages"]:
        exs = []
        for row in iter_raw(raw_val, lang):
            ex, errs = convert_row(row)
            if ex is not None:
                ex.meta["language"] = lang
                exs.append(ex)
        out.extend(rng.sample(exs, min(sub["n_per_language"], len(exs))))
        print(f"openpii_xx {lang}: {len(exs)} available", file=sys.stderr)
    return {"openpii_xx": out}, {}


def build_val_in_region(cfg: dict, rng: random.Random) -> dict[str, list[Example]]:
    """HPO plan val_ood: 1,000 OpenPII validation docs from the held-out region (IN),
    disjoint from test_holdout_regions and near-duplicate-free against train_50k."""
    from datasketch import MinHashLSH

    from pii_gateway.spans import read_examples

    sub = cfg["val_in_region"]
    processed = Path(cfg["processed_dir"])
    test_ids = {ex.id for ex in read_examples(processed / "test_holdout_regions.jsonl")}
    pool = [
        ex
        for ex in read_examples(processed / "openpii_en_validation.jsonl")
        if ex.meta.get("region") in sub["regions"] and ex.id not in test_ids
    ]
    mh = {"num_perm": 128, "threshold": 0.8, "ngram": 5}
    lsh = MinHashLSH(threshold=mh["threshold"], num_perm=mh["num_perm"])
    for ex in read_examples(processed / "train_50k.jsonl"):
        lsh.insert(ex.id, _minhash(mask_text(ex), mh["num_perm"], mh["ngram"]))
    order = stratified_order(pool, cfg["seed"] + 7)
    got, dropped = take_deduped(order, sub["n"], lsh, set(test_ids), mh)
    for ex in got:
        ex.meta["language"] = "en"
    print(f"val_in_region: {len(got)} kept, {dropped} near-duplicates dropped", file=sys.stderr)
    return {"val_in_region": got}, {"val_in_region": 0}


BUILDERS = {
    "nemotron": build_nemotron,
    "tab": build_tab,
    "gretel": build_gretel,
    "openpii_xx": build_openpii_xx,
    "val_in_region": build_val_in_region,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sets", nargs="*", help=f"any of {', '.join(BUILDERS)} (default: all)")
    ap.add_argument("--config", default="configs/data/eval_sets.yaml")
    args = ap.parse_args()
    unknown = set(args.sets) - set(BUILDERS)
    if unknown:
        ap.error(f"unknown sets: {', '.join(sorted(unknown))}")
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["processed_dir"])
    stats_path = out_dir / "stats_eval.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for name in args.sets or list(BUILDERS):
        rng = random.Random(f"{cfg['seed']}-{name}")  # independent of build order
        splits, dropped = BUILDERS[name](cfg, rng)
        for split, exs in splits.items():
            write_examples(out_dir / f"{split}.jsonl", exs)
            stats[split] = stats_for(exs, dropped.get(split, 0))
            print(f"{split}: {len(exs)} docs, {stats[split]['chars']:,} chars", file=sys.stderr)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
