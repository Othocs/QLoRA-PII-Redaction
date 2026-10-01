"""Build the English OpenPII splits.

    python data/prepare_openpii.py extract   # download, keep English rows, write stats.json
    python data/prepare_openpii.py split     # stratified pools, held-out regions, MinHash dedup
    python data/prepare_openpii.py all       # both

Outputs go to `processed_dir` (see configs/data/openpii_en.yaml):
train_2k / train_10k / train_50k / dev / test_id / test_holdout_regions (.jsonl) + stats.json.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import yaml

from pii_gateway.spans import Example, Span, read_examples, validate_spans, write_examples

RAW_FILES = {"train": "data/train.jsonl", "validation": "data/validation.jsonl"}


# ---------------------------------------------------------------- conversion


def convert_row(row: dict) -> tuple[Example | None, list[str]]:
    """OpenPII row -> Example. Returns (None, errors) if any span doesn't match its text."""
    text = row["source_text"]
    spans = [
        Span(start=m["start"], end=m["end"], label=m["label"], text=m["value"])
        for m in row["privacy_mask"]
    ]
    errors = validate_spans(text, spans)
    if errors:
        return None, errors
    spans.sort(key=lambda s: (s.start, s.end))
    ex = Example(
        id=f"openpii-{row['uid']}",
        text=text,
        spans=spans,
        meta={
            "source": "openpii",
            "uid": row["uid"],
            "region": row.get("region"),
            "split": row.get("split"),
        },
    )
    return ex, []


def iter_raw(path: Path, language: str) -> Iterator[dict]:
    # A cheap substring test skips json.loads for the ~87% of rows in other languages.
    needle = re.compile(rf'"language":\s*"{re.escape(language)}"')
    with open(path, encoding="utf-8") as f:
        for line in f:
            if needle.search(line):
                row = json.loads(line)
                if row.get("language") == language:
                    yield row


class _ExtractStats:
    def __init__(self) -> None:
        self.n_ok = self.n_bad = 0
        self.bad_examples: list[dict] = []
        self.regions: Counter = Counter()
        self.labels: Counter = Counter()
        self.lengths: list[int] = []

    def scan(self, rows: Iterable[dict]) -> Iterator[Example]:
        """Convert rows, yielding good Examples and counting the rest."""
        for row in rows:
            ex, errs = convert_row(row)
            if ex is None:
                self.n_bad += 1
                if len(self.bad_examples) < 5:
                    self.bad_examples.append({"uid": row["uid"], "errors": errs[:3]})
                continue
            self.n_ok += 1
            self.regions[ex.meta["region"]] += 1
            self.labels.update(s.label for s in ex.spans)
            self.lengths.append(len(ex.text))
            yield ex

    def to_dict(self) -> dict:
        lengths = self.lengths
        return {
            "rows": self.n_ok,
            "rows_dropped_offset_mismatch": self.n_bad,
            "dropped_examples": self.bad_examples,
            "rows_per_region": dict(self.regions.most_common()),
            "spans_per_label": dict(self.labels.most_common()),
            "text_len_chars_p50_p95_max": [
                int(np.percentile(lengths, 50)),
                int(np.percentile(lengths, 95)),
                max(lengths),
            ]
            if lengths
            else [],
        }


def extract(cfg: dict) -> dict:
    raw_dir, out_dir = Path(cfg["raw_dir"]), Path(cfg["processed_dir"])
    stats: dict = {"extract": {}}
    for split, rel in RAW_FILES.items():
        raw = raw_dir / Path(rel).name
        if not raw.exists():
            from huggingface_hub import hf_hub_download

            print(f"downloading {rel} ...", file=sys.stderr)
            tmp = raw_dir / "_hf"
            got = hf_hub_download(cfg["repo_id"], rel, repo_type="dataset", local_dir=tmp)
            raw_dir.mkdir(parents=True, exist_ok=True)
            Path(got).rename(raw)
        acc = _ExtractStats()
        out = out_dir / f"openpii_{cfg['language']}_{split}.jsonl"
        write_examples(out, acc.scan(iter_raw(raw, cfg["language"])))
        stats["extract"][split] = acc.to_dict()
        print(f"{split}: {acc.n_ok} English rows ({acc.n_bad} dropped) -> {out}", file=sys.stderr)
    _update_stats(out_dir, stats)
    return stats


# ---------------------------------------------------------------- splitting


def mask_text(ex: Example) -> str:
    """Replace every gold span with [LABEL]; what's left is the generator's template."""
    parts, pos = [], 0
    for s in sorted(ex.spans, key=lambda s: s.start):
        if s.start < pos:  # overlapping gold spans: keep the first
            continue
        parts.append(ex.text[pos : s.start])
        parts.append(f"[{s.label}]")
        pos = s.end
    parts.append(ex.text[pos:])
    return "".join(parts)


def shingles(text: str, n: int) -> set[str]:
    words = text.lower().split()
    if len(words) <= n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def stratum(ex: Example, label_freq: Counter) -> tuple[str, str]:
    """Region x rarest label present (by global frequency)."""
    labels = {s.label for s in ex.spans}
    rarest = min(labels, key=lambda lab: (label_freq[lab], lab)) if labels else "NONE"
    return (ex.meta.get("region") or "?", rarest)


def stratified_order(examples: list[Example], seed: int) -> list[Example]:
    """Order examples so that *every prefix* is approximately stratified.

    Each stratum is shuffled; example k of a stratum of size n gets position
    (k + u) / n with u ~ U(0,1), and the global order sorts on that. A prefix of
    length m therefore holds ~m * n / N items from each stratum, which gives the
    nested 2k / 10k / 50k training sets for free.
    """
    rng = random.Random(seed)
    label_freq = Counter(s.label for ex in examples for s in ex.spans)
    groups: dict[tuple, list[Example]] = defaultdict(list)
    for ex in examples:
        groups[stratum(ex, label_freq)].append(ex)
    keyed = []
    for key in sorted(groups):
        g = groups[key]
        rng.shuffle(g)
        n = len(g)
        keyed.extend(((k + rng.random()) / n, ex.id, ex) for k, ex in enumerate(g))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [ex for _, _, ex in keyed]


def _minhash(text: str, num_perm: int, ngram: int):
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm, seed=1)
    m.update_batch([s.encode("utf-8") for s in shingles(text, ngram)])
    return m


def take_deduped(
    candidates: Iterable[Example], k: int, lsh, taken_ids: set[str], mh_cfg: dict
) -> tuple[list[Example], int]:
    """Take up to k candidates that are not near-duplicates of anything indexed in `lsh`."""
    out, dropped = [], 0
    for ex in candidates:
        if len(out) >= k:
            break
        if ex.id in taken_ids:
            continue
        m = _minhash(mask_text(ex), mh_cfg["num_perm"], mh_cfg["ngram"])
        if lsh.query(m):
            dropped += 1
            continue
        out.append(ex)
        taken_ids.add(ex.id)
    return out, dropped


def size_name(n: int) -> str:
    """50000 -> '50k', 2500 -> '2500'."""
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)


def build_splits(
    train: list[Example], validation: list[Example], cfg: dict
) -> tuple[dict[str, list[Example]], dict]:
    from datasketch import MinHashLSH

    sizes, mh_cfg, seed = cfg["sizes"], cfg["minhash"], cfg["seed"]
    holdout = set(cfg.get("holdout_regions") or [])

    def in_dist(ex: Example) -> bool:
        return ex.meta.get("region") not in holdout

    train_order = stratified_order([ex for ex in train if in_dist(ex)], seed)
    pool = train_order[: sizes["train_pool"]]
    splits: dict[str, list[Example]] = {f"train_{size_name(sizes['train_pool'])}": pool}
    for n in sizes["train_nested"]:
        splits[f"train_{size_name(n)}"] = pool[:n]

    lsh = MinHashLSH(threshold=mh_cfg["threshold"], num_perm=mh_cfg["num_perm"])
    for ex in pool:
        lsh.insert(ex.id, _minhash(mask_text(ex), mh_cfg["num_perm"], mh_cfg["ngram"]))

    val_order = stratified_order(validation, seed + 1)
    taken: set[str] = set()
    dedup: dict[str, dict] = {}
    for name, want, pred in [
        ("dev", sizes["dev"], in_dist),
        ("test_id", sizes["test_id"], in_dist),
        ("test_holdout_regions", sizes["test_holdout_regions"], lambda ex: not in_dist(ex)),
    ]:
        if name == "test_holdout_regions" and not holdout:
            continue
        got, dropped = take_deduped((ex for ex in val_order if pred(ex)), want, lsh, taken, mh_cfg)
        splits[name] = got
        dedup[name] = {
            "kept": len(got),
            "dropped_near_duplicate_of_train": dropped,
            "drop_rate": round(dropped / max(1, dropped + len(got)), 4),
        }

    stats = {
        "holdout_regions": sorted(holdout),
        "sizes": {k: len(v) for k, v in splits.items()},
        "dedup": dedup,
        "train_pool_rows_per_region": dict(
            Counter(ex.meta.get("region") for ex in pool).most_common()
        ),
    }
    return splits, stats


def split(cfg: dict) -> dict:
    out_dir = Path(cfg["processed_dir"])
    lang = cfg["language"]
    if not cfg.get("holdout_regions"):
        print(
            "warning: holdout_regions is empty; test_holdout_regions will not be built. "
            "Pick regions from stats.json and set them in the config.",
            file=sys.stderr,
        )
    train = list(read_examples(out_dir / f"openpii_{lang}_train.jsonl"))
    validation = list(read_examples(out_dir / f"openpii_{lang}_validation.jsonl"))
    splits, stats = build_splits(train, validation, cfg)
    for name, exs in splits.items():
        write_examples(out_dir / f"{name}.jsonl", exs)
        print(f"{name}: {len(exs)}", file=sys.stderr)
    _update_stats(out_dir, {"split": stats})
    return stats


def _update_stats(out_dir: Path, new: dict) -> None:
    path = out_dir / "stats.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(old, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("step", choices=["extract", "split", "all"])
    ap.add_argument("--config", default="configs/data/openpii_en.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.step in ("extract", "all"):
        extract(cfg)
    if args.step in ("split", "all"):
        split(cfg)


if __name__ == "__main__":
    main()
