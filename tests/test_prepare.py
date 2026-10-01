import json
from pathlib import Path

import pytest
from prepare_openpii import build_splits, convert_row, mask_text, shingles, stratified_order

from pii_gateway.spans import Example, Span, read_examples

FIX = Path(__file__).parent / "fixtures"


def test_convert_raw_rows():
    rows = [json.loads(line) for line in (FIX / "openpii_raw_sample.jsonl").open()]
    for row in rows:
        ex, errs = convert_row(row)
        assert errs == []
        assert ex.id == f"openpii-{row['uid']}"
        assert len(ex.spans) == len(row["privacy_mask"])
        for s in ex.spans:
            assert ex.text[s.start : s.end] == s.text


def test_convert_rejects_bad_offsets():
    row = {
        "uid": 1,
        "source_text": "Hi Ann",
        "privacy_mask": [{"label": "GIVENNAME", "start": 0, "end": 2, "value": "Ann"}],
    }
    ex, errs = convert_row(row)
    assert ex is None and errs


def test_mask_text_and_shingles():
    ex = Example("x", "Hi Ann Lee, bye", [Span(3, 6, "GIVENNAME"), Span(7, 10, "SURNAME")])
    assert mask_text(ex) == "Hi [GIVENNAME] [SURNAME], bye"
    assert shingles("a b", 5) == {"a b"}
    assert len(shingles("a b c d e f", 5)) == 2


def _toy(n: int, region: str, prefix: str) -> list[Example]:
    out = []
    for i in range(n):
        label = ["EMAIL", "SOCIALNUM", "CITY"][i % 3]
        # distinct wording per example so MinHash doesn't treat them as duplicates
        text = f"{prefix} {i} " + " ".join(f"w{prefix}{i}_{k}" for k in range(12)) + " value"
        out.append(
            Example(
                f"{prefix}{i}", text, [Span(len(text) - 5, len(text), label)], {"region": region}
            )
        )
    return out


def test_stratified_order_is_deterministic_and_balanced():
    exs = _toy(60, "US", "a") + _toy(60, "GB", "b")
    o1 = [e.id for e in stratified_order(list(exs), seed=1)]
    o2 = [e.id for e in stratified_order(list(exs), seed=1)]
    assert o1 == o2
    first = stratified_order(list(exs), seed=1)[:30]
    n_us = sum(e.meta["region"] == "US" for e in first)
    assert 12 <= n_us <= 18  # every prefix is roughly half and half


@pytest.fixture
def cfg():
    return {
        "seed": 3,
        "holdout_regions": ["IN"],
        "sizes": {
            "train_pool": 40,
            "train_nested": [10, 20],
            "dev": 10,
            "test_id": 10,
            "test_holdout_regions": 5,
        },
        "minhash": {"num_perm": 64, "threshold": 0.8, "ngram": 3},
    }


def test_build_splits(cfg):
    train = _toy(30, "US", "t") + _toy(30, "GB", "u") + _toy(30, "IN", "v")
    val = _toy(20, "US", "d") + _toy(20, "IN", "h")
    # a validation item that copies a training template must be dropped
    pool_first = stratified_order([e for e in train if e.meta["region"] != "IN"], cfg["seed"])[0]
    val.append(Example("dup", pool_first.text, pool_first.spans, {"region": "US"}))

    splits, stats = build_splits(train, val, cfg)
    pool = splits["train_40"]
    assert len(pool) == 40
    assert splits["train_10"] == pool[:10] and splits["train_20"] == pool[:20]
    assert all(e.meta["region"] != "IN" for e in pool)
    assert all(e.meta["region"] == "IN" for e in splits["test_holdout_regions"])
    assert len(splits["test_holdout_regions"]) == 5
    assert not {e.id for e in splits["dev"]} & {e.id for e in splits["test_id"]}
    all_eval = splits["dev"] + splits["test_id"] + splits["test_holdout_regions"]
    assert "dup" not in {e.id for e in all_eval}
    dropped = sum(d["dropped_near_duplicate_of_train"] for d in stats["dedup"].values())
    assert dropped >= 1


def test_real_fixture_reads():
    exs = list(read_examples(FIX / "openpii_sample.jsonl"))
    assert len(exs) == 20


def test_training_windows_never_cut_a_span():
    from prepare_train_mix import windows

    text = " ".join(f"w{i}" for i in range(300)) + " Ann Lee"
    i = text.index("Ann Lee")
    ex = Example("d", text, [Span(i, i + 7, "GIVENNAME")])
    ws = windows(ex, max_chars=200)
    assert all(len(w.text) <= 200 for w in ws)
    for w in ws:
        for s in w.spans:
            assert w.text[s.start : s.end] == "Ann Lee"
    assert sum(len(w.spans) for w in ws) == 1  # the span lands in exactly one window
