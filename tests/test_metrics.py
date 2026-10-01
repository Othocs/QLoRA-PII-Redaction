import pytest

from eval.metrics import aggregate, score_doc
from pii_gateway.spans import Example, Span

TEXT = "Hi Camille Laurent, call 06 12 34 56 78."
#       0123456789012345678901234567890123456789
FIRST = Span(3, 10, "GIVENNAME")  # Camille
LAST = Span(11, 18, "SURNAME")  # Laurent
PHONE = Span(25, 39, "TELEPHONENUM")  # 06 12 34 56 78
GOLD = [FIRST, LAST, PHONE]
GOLD_CHARS = 7 + 7 + 14


def run(pred, gold=GOLD, text=TEXT):
    return aggregate([Example("d1", text, gold, {"region": "US"})], {"d1": pred})


def test_offsets_in_fixture():
    assert TEXT[FIRST.start : FIRST.end] == "Camille"
    assert TEXT[LAST.start : LAST.end] == "Laurent"
    assert TEXT[PHONE.start : PHONE.end] == "06 12 34 56 78"


def test_perfect_match():
    r = run(GOLD)
    assert r["leakage_chars"] == 0
    assert r["leakage_docs"] == 0
    assert r["over_redaction"] == 0
    assert r["strict"]["f1"] == 1
    assert r["partial"]["f1"] == 1


def test_total_miss():
    r = run([])
    assert r["leakage_chars"] == 1
    assert r["leakage_docs"] == 1
    assert r["over_redaction"] == 0
    assert r["strict"] == {"precision": 1.0, "recall": 0.0, "f1": 0.0}
    assert r["partial"]["recall"] == 0


def test_partial_overlap_phone():
    # Only "12 34 56 78" (28..39) predicted: "06 " leaks (3 chars).
    r = run([FIRST, LAST, Span(28, 39, "TELEPHONENUM")])
    assert r["leakage_chars"] == pytest.approx(3 / GOLD_CHARS)
    assert r["leakage_docs"] == 1  # a value partly survives
    assert r["strict"]["recall"] == pytest.approx(2 / 3)
    assert r["partial"]["recall"] == 1
    assert r["per_label"]["TELEPHONENUM"]["leakage_chars"] == pytest.approx(3 / 14)


def test_over_masking():
    # Masking "Hi" (0..2) as well: 2 extra chars out of GOLD_CHARS + 2 masked.
    r = run([*GOLD, Span(0, 2, "OTHER")])
    assert r["leakage_chars"] == 0
    assert r["over_redaction"] == pytest.approx(2 / (GOLD_CHARS + 2))
    assert r["strict"]["precision"] == pytest.approx(3 / 4)
    assert r["partial"]["precision"] == pytest.approx(3 / 4)
    assert r["per_label"]["OTHER"]["strict"]["precision"] == 0


def test_wrong_label_full_coverage():
    # Presidio-style: one NAME span over "Camille Laurent" and the phone right.
    r = run([Span(3, 18, "NAME"), PHONE])
    assert r["leakage_chars"] == 0
    # The space between first and last name (10..11) isn't gold PII.
    assert r["over_redaction"] == pytest.approx(1 / (GOLD_CHARS + 1))
    assert r["strict"]["recall"] == pytest.approx(1 / 3)
    assert r["partial"]["recall"] == 1


def test_wrong_label_same_boundaries():
    r = run([Span(3, 10, "SURNAME"), LAST, PHONE])
    assert r["leakage_chars"] == 0
    assert r["strict"]["recall"] == pytest.approx(2 / 3)
    assert r["per_label"]["GIVENNAME"]["strict"]["recall"] == 0


def test_empty_doc_and_missing_prediction():
    r = aggregate(
        [Example("a", "nothing here", [], {}), Example("b", TEXT, GOLD, {})],
        {"a": []},  # "b" has no prediction entry -> counts as predicting nothing
    )
    assert r["docs"] == 2
    assert r["leakage_chars"] == 1
    assert r["leakage_docs"] == 0.5


def test_all_empty():
    r = aggregate([Example("a", "nothing", [], {})], {"a": []})
    assert r["leakage_chars"] == 0
    assert r["strict"]["f1"] == 1


def test_per_region_groups():
    exs = [
        Example("us", TEXT, GOLD, {"region": "US"}),
        Example("gb", TEXT, GOLD, {"region": "GB"}),
    ]
    r = aggregate(exs, {"us": GOLD, "gb": []})
    assert r["per_region"]["US"]["leakage_chars"] == 0
    assert r["per_region"]["GB"]["leakage_chars"] == 1
    assert r["leakage_chars"] == 0.5


def test_duplicate_predictions_dont_double_count():
    d = score_doc(TEXT, GOLD, [*GOLD, FIRST])
    assert d.n_pred == 3
    assert d.strict_tp == 3
    assert d.by_label["GIVENNAME"]["n_pred"] == 1


def test_ignore_regions():
    # "Acme Ltd" (0..8) is out of scope: not a leak if missed, not over-redaction if masked.
    text = "Acme Ltd hired Ann Lee."
    gold = [Span(0, 8, "IGNORE"), Span(15, 18, "GIVENNAME"), Span(19, 22, "SURNAME")]
    missed = aggregate([Example("a", text, gold)], {"a": [Span(15, 18, "GIVENNAME")]})
    assert missed["n_gold_spans"] == 2
    assert missed["leakage_chars"] == pytest.approx(3 / 6)
    masked = aggregate(
        [Example("a", text, gold)],
        {"a": [Span(0, 8, "OTHER"), Span(15, 18, "GIVENNAME"), Span(19, 22, "SURNAME")]},
    )
    assert masked["leakage_chars"] == 0
    assert masked["over_redaction"] == 0
    assert masked["n_pred_spans"] == 2  # the prediction inside the ignored region is dropped
    assert masked["strict"]["f1"] == 1
    # a prediction straddling an ignored region and plain text still counts, minus the ignored part
    spill = aggregate([Example("a", text, gold)], {"a": [Span(0, 14, "OTHER")]})
    assert spill["over_redaction"] == 1.0  # " hired" is over-redacted; "Acme Ltd" is not counted
    assert spill["n_pred_spans"] == 1


def test_groups_by_every_present_key():
    exs = [
        Example("fr", TEXT, GOLD, {"language": "fr"}),
        Example("de", TEXT, GOLD, {"language": "de", "region": "DE"}),
    ]
    r = aggregate(exs, {"fr": GOLD, "de": []})
    assert r["per_language"]["fr"]["leakage_chars"] == 0
    assert r["per_language"]["de"]["leakage_chars"] == 1
    assert list(r["per_region"]) == ["DE"]  # only examples that have the key
    assert "partial" in r["per_language"]["fr"]
