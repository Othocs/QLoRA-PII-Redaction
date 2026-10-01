"""Prompt/answer format, output parsing and span alignment (no model needed)."""

import json

from pii_gateway.detectors.llm import (
    MODEL_LABELS,
    align_spans,
    build_messages,
    format_completion,
    output_schema,
    parse_output,
)
from pii_gateway.spans import Span

TEXT = "Dear Ann Lee, call Ann on 555-0100. Ann"


def test_completion_lists_spans_in_order_with_repeats():
    spans = [Span(26, 34, "TELEPHONENUM"), Span(5, 8, "GIVENNAME"), Span(19, 22, "GIVENNAME")]
    out = json.loads(format_completion(TEXT, spans))
    assert out == [
        {"label": "GIVENNAME", "text": "Ann"},
        {"label": "GIVENNAME", "text": "Ann"},
        {"label": "TELEPHONENUM", "text": "555-0100"},
    ]


def test_empty_completion():
    assert format_completion("no pii", []) == "[]"


def test_prompt_wraps_text_and_lists_labels():
    msg = build_messages("hello")[0]["content"]
    assert "<text>\nhello\n</text>" in msg
    assert all(lab in msg for lab in MODEL_LABELS)


def test_parse_valid_and_broken_json():
    p = parse_output('[{"label": "EMAIL", "text": "a@b.c"}]')
    assert p.valid_json and p.items == [{"label": "EMAIL", "text": "a@b.c"}]
    p = parse_output('```json\n[{"label": "EMAIL", "text": "a@b.c"}]\n```')
    assert p.valid_json
    # truncated output: salvage the complete items
    p = parse_output('[{"label": "EMAIL", "text": "a@b.c"}, {"label": "GIVENNAME", "text": "An')
    assert not p.valid_json and p.items == [{"label": "EMAIL", "text": "a@b.c"}]
    p = parse_output('[{"label": "SURNAME", "text": "O\\"Neil"}, {"label"')
    assert p.items == [{"label": "SURNAME", "text": 'O"Neil'}]


def test_align_repeated_values_left_to_right():
    items = [{"label": "GIVENNAME", "text": "Ann"}] * 3
    spans, dropped = align_spans(TEXT, items)
    assert dropped == 0
    assert [s.start for s in spans] == [5, 19, 36]


def test_align_out_of_order_and_missing():
    items = [
        {"label": "TELEPHONENUM", "text": "555-0100"},
        {"label": "GIVENNAME", "text": "Ann"},  # searched from the cursor: the "Ann" after it
        {"label": "SURNAME", "text": "Smith"},  # not in text: dropped
        {"label": "NOTALABEL", "text": "Lee"},  # unknown label: dropped
        {"label": "SURNAME", "text": ""},  # empty: dropped
    ]
    spans, dropped = align_spans(TEXT, items)
    assert dropped == 3
    assert [(s.label, s.start, s.end) for s in spans] == [
        ("TELEPHONENUM", 26, 34),
        ("GIVENNAME", 36, 39),
    ]


def test_align_falls_back_to_earlier_occurrence():
    items = [{"label": "TELEPHONENUM", "text": "555-0100"}, {"label": "SURNAME", "text": "Lee"}]
    spans, dropped = align_spans(TEXT, items)
    assert dropped == 0
    assert [(s.label, s.start) for s in spans] == [("SURNAME", 9), ("TELEPHONENUM", 26)]


def test_align_never_reuses_an_occurrence():
    spans, dropped = align_spans("Ann and Ann", [{"label": "GIVENNAME", "text": "Ann"}] * 3)
    assert len(spans) == 2 and dropped == 1


def test_schema_restricts_labels():
    schema = output_schema()
    assert schema["items"]["properties"]["label"]["enum"] == list(MODEL_LABELS)


def test_fuzzy_alignment_recovers_case_whitespace_and_quotes():
    text = "Contact  John\nSmith at JOHN@EXAMPLE.COM or 'Acme Ltd'."
    items = [
        {"label": "GIVENNAME", "text": "John Smith"},  # newline/double space in the source
        {"label": "EMAIL", "text": "john@example.com"},  # case differs
        {"label": "SURNAME", "text": '"Smith"'},  # model added quotes
    ]
    strict, n_strict = align_spans(text, items, fuzzy=False)
    assert n_strict == 3 and strict == []
    drops: list[dict] = []
    spans, n = align_spans(text, items, drops=drops)
    got = {(s.label, text[s.start : s.end]) for s in spans}
    assert ("GIVENNAME", "John\nSmith") in got  # span covers the source text, not the model's copy
    assert ("EMAIL", "JOHN@EXAMPLE.COM") in got
    assert n == 1 and drops == [{"label": "SURNAME", "text": '"Smith"', "reason": "not_found"}]


def test_drop_reasons():
    drops: list[dict] = []
    align_spans(
        "abc", [{"label": "X", "text": "abc"}, {"label": "EMAIL", "text": " "}], drops=drops
    )
    assert [d["reason"] for d in drops] == ["unknown_label", "empty"]
