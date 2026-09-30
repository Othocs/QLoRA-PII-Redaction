import pytest

from pii_gateway.detectors.base import (
    chunk_text,
    detect_chunked,
    drop_contained,
    merge_overlapping,
    trim_span,
)
from pii_gateway.spans import Example, Span, char_mask, dedupe_spans, validate_spans


def test_span_rejects_bad_offsets():
    with pytest.raises(ValueError):
        Span(5, 5, "EMAIL")
    with pytest.raises(ValueError):
        Span(-1, 3, "EMAIL")


def test_validate_spans():
    text = "Call Ann on 555-0100"
    assert validate_spans(text, [Span(5, 8, "GIVENNAME", "Ann")]) == []
    assert validate_spans(text, [Span(5, 8, "GIVENNAME", "Bob")])
    assert validate_spans(text, [Span(5, 99, "GIVENNAME")])


def test_example_roundtrip():
    ex = Example("x", "Ann Lee", [Span(0, 3, "GIVENNAME", "Ann")], {"region": "GB"})
    back = Example.from_json(ex.to_json())
    assert back.spans[0].text == "Ann"  # text is recovered from offsets
    assert back.meta == {"region": "GB"}


def test_char_mask_and_dedupe():
    spans = [Span(0, 3, "A"), Span(2, 5, "B"), Span(0, 3, "A")]
    assert list(char_mask(6, spans)) == [1, 1, 1, 1, 1, 0]
    assert len(dedupe_spans(spans)) == 2


def test_trim_span():
    assert trim_span("  ab  ", 0, 6) == (2, 4)


def test_chunk_text_covers_everything_with_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert chunks[0][0] == 0
    assert all(len(c) <= 300 for _, c in chunks)
    assert all(text[off : off + len(c)] == c for off, c in chunks)
    covered = set()
    for off, c in chunks:
        covered.update(range(off, off + len(c)))
    assert covered >= set(range(len(text))) - {i for i, ch in enumerate(text) if ch == " "}
    for (o1, c1), (o2, _) in zip(chunks, chunks[1:], strict=False):
        assert o2 < o1 + len(c1)  # consecutive windows overlap


def test_detect_chunked_maps_offsets_back():
    text = ("filler " * 60) + "secret@example.com " + ("filler " * 60)
    target = text.index("secret@example.com")

    def fake(chunk):
        i = chunk.find("secret@example.com")
        return [] if i == -1 else [Span(i, i + 18, "EMAIL", source="fake")]

    spans = detect_chunked(text, fake, max_chars=200, overlap=40)
    assert [(s.start, s.end, s.text) for s in spans] == [
        (target, target + 18, "secret@example.com")
    ]


def test_merge_overlapping_same_label_only():
    out = merge_overlapping([Span(0, 5, "A"), Span(3, 8, "A"), Span(3, 8, "B")])
    assert [(s.start, s.end, s.label) for s in out] == [(0, 8, "A"), (3, 8, "B")]


def test_drop_contained():
    out = drop_contained([Span(0, 20, "EMAIL"), Span(8, 20, "OTHER"), Span(25, 30, "DATE")])
    assert [(s.start, s.end) for s in out] == [(0, 20), (25, 30)]
