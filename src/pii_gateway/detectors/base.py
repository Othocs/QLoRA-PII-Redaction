"""Detector protocol and helpers for long inputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from pii_gateway.spans import Span


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, text: str) -> list[Span]: ...


def chunk_text(text: str, max_chars: int, overlap: int) -> list[tuple[int, str]]:
    """Split text into (offset, chunk) windows of at most max_chars.

    Cuts fall on whitespace when possible, and consecutive windows share about
    `overlap` characters so an entity cut by one window is whole in the next.
    """
    if max_chars <= overlap:
        raise ValueError("max_chars must be larger than overlap")
    if len(text) <= max_chars:
        return [(0, text)]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = text.rfind(" ", start + overlap + 1, end)
            if cut == -1:
                cut = max(text.rfind("\n", start + overlap + 1, end), -1)
            if cut != -1:
                end = cut
        chunks.append((start, text[start:end]))
        if end >= len(text):
            break
        nxt = end - overlap
        ws = text.find(" ", nxt, end)
        start = ws + 1 if ws != -1 else nxt
    return chunks


def merge_overlapping(spans: list[Span]) -> list[Span]:
    """Union overlapping spans that share a label (e.g. the same entity seen from two chunks)."""
    out: list[Span] = []
    for s in sorted(spans, key=lambda s: (s.label, s.start, s.end)):
        if out and out[-1].label == s.label and s.start < out[-1].end:
            prev = out[-1]
            if s.end > prev.end:
                out[-1] = Span(
                    prev.start,
                    s.end,
                    prev.label,
                    source=prev.source,
                    score=max(prev.score or 0, s.score or 0),
                )
        else:
            out.append(s)
    return sorted(out, key=lambda s: (s.start, s.end, s.label))


def drop_contained(spans: list[Span]) -> list[Span]:
    """Drop spans that lie entirely inside another (longer) span, whatever their label.

    Mirrors what Presidio's anonymizer does to overlapping analyzer results, so
    nested hits (a URL inside an email) don't count as extra predictions.
    """
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    out: list[Span] = []
    for s in ordered:
        if any(o.start <= s.start and s.end <= o.end and o != s for o in out):
            continue
        out.append(s)
    return sorted(out, key=lambda s: (s.start, s.end, s.label))


def detect_chunked(
    text: str,
    fn: Callable[[str], list[Span]],
    max_chars: int,
    overlap: int,
) -> list[Span]:
    """Run `fn` on each chunk, shift offsets back to `text`, merge boundary duplicates."""
    spans = []
    for off, chunk in chunk_text(text, max_chars, overlap):
        for s in fn(chunk):
            spans.append(Span(s.start + off, s.end + off, s.label, source=s.source, score=s.score))
    spans = merge_overlapping(spans)
    return [Span(s.start, s.end, s.label, text[s.start : s.end], s.source, s.score) for s in spans]


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) so it doesn't begin or end on whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
