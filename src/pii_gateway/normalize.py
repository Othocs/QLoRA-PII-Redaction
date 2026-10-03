"""Unicode NFKC, zero-width stripping and look-alike mapping before detection, with an offset map
back to the original text.

Obfuscated input ("j​ohn@example.com", full-width digits, Cyrillic "о" inside a phone
number) defeats both regexes and the model. Detection runs on the normalised text; spans
are mapped back through `Normalized.to_original` so redaction edits the original bytes.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from pii_gateway.spans import Span

ZERO_WIDTH = {"​", "‌", "‍", "⁠", "﻿", "­"}
# Characters that look like ASCII digits or letters in PII and survive NFKC.
LOOKALIKES = {
    "о": "o", "О": "O", "а": "a", "е": "e", "р": "p", "с": "c", "х": "x", "у": "y",  # Cyrillic
    "Ο": "O", "ο": "o", "Ι": "I", "ι": "i",  # Greek
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ",
}  # fmt: skip


@dataclass
class Normalized:
    text: str
    origin: list[int]  # origin[i] = index in the original text of normalised char i
    original_len: int

    def to_original(self, start: int, end: int) -> tuple[int, int]:
        """Map a [start, end) range of the normalised text back to the original."""
        if start >= end:
            return start, start
        # an original char that NFKC expanded to several (e.g. "½") maps whole
        return self.origin[start], self.origin[end - 1] + 1


def normalize(text: str) -> Normalized:
    out: list[str] = []
    origin: list[int] = []
    for i, ch in enumerate(text):
        if ch in ZERO_WIDTH:
            continue
        mapped = LOOKALIKES.get(ch)
        piece = mapped if mapped is not None else unicodedata.normalize("NFKC", ch)
        for c in piece:
            out.append(c)
            origin.append(i)
    return Normalized("".join(out), origin, len(text))


def spans_to_original(norm: Normalized, original: str, spans: list[Span]) -> list[Span]:
    out = []
    for s in spans:
        a, b = norm.to_original(s.start, s.end)
        out.append(Span(a, b, s.label, original[a:b], s.source, s.score))
    return out
