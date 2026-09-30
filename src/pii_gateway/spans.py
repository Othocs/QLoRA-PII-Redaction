"""Span type, label set and the JSONL example format shared by data, detectors and eval."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

# The 19 OpenPII labels, in the dataset's own order of popularity.
OPENPII_LABELS: tuple[str, ...] = (
    "GIVENNAME",
    "SURNAME",
    "TITLE",
    "AGE",
    "SEX",
    "GENDER",
    "DATE",
    "EMAIL",
    "TELEPHONENUM",
    "STREET",
    "BUILDINGNUM",
    "CITY",
    "ZIPCODE",
    "IDCARDNUM",
    "PASSPORTNUM",
    "DRIVERLICENSENUM",
    "SOCIALNUM",
    "TAXNUM",
    "CREDITCARDNUMBER",
)

# Labels only the deterministic validators produce (no OpenPII training data).
VALIDATOR_LABELS: tuple[str, ...] = ("IBAN", "IPADDRESS")

# Buckets for baselines whose label sets are coarser than ours. They count for
# leakage and over-redaction but can never match a gold label in strict F1.
NAME = "NAME"  # e.g. Presidio PERSON covers GIVENNAME + SURNAME
OTHER = "OTHER"  # any prediction we could not map

LABELS: tuple[str, ...] = OPENPII_LABELS + VALIDATOR_LABELS


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    label: str
    text: str = ""
    source: str = "gold"
    score: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span offsets [{self.start}, {self.end})")

    def to_dict(self, *, full: bool = False) -> dict:
        if full:
            return asdict(self)
        return {"start": self.start, "end": self.end, "label": self.label}

    @classmethod
    def from_dict(cls, d: dict, text: str | None = None, source: str = "gold") -> Span:
        s, e = int(d["start"]), int(d["end"])
        return cls(
            start=s,
            end=e,
            label=d["label"],
            text=d.get("text") or (text[s:e] if text is not None else ""),
            source=d.get("source", source),
            score=d.get("score"),
        )


@dataclass(slots=True)
class Example:
    id: str
    text: str
    spans: list[Span]
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "text": self.text,
                "spans": [s.to_dict() for s in self.spans],
                "meta": self.meta,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, line: str) -> Example:
        d = json.loads(line)
        text = d["text"]
        return cls(
            id=str(d["id"]),
            text=text,
            spans=[Span.from_dict(s, text) for s in d["spans"]],
            meta=d.get("meta", {}),
        )


def read_examples(path: str | Path, limit: int | None = None) -> Iterator[Example]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield Example.from_json(line)


def write_examples(path: str | Path, examples: Iterable[Example]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")
            n += 1
    return n


def char_mask(text_len: int, spans: Iterable[Span]) -> bytearray:
    """1 for every character covered by at least one span, else 0."""
    mask = bytearray(text_len)
    for s in spans:
        a, b = max(0, s.start), min(text_len, s.end)
        if b > a:
            mask[a:b] = b"\x01" * (b - a)
    return mask


def validate_spans(text: str, spans: Iterable[Span]) -> list[str]:
    """Return a list of problems (empty if every span is in range and matches its text)."""
    errors = []
    for s in spans:
        if s.end > len(text):
            errors.append(f"{s.label} [{s.start},{s.end}) out of range (len={len(text)})")
        elif s.text and text[s.start : s.end] != s.text:
            errors.append(
                f"{s.label} [{s.start},{s.end}) text {text[s.start : s.end]!r} != {s.text!r}"
            )
    return errors


def dedupe_spans(spans: Iterable[Span]) -> list[Span]:
    """Drop exact (start, end, label) duplicates and sort by position."""
    seen: dict[tuple[int, int, str], Span] = {}
    for s in spans:
        key = (s.start, s.end, s.label)
        if key not in seen or (s.score or 0) > (seen[key].score or 0):
            seen[key] = s
    return sorted(seen.values(), key=lambda s: (s.start, s.end, s.label))
