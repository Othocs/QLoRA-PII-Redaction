"""Deterministic validators: email, card numbers (Luhn), IBAN (mod-97), phone numbers
(libphonenumber), IPv4/IPv6.

The gateway's high-precision fast path. Every hit is checked, not just pattern-matched: a
16-digit gift-card number fails Luhn and is not reported as a card, an order number is
not a valid phone number in any of the supported regions. Spelled-out or line-split
values are left to the LLM detector; the merge (pii_gateway.merge) takes the union.
"""

from __future__ import annotations

import ipaddress
import re

from pii_gateway.spans import Span

SOURCE = "validator"

EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}(?![\w-])"
)
# 13-19 digits, optionally grouped by single spaces, hyphens or dots.
CARD = re.compile(r"(?<![\w.-])\d(?:[ .-]?\d){12,18}(?![\w-])")
IBAN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}(?![A-Za-z0-9])")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")

# Registered IBAN lengths (SWIFT IBAN registry), for the countries most likely in our traffic
# plus the rest of SEPA.
IBAN_LENGTHS = {
    "AD": 24, "AT": 20, "BE": 16, "BG": 22, "CH": 21, "CY": 28, "CZ": 24, "DE": 22, "DK": 18,
    "EE": 20, "ES": 24, "FI": 18, "FR": 27, "GB": 22, "GI": 23, "GR": 27, "HR": 21, "HU": 28,
    "IE": 22, "IS": 26, "IT": 27, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MT": 31,
    "NL": 18, "NO": 15, "PL": 28, "PT": 25, "RO": 24, "SE": 24, "SI": 19, "SK": 24, "SM": 27,
    "VA": 22, "AE": 23, "SA": 24, "TR": 26, "IL": 23, "BR": 29, "MU": 30, "PK": 24,
}  # fmt: skip
PHONE_REGIONS = ("US", "GB", "CA", "IN")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, c in enumerate(reversed(digits)):
        d = int(c) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def iban_ok(iban: str) -> bool:
    s = iban.replace(" ", "").upper()
    if IBAN_LENGTHS.get(s[:2]) != len(s):
        return False
    rearranged = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


def find_emails(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), "EMAIL", m.group(), SOURCE) for m in EMAIL.finditer(text)]


def find_cards(text: str) -> list[Span]:
    out = []
    for m in CARD.finditer(text):
        seps = set(re.sub(r"\d", "", m.group()))
        digits = re.sub(r"\D", "", m.group())
        if len(seps) <= 1 and 13 <= len(digits) <= 19 and luhn_ok(digits):
            out.append(Span(m.start(), m.end(), "CREDITCARDNUMBER", m.group(), SOURCE))
    return out


def find_ibans(text: str) -> list[Span]:
    return [
        Span(m.start(), m.end(), "IBAN", m.group(), SOURCE)
        for m in IBAN.finditer(text)
        if iban_ok(m.group())
    ]


def find_ips(text: str) -> list[Span]:
    out = []
    for pat in (IPV4, IPV6):
        for m in pat.finditer(text):
            try:
                ipaddress.ip_address(m.group())
            except ValueError:
                continue
            out.append(Span(m.start(), m.end(), "IPADDRESS", m.group(), SOURCE))
    return out


def find_phones(text: str, regions=PHONE_REGIONS) -> list[Span]:
    """Valid numbers for any of `regions` (libphonenumber, VALID leniency)."""
    import phonenumbers

    seen: dict[tuple[int, int], Span] = {}
    for region in regions:
        for m in phonenumbers.PhoneNumberMatcher(text, region, phonenumbers.Leniency.VALID):
            seen.setdefault(
                (m.start, m.end), Span(m.start, m.end, "TELEPHONENUM", m.raw_string, SOURCE)
            )
    return list(seen.values())


class ValidatorDetector:
    """All validators as one Detector. Overlaps resolve to the longer span."""

    name = "validators"

    def __init__(self, phone_regions=PHONE_REGIONS) -> None:
        self.phone_regions = tuple(phone_regions)

    def detect(self, text: str) -> list[Span]:
        spans = find_emails(text) + find_ibans(text) + find_cards(text) + find_ips(text)
        taken = [(s.start, s.end) for s in spans]
        for p in find_phones(text, self.phone_regions):  # a card is never also a phone
            if not any(p.start < b and a < p.end for a, b in taken):
                spans.append(p)
        return sorted(spans, key=lambda s: (s.start, s.end))
