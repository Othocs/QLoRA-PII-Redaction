"""Milestone 1 targeted data: value renderers, message labelling/validation, mix, hygiene."""

import json
import random
import re
from pathlib import Path

import pytest
from synthetic import generate_targeted as gen
from synthetic import values as V


def test_phone_renderings_keep_the_digits():
    rng = random.Random(0)
    seen = set()
    for i in range(600):
        locale = ["US", "UK", "CA", "IN"][i % 4]
        state = rng.getstate()
        groups = V.phone_digits(rng, locale)
        rng.setstate(state)
        _, text, fmt = V.phone(rng, locale)
        seen.add(fmt)
        digits = "".join(groups)
        if fmt == "intl_spoken":
            cc = {"UK": "44", "IN": "91", "US": "1", "CA": "1"}[locale]
            digits = cc + (digits[1:] if locale == "UK" else digits)
        assert V.unspeak(text) == digits, (fmt, text)
    assert seen >= {"spoken", "split_lines", "unspaced", "mixed", "intl_spoken"}


def test_spoken_runs_and_oh():
    rng = random.Random(1)
    texts = {V.spoken("07700", rng) for _ in range(50)}
    assert any("double" in t for t in texts) and any("oh" in t for t in texts)
    assert all(V.unspeak(t) == "07700" for t in texts)


def test_id_and_card_formats():
    rng = random.Random(2)
    for _ in range(200):
        _, c, _ = V.card(rng)
        assert V.luhn_ok(re.sub(r"\D", "", c))
        _, s, fmt = V.ssn(rng)
        assert V.unspeak(s)[0] == "9" and len(V.unspeak(s)) == 9, (fmt, s)
        _, n, _ = V.nino(rng)
        assert re.fullmatch(r"qq\d{6}[a-d]|QQ( \d\d){3} [A-D]|QQ\d{6}[A-D]", n)
    assert V.number_words(92) == "ninety-two" and V.number_words(101) == "one hundred and one"


def _values(*pairs):
    return [{"label": lab, "text": t, "format": "x"} for lab, t in pairs]


def test_label_message_every_occurrence_longest_first():
    text = "Dr Dr Aiko Okoro here. Okoro again; call oh two oh, nine four six."
    vals = _values(("TITLE", "Dr Dr"), ("GIVENNAME", "Aiko"), ("SURNAME", "Okoro"),
                   ("TELEPHONENUM", "oh two oh, nine four six"))  # fmt: skip
    spans, why = gen.label_message(text, vals)
    assert why == ""
    assert [(s["label"], s["text"]) for s in spans] == [
        ("TITLE", "Dr Dr"), ("GIVENNAME", "Aiko"), ("SURNAME", "Okoro"), ("SURNAME", "Okoro"),
        ("TELEPHONENUM", "oh two oh, nine four six")]  # fmt: skip
    for s in spans:
        assert text[s["start"] : s["end"]] == s["text"]


def test_case_variants_are_labelled():
    vals = _values(("TITLE", "Fr."), ("GIVENNAME", "Aaliyah"), ("SURNAME", "Ballantyne"))
    rec, why = gen.validate("i'm fr. aaliyah ballantyne\n\nFr. Aaliyah Ballantyne", vals, [])
    assert why == "" and [s["text"] for s in rec["spans"]] == [
        "fr.", "aaliyah", "ballantyne", "Fr.", "Aaliyah", "Ballantyne"]  # fmt: skip


def test_validate_accepts_and_rejects():
    vals = _values(("GIVENNAME", "Saoirse"), ("SURNAME", "Kerrigan"), ("AGE", "34M"))
    ok = "cust Saoirse Kerrigan, 34M, gift card 6034 9321 0075 5512 has no balance"
    rec, why = gen.validate(ok, vals, ["6034 9321 0075 5512"])
    assert why == "" and rec["keep"] == ["6034 9321 0075 5512"]
    assert len(rec["spans"]) == 3
    cases = {
        "missing value AGE": "cust Saoirse Kerrigan has no balance",
        "unlabelled digit run": "cust Saoirse Kerrigan, 34M, call 07700 900 123",
        "unlabelled email": "cust Saoirse Kerrigan, 34M, mail s.k@example.com",
        "unlabelled title + name": "cust Saoirse Kerrigan, 34M, spoke to Mr Hollis",
        "length": "cust Saoirse Kerrigan, 34M " + "x" * 1200,
    }
    for reason, text in cases.items():
        assert gen.validate(text, vals, [])[1] == reason, text


def test_semantic_rejections():
    tel = _values(("GIVENNAME", "Saoirse"), ("TELEPHONENUM", "oh one one three"))
    assert gen.validate("Saoirse, please call us at oh one one three", tel, [])[1] == (
        "phone given as the company's"
    )
    assert gen.validate("Saoirse here, ring me on oh one one three", tel, [])[1] == ""
    dob = _values(("GIVENNAME", "Saoirse"), ("DATE", "9th Mar '09"))
    assert gen.validate("From: Saoirse\nSent: 9th Mar '09\nhi", dob, [])[1] == (
        "date used as a header date"
    )
    card = _values(
        ("GIVENNAME", "Saoirse"),
    )
    msg = "Saoirse: my card number 4777 5307 3765 3044 was declined"
    assert gen.validate(msg, card, ["4777 5307 3765 3044"])[1] == "look-alike presented as a card"
    msg = "Saoirse: my gift card 4777 5307 3765 3044 is empty"
    assert gen.validate(msg, card, ["4777 5307 3765 3044"])[1] == ""


def test_wordlike_names_and_bare_numbers_must_be_unique():
    vals = _values(("GIVENNAME", "Hope"), ("SURNAME", "Brennan"), ("AGE", "45"))
    assert gen.validate("Hope Brennan, 45, says hi", vals, [])[1] == ""
    assert gen.validate("Hope Brennan, 45, says I hope so", vals, [])[1] == "ambiguous GIVENNAME"
    assert gen.validate("Hope Brennan, 45, paid £45", vals, [])[1] == "ambiguous AGE"


def test_collect_parses_a_mocked_response():
    specs = gen.make_specs(7, 0)[:2]
    msgs = []
    for s in specs:
        text = "hi, " + " and ".join(v["text"] for v in s["values"]) + ". thanks"
        msgs.append({"id": s["id"], "text": text, "keep": []})
    resp = {"choices": [{"message": {"content": json.dumps({"messages": msgs})}}]}
    from collections import Counter

    reasons = Counter()
    out = gen.collect(resp, specs, reasons)
    assert len(out) + sum(reasons.values()) == 2
    for rec in out:
        assert rec["meta"]["source"] == "targeted"
        for s in rec["spans"]:
            assert rec["text"][s["start"] : s["end"]] == s["text"]


def test_specs_are_deterministic():
    assert gen.make_specs(2026, 5) == gen.make_specs(2026, 5)
    assert gen.make_specs(2026, 5) != gen.make_specs(2026, 6)


TARGETED = Path("data/synthetic/targeted_2k.jsonl")


@pytest.mark.skipif(not TARGETED.exists(), reason="targeted data not generated yet")
def test_targeted_file_is_valid_and_clean():
    recs = [json.loads(line) for line in TARGETED.read_text(encoding="utf-8").splitlines()]
    assert len({r["id"] for r in recs}) == len(recs)
    grams, vals = gen.eval_fingerprints()
    for r in recs:
        assert len(r["text"]) <= gen.MAX_CHARS
        assert not gen.contaminated(r, grams, vals), r["id"]
        prev = -1
        for s in r["spans"]:
            assert r["text"][s["start"] : s["end"]] == s["text"]
            assert s["start"] >= prev
            prev = s["end"]
