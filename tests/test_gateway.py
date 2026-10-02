"""Gateway: validators, normalisation, merge, policy, vault, API (stub detector, no model)."""

import base64
import logging
import os

import pytest

from pii_gateway.detectors import validators as V
from pii_gateway.merge import recall_first_union
from pii_gateway.normalize import normalize, spans_to_original
from pii_gateway.policy import Policy, apply
from pii_gateway.spans import Span
from pii_gateway.vault import Vault, VaultError

KEY = os.urandom(32)


def labels(spans):
    return [(s.label, s.text) for s in spans]


# ---------------------------------------------------------------- validators


def test_cards_need_luhn():
    text = "card 4111 1111 1111 1111, gift card 6034 9321 0075 5510, amex 3782-822463-10005"
    assert labels(V.find_cards(text)) == [
        ("CREDITCARDNUMBER", "4111 1111 1111 1111"),
        ("CREDITCARDNUMBER", "3782-822463-10005"),
    ]
    assert V.find_cards("tracking 1Z999AA10123456784 and order 123-4567890-1234567") == []
    assert V.find_cards("mixed separators 4111-1111 1111.1111") == []


def test_iban_mod97_and_length():
    assert labels(V.find_ibans("pay GB82 WEST 1234 5698 7654 32 today")) == [
        ("IBAN", "GB82 WEST 1234 5698 7654 32")]  # fmt: skip
    assert V.find_ibans("GB82 WEST 1234 5698 7654 33") == []  # checksum off by one
    assert labels(V.find_ibans("DE89370400440532013000")) == [("IBAN", "DE89370400440532013000")]


def test_emails_ips_phones():
    assert labels(V.find_emails("mail j.doe+x@example.co.uk, not a@b")) == [
        ("EMAIL", "j.doe+x@example.co.uk")]  # fmt: skip
    ips = labels(V.find_ips("from 192.0.2.17 and 2001:db8::1, version 1.2.3, 999.1.1.1"))
    assert ips == [("IPADDRESS", "192.0.2.17"), ("IPADDRESS", "2001:db8::1")]
    phones = V.find_phones("call (415) 555-2671 or 020 7946 0958; order 88213409")
    assert {t for _, t in labels(phones)} == {"(415) 555-2671", "020 7946 0958"}


def test_validator_detector_no_double_counting():
    det = V.ValidatorDetector()
    spans = det.detect("card 4111111111111111 and mail a.b@example.com")
    assert [s.label for s in spans] == ["CREDITCARDNUMBER", "EMAIL"]
    assert all(s.source == "validator" for s in spans)


# ---------------------------------------------------------------- normalisation


def test_normalize_maps_offsets_back():
    original = "mail: j​o​hn@example.com, ph ０２０ ７９４６ ０９５８"
    norm = normalize(original)
    assert "john@example.com" in norm.text and "020 7946 0958" in norm.text
    found = V.find_emails(norm.text)
    back = spans_to_original(norm, original, found)
    assert back[0].text == "j​o​hn@example.com"
    assert original[back[0].start : back[0].end] == back[0].text


def test_cyrillic_lookalike_in_email():
    original = "write to jоhn@example.com"  # Cyrillic о
    norm = normalize(original)
    back = spans_to_original(norm, original, V.find_emails(norm.text))
    assert back[0].text == "jоhn@example.com"


# ---------------------------------------------------------------- merge


def test_recall_first_union():
    text = "Call 07700 900 123 or mail ann@example.com, Ann Lee"
    llm = [Span(5, 15, "TELEPHONENUM", source="llm"), Span(44, 47, "GIVENNAME", source="llm"),
           Span(48, 51, "SURNAME", source="llm")]  # fmt: skip
    val = [Span(5, 18, "TELEPHONENUM", source="validator"),
           Span(27, 42, "EMAIL", source="validator")]  # fmt: skip
    out = recall_first_union(text, llm, val)
    assert labels(out) == [("TELEPHONENUM", "07700 900 123"), ("EMAIL", "ann@example.com"),
                           ("GIVENNAME", "Ann"), ("SURNAME", "Lee")]  # fmt: skip
    assert out[0].source == "merged" and out[1].source == "validator"


def test_merge_prefers_validator_label():
    text = "id 4111111111111111"
    out = recall_first_union(text, [Span(3, 19, "PASSPORTNUM", source="llm")],
                             [Span(3, 19, "CREDITCARDNUMBER", source="validator")])  # fmt: skip
    assert labels(out) == [("CREDITCARDNUMBER", "4111111111111111")]


# ---------------------------------------------------------------- policy and vault


def test_policy_actions():
    text = "Ann Lee from Leeds, 4111111111111111"
    spans = [Span(0, 3, "GIVENNAME"), Span(4, 7, "SURNAME"), Span(13, 18, "CITY"),
             Span(20, 36, "CREDITCARDNUMBER")]  # fmt: skip
    support = Policy.load("support")
    assert (
        apply(text, spans, support).text == "[GIVENNAME] [SURNAME] from Leeds, [CREDITCARDNUMBER]"
    )
    vault = Vault(KEY)
    pol = Policy("t", "pseudonymize", {"CREDITCARDNUMBER": "hash", "CITY": "keep"})
    r = apply(text, spans, pol, vault=vault, tenant="t1", conversation="c1", hash_key=b"k" * 32)
    assert r.text.startswith("<GIVENNAME_1> <SURNAME_1> from Leeds, <CREDITCARDNUMBER:")
    assert all("text" not in e for e in r.entities)
    assert [e["action"] for e in r.entities] == ["pseudonymize", "pseudonymize", "keep", "hash"]
    with pytest.raises(ValueError):
        Policy("bad", "shred")
    with pytest.raises(ValueError):
        Policy.load("../secrets")


def test_vault_roundtrip_stability_scope_ttl_tamper():
    now = [1000.0]
    v = Vault(KEY, ttl_s=60, clock=lambda: now[0])
    a = v.token("t", "c", "GIVENNAME", "Ann")
    assert v.token("t", "c", "GIVENNAME", "Ann") == a == "<GIVENNAME_1>"
    assert v.token("t", "c", "GIVENNAME", "Bob") == "<GIVENNAME_2>"
    assert v.restore("t", "c", f"hi {a}, <GIVENNAME_9>") == ("hi Ann, <GIVENNAME_9>", 1)
    with pytest.raises(VaultError):
        v.reveal("t", "other-conversation", a)  # scopes don't leak
    blob = v._scopes[("t", "c")].by_token[a]
    v._scopes[("t", "c")].by_token[a] = blob[:-1] + bytes([blob[-1] ^ 1])
    with pytest.raises(VaultError):
        v.reveal("t", "c", a)  # tampered ciphertext
    now[0] += 61
    with pytest.raises(VaultError):
        v.reveal("t", "c", "<GIVENNAME_2>")  # expired
    with pytest.raises(ValueError):
        Vault(os.urandom(16))  # AES-256 only


# ---------------------------------------------------------------- API


class StubLLM:
    name = "stub"

    def __init__(self, fail=False):
        self.fail = fail

    def detect(self, text):
        if self.fail:
            raise RuntimeError("model down")
        i = text.find("Priya")
        return [Span(i, i + 5, "GIVENNAME", "Priya", "llm")] if i >= 0 else []


@pytest.fixture
def client_factory():
    from fastapi.testclient import TestClient

    from pii_gateway.api import create_app

    def make(detector=None, restore_key="r-key", api_key=None):
        app = create_app(detector=detector, vault=Vault(KEY), use_env=False)
        app.state.restore_key, app.state.api_key = restore_key, api_key
        return TestClient(app)

    return make


def test_redact_and_restore(client_factory, caplog):
    c = client_factory(StubLLM())
    secret = "Priya, card 4111 1111 1111 1111, mail priya@example.com"
    with caplog.at_level(logging.DEBUG, logger="pii_gateway"):
        r = c.post("/redact", json={"text": secret, "policy": "analytics",
                                    "conversation_id": "conv-1"})  # fmt: skip
    assert r.status_code == 200
    body = r.json()
    assert body["redacted"] == "<GIVENNAME_1>, card <CREDITCARDNUMBER_1>, mail <EMAIL_1>"
    for value in ("Priya", "4111", "priya@example.com"):
        assert value not in caplog.text  # never logged
        assert value not in str(body["entities"])
    denied = c.post("/restore", json={"text": body["redacted"], "conversation_id": "conv-1"})
    assert denied.status_code == 403
    ok = c.post("/restore", json={"text": body["redacted"], "conversation_id": "conv-1"},
                headers={"X-Restore-Key": "r-key"})  # fmt: skip
    assert ok.json()["restored"] == secret


def test_fail_closed_and_auth(client_factory):
    c = client_factory(StubLLM(fail=True))
    r = c.post("/redact", json={"text": "Priya 4111 1111 1111 1111"})
    assert r.status_code == 503 and "Priya" not in r.text and "4111" not in r.text
    c = client_factory(api_key="a-key")
    assert c.post("/redact", json={"text": "hi"}).status_code == 401
    assert c.post("/redact", json={"text": "hi"}, headers={"X-API-Key": "a-key"}).status_code == 200
    assert c.post("/redact", json={"text": "hi", "policy": "nope"},
                  headers={"X-API-Key": "a-key"}).status_code == 400  # fmt: skip
    c = client_factory(restore_key=None)
    assert c.post("/restore", json={"text": "x", "conversation_id": "c"},
                  headers={"X-Restore-Key": "anything"}).status_code == 403  # fmt: skip
    assert c.get("/health").json()["status"] == "ok"


def test_vault_from_env(monkeypatch):
    monkeypatch.setenv("PII_VAULT_KEY", base64.b64encode(KEY).decode())
    assert Vault.from_env().token("t", "c", "EMAIL", "a@b.co") == "<EMAIL_1>"
