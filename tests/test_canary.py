"""Canary log-leak test: unique PII values pushed through every gateway path must never appear
in any log record, in any error response, or in what is sent upstream.

Every logger is captured at DEBUG (root, pii_gateway, uvicorn, httpx, fastapi). Pseudonym
tokens are expected in upstream traffic; raw values never are.
"""

import json
import logging
import os

import httpx
from fastapi.testclient import TestClient

from pii_gateway.api import MAX_CHARS, create_app
from pii_gateway.spans import Span
from pii_gateway.vault import Vault

CANARY_NAME = "Zephyrine"
CANARY_EMAIL = "zephyrine.canary.7f3a@example.com"
CANARY_CARD = "4000 0566 5566 5556"
CANARY_PHONE = "+44 20 7946 0321"
CANARIES = [CANARY_NAME, CANARY_EMAIL, CANARY_CARD, CANARY_PHONE, "7f3a"]
TEXT = f"Hi, {CANARY_NAME} here: card {CANARY_CARD}, mail {CANARY_EMAIL}, call {CANARY_PHONE}"


class CanaryStub:
    name = "stub"
    fail = False

    def detect(self, text):
        if self.fail:
            raise RuntimeError(f"model crashed on input of length {len(text)}")
        i = text.find(CANARY_NAME)
        return [Span(i, i + len(CANARY_NAME), "GIVENNAME", CANARY_NAME, "llm")] if i >= 0 else []


def test_no_canary_reaches_logs_errors_or_upstream(caplog):
    sent_upstream: list[str] = []
    upstream_mode = {"fail": False}

    def handler(request: httpx.Request) -> httpx.Response:
        sent_upstream.append(request.content.decode())
        if upstream_mode["fail"]:
            return httpx.Response(500, text="upstream exploded")
        last = json.loads(request.content)["messages"][-1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": last}}]})

    stub = CanaryStub()
    upstream = httpx.Client(
        base_url="https://llm.example.test", transport=httpx.MockTransport(handler)
    )
    app = create_app(detector=stub, vault=Vault(os.urandom(32)), use_env=False, upstream=upstream)
    app.state.restore_key = "restore-key"
    client = TestClient(app)
    for name in (
        "",
        "pii_gateway",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "httpx",
        "fastapi",
    ):
        logging.getLogger(name).setLevel(logging.DEBUG)

    error_bodies, responses = [], []
    with caplog.at_level(logging.DEBUG):
        for policy in ("support", "analytics", "strict"):
            r = client.post(
                "/redact", json={"text": TEXT, "policy": policy, "conversation_id": "k"}
            )
            assert r.status_code == 200
            responses.append(r.json()["redacted"])
        red = client.post(
            "/redact", json={"text": TEXT, "policy": "analytics", "conversation_id": "k"}
        )
        token_text = red.json()["redacted"]
        denied = client.post("/restore", json={"text": token_text, "conversation_id": "k"})
        error_bodies.append(denied.text)
        ok = client.post("/restore", json={"text": token_text, "conversation_id": "k"},
                         headers={"X-Restore-Key": "restore-key"})  # fmt: skip
        assert ok.json()["restored"] == TEXT  # restore works, but its log line must stay clean
        r = client.post("/proxy", json={"messages": [{"role": "user", "content": TEXT}]})
        assert r.status_code == 200 and r.json()["reply"] == TEXT
        upstream_mode["fail"] = True
        error_bodies.append(
            client.post("/proxy", json={"messages": [{"role": "user", "content": TEXT}]}).text
        )
        upstream_mode["fail"] = False
        stub.fail = True
        error_bodies.append(client.post("/redact", json={"text": TEXT}).text)
        error_bodies.append(
            client.post("/proxy", json={"messages": [{"role": "user", "content": TEXT}]}).text
        )
        stub.fail = False
        error_bodies.append(client.post("/redact", json={"text": TEXT, "policy": "nope"}).text)
        error_bodies.append(client.post("/redact", json={"text": TEXT + "x" * MAX_CHARS}).text)
        error_bodies.append(client.post("/redact", json={"txt": TEXT}).text)

    logs = "\n".join(f"{r.name} {r.levelname} {r.getMessage()}" for r in caplog.records)
    assert caplog.records, "logging was not captured"
    for canary in CANARIES:
        assert canary not in logs, f"canary {canary!r} leaked into logs"
        for body in error_bodies:
            assert canary not in body, f"canary {canary!r} echoed in an error response"
        for body in sent_upstream:
            assert canary not in body, f"canary {canary!r} sent upstream"
        for redacted in responses:
            assert canary not in redacted, f"canary {canary!r} survived redaction"
    assert any("<GIVENNAME_1>" in b for b in sent_upstream)  # tokens, not values, went out
