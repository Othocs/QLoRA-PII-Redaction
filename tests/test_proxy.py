"""/proxy: redact -> upstream LLM (mocked) -> restore; fails closed; disabled by default."""

import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from pii_gateway.api import create_app
from pii_gateway.spans import Span
from pii_gateway.vault import Vault

KEY = os.urandom(32)


class NameStub:
    name = "stub"

    def __init__(self, fail=False):
        self.fail = fail

    def detect(self, text):
        if self.fail:
            raise RuntimeError("model down")
        i = text.find("Priya")
        return [Span(i, i + 5, "GIVENNAME", "Priya", "llm")] if i >= 0 else []


def make(handler=None, detector=None):
    seen = []

    def default(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        last = body["messages"][-1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": f"Echo: {last}"}}]})

    upstream = httpx.Client(base_url="https://llm.example.test/v1",
                            transport=httpx.MockTransport(handler or default))  # fmt: skip
    app = create_app(detector=detector or NameStub(), vault=Vault(KEY), use_env=False,
                     upstream=upstream)  # fmt: skip
    return TestClient(app), seen


MSG = "Hi, I'm Priya, card 4111 1111 1111 1111, mail priya@example.com"


def test_proxy_redacts_upstream_and_restores_reply():
    client, seen = make()
    r = client.post("/proxy", json={"messages": [{"role": "user", "content": MSG}]})
    assert r.status_code == 200
    sent = seen[0]["messages"][0]["content"]
    for value in ("Priya", "4111", "priya@example.com"):
        assert value not in sent  # the upstream never sees a raw value
    assert "<GIVENNAME_1>" in sent and "<CREDITCARDNUMBER_1>" in sent
    assert r.json()["reply"] == f"Echo: {MSG}"  # tokens in the reply are restored
    assert all("text" not in e for e in r.json()["entities"][0])


def test_proxy_keeps_tokens_stable_across_messages():
    client, seen = make()
    msgs = [{"role": "user", "content": "I'm Priya"}, {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "Priya again"}]  # fmt: skip
    client.post("/proxy", json={"messages": msgs, "conversation_id": "c1"})
    contents = [m["content"] for m in seen[0]["messages"]]
    assert contents == ["I'm <GIVENNAME_1>", "ok", "<GIVENNAME_1> again"]


def test_proxy_fails_closed_without_calling_upstream():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    client, _ = make(handler, detector=NameStub(fail=True))
    r = client.post("/proxy", json={"messages": [{"role": "user", "content": MSG}]})
    assert r.status_code == 503 and calls == [] and "Priya" not in r.text


def test_proxy_upstream_error_is_502_without_echo():
    client, _ = make(lambda request: httpx.Response(500, text="boom"))
    r = client.post("/proxy", json={"messages": [{"role": "user", "content": MSG}]})
    assert r.status_code == 502 and "Priya" not in r.text and "4111" not in r.text


def test_proxy_disabled_by_default():
    client = TestClient(create_app(detector=NameStub(), vault=Vault(KEY), use_env=False))
    r = client.post("/proxy", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404


@pytest.mark.parametrize("body", [{"messages": []}, {"messages": [{"role": "user"}]}])
def test_proxy_validation_errors(body):
    client, _ = make()
    assert client.post("/proxy", json=body).status_code == 422
