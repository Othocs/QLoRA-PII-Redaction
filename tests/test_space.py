"""Hosted demo logic (space/app.py): guardrails, model path, labelled validator fallback."""

import importlib.util
import os
from pathlib import Path

import httpx

from pii_gateway.pipeline import Gateway
from pii_gateway.spans import Span
from pii_gateway.vault import Vault

spec = importlib.util.spec_from_file_location("space_app", Path("space/app.py"))
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

TEXT = "Hi, Amara Okafor here, card 4111 1111 1111 1111."


class NameModel:
    name = "lora_remote"

    def __init__(self, fail=False, error=None):
        self.fail, self.error = fail, error

    def detect(self, text):
        if self.error is not None:
            raise self.error
        if self.fail:
            raise TimeoutError("cold start took too long")
        i = text.find("Amara")
        return [Span(i, i + 5, "GIVENNAME", "Amara", "llm"),
                Span(i + 6, i + 12, "SURNAME", "Okafor", "llm")] if i >= 0 else []  # fmt: skip


def gateways(fail=False):
    vault = Vault(os.urandom(32))
    return Gateway(NameModel(fail), vault), Gateway(None, vault)


def test_model_path_redacts_names_and_cards():
    model, fb = gateways()
    red, spans, status = app.run(TEXT, "support", "v1", model, fb, app.Limiter(10, 300))
    assert red == "Hi, [GIVENNAME] [SURNAME] here, card [CREDITCARDNUMBER]."
    assert "M5 + validators" in status and "3 entities" in status
    assert ("Amara", "GIVENNAME (mask)") in spans


def test_fallback_is_labelled_and_never_unredacted():
    model, fb = gateways(fail=True)
    red, _, status = app.run(TEXT, "support", "v1", model, fb, app.Limiter(10, 300))
    assert "4111" not in red and "[CREDITCARDNUMBER]" in red
    assert "validators only" in status and "unavailable" in status and "TimeoutError" in status
    red, _, status = app.run(TEXT, "support", "v1", None, fb, app.Limiter(10, 300))
    assert "validators only" in status  # endpoint not configured


def test_fallback_names_the_http_status():
    req = httpx.Request("POST", "https://api.runpod.ai/v2/x/openai/v1/chat/completions")
    resp = httpx.Response(500, request=req)
    err = httpx.HTTPStatusError("job failed", request=req, response=resp)
    vault = Vault(os.urandom(32))
    model, fb = Gateway(NameModel(error=err), vault), Gateway(None, vault)
    red, _, status = app.run(TEXT, "support", "v1", model, fb, app.Limiter(10, 300))
    assert "HTTP 500" in status and "validators only" in status and "4111" not in red


def test_rate_limit_and_daily_cap():
    now = [1_700_000_000.0]
    lim = app.Limiter(per_min=2, daily_cap=3, clock=lambda: now[0])
    assert lim.check("a") is None and lim.check("a") is None
    assert "wait a minute" in lim.check("a")
    now[0] += 61
    assert lim.check("a") is None  # window slid; 3rd request of the day
    assert "daily limit" in lim.check("b")
    now[0] += 86_400
    assert lim.check("b") is None  # new day


def test_input_capped_and_empty_input():
    model, fb = gateways()
    lim = app.Limiter(10, 300)
    assert app.run("   ", "support", "v", model, fb, lim)[2].startswith("Enter some text")
    red, _, _ = app.run("x" * (app.MAX_CHARS + 500), "strict", "v", model, fb, lim)
    assert len(red) == app.MAX_CHARS
