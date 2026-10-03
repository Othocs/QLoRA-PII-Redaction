"""Remote M5 backend (OpenAI-compatible vLLM, e.g. RunPod Serverless) against a mocked server."""

import json

import httpx
import pytest

from pii_gateway.detectors.llm import INSTRUCTION, LLMDetector, output_schema
from pii_gateway.pipeline import Gateway
from pii_gateway.policy import Policy

TEXT = "Hi, I'm Priya Raman, call 020 7946 0958."


def detector(handler, **kw):
    url = "https://api.runpod.ai/v2/abc/openai/v1"
    det = LLMDetector("Qwen/Qwen3-1.7B", adapter="m5", backend="openai", api_base=url,
                      api_key="rp-test", **kw)  # fmt: skip
    det.client = httpx.Client(base_url=det.client.base_url, headers=det.client.headers,
                              transport=httpx.MockTransport(handler))  # fmt: skip
    return det


def answer(items, finish="stop"):
    content = json.dumps(items)
    return httpx.Response(200, json={"choices": [{"message": {"content": content},
                                                  "finish_reason": finish}]})  # fmt: skip


def test_request_matches_the_evaluated_setup():
    seen = []

    def handler(request):
        seen.append((request, json.loads(request.content)))
        return answer(
            [{"label": "GIVENNAME", "text": "Priya"}, {"label": "SURNAME", "text": "Raman"}]
        )

    spans = detector(handler).detect(TEXT)
    request, body = seen[0]
    assert request.url.path == "/v2/abc/openai/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer rp-test"
    assert body["model"] == "m5" and body["temperature"] == 0.0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["json_schema"]["schema"] == output_schema()
    assert body["messages"][0]["content"].startswith(INSTRUCTION)
    assert TEXT in body["messages"][0]["content"]
    assert [(s.label, s.text) for s in spans] == [("GIVENNAME", "Priya"), ("SURNAME", "Raman")]


def test_invented_values_dropped_and_truncation_counted():
    det = detector(lambda r: answer([{"label": "GIVENNAME", "text": "Zelda"}], finish="length"))
    assert det.detect(TEXT) == []
    assert det.stats["dropped_items"] == 1 and det.stats["truncated"] == 1


def test_long_text_is_chunked_and_offsets_shifted():
    calls = []

    def handler(request):
        content = json.loads(request.content)["messages"][0]["content"]
        calls.append(content)
        return answer([{"label": "EMAIL", "text": "end@example.com"}] if "end@" in content else [])

    text = ("filler words " * 120) + "mail end@example.com"
    spans = detector(handler).detect(text)
    assert len(calls) >= 2
    assert [text[s.start : s.end] for s in spans] == ["end@example.com"]


def test_cold_start_retry_then_success(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(503, text="worker starting")
        return answer([{"label": "GIVENNAME", "text": "Priya"}])

    assert [s.text for s in detector(handler).detect(TEXT)] == ["Priya"]


def test_failed_job_500_is_resubmitted(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    codes = iter([500, 500])

    def handler(request):
        code = next(codes, 200)
        return answer([{"label": "GIVENNAME", "text": "Priya"}]) if code == 200 else (
            httpx.Response(code, text="job failed"))  # fmt: skip

    assert [s.text for s in detector(handler).detect(TEXT)] == ["Priya"]


def test_client_errors_fail_fast_without_retry(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, text="bad key")

    with pytest.raises(httpx.HTTPStatusError):
        detector(handler).detect(TEXT)
    assert len(calls) == 1


def test_persistent_5xx_gives_up_after_four_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500, text="down")

    with pytest.raises(httpx.HTTPStatusError):
        detector(handler).detect(TEXT)
    assert len(calls) == 4


def test_errors_propagate_so_the_gateway_fails_closed(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    det = detector(lambda r: httpx.Response(401, text="bad key"))
    with pytest.raises(httpx.HTTPStatusError):
        det.detect(TEXT)
    with pytest.raises(httpx.HTTPStatusError):
        Gateway(det).redact(TEXT, Policy.load("support"))


def test_registry_reads_env(monkeypatch):
    from pii_gateway.detectors.registry import build

    monkeypatch.setenv("PII_LLM_URL", "https://api.runpod.ai/v2/xyz/openai/v1")
    monkeypatch.setenv("PII_LLM_KEY", "k")
    det = build("lora_remote")
    assert det.served_model == "m5" and det.name == "lora_remote"
    monkeypatch.delenv("PII_LLM_URL")
    with pytest.raises(ValueError):
        build("lora_remote")
