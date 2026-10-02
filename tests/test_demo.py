"""Demo client logic against a mocked gateway (Gradio itself is not launched)."""

import importlib.util
from pathlib import Path

import httpx

spec = importlib.util.spec_from_file_location("demo_app", Path("demo/app.py"))
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

TEXT = "Ann, card 4111 1111 1111 1111."


def gateway(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok", "detector": "validators", "vault": False})
    return httpx.Response(200, json={
        "redacted": "Ann, card [CREDITCARDNUMBER].", "policy": "support", "conversation_id": "c",
        "entities": [{"start": 10, "end": 29, "label": "CREDITCARDNUMBER", "action": "mask"}],
    })  # fmt: skip


def client(handler):
    return httpx.Client(base_url="http://gw.test", transport=httpx.MockTransport(handler))


def test_redact_formats_highlights_and_status():
    red, spans, status = demo.redact(TEXT, "support", client(gateway))
    assert red == "Ann, card [CREDITCARDNUMBER]."
    assert spans == [("Ann, card ", None), ("4111 1111 1111 1111", "CREDITCARDNUMBER (mask)"),
                     (".", None)]  # fmt: skip
    assert "1 entities" in status and "validators" in status


def test_redact_reports_gateway_errors():
    err = client(lambda r: httpx.Response(503, json={"detail": "detection unavailable"}))
    assert demo.redact(TEXT, "support", err) == (
        "",
        [],
        "Gateway answered HTTP 503: detection unavailable",
    )
    assert demo.redact("  ", "support", client(gateway))[2] == "Enter some text."
