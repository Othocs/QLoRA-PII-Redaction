"""Gradio demo: paste text, see what the gateway masks and why.

    make serve            # gateway on :8000 (validators only), or a GPU host with M5
    make demo             # this app on :7860 (GATEWAY_URL defaults to http://127.0.0.1:8000)

A thin client of the gateway's /redact endpoint: it never loads a model itself, so the same
demo shows the validators-only CPU container or the full M5 + validators gateway on a GPU,
depending on what GATEWAY_URL points at.
"""

from __future__ import annotations

import os

import httpx

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000")
POLICIES = ["support", "analytics", "strict"]
EXAMPLE = (
    "Hi, this is Priya Raman (Dr.), order 88213409. My card 4111 1111 1111 1111 was charged "
    "twice - please call me on +44 20 7946 0958 or mail priya.raman@example.com. "
    "I live at 41 Larkspur Rise, Harrogate HG2 8QT."
)


def redact(text: str, policy: str, client: httpx.Client | None = None) -> tuple[str, list, str]:
    """(redacted text, highlighted original as [(segment, label or None)], status line)."""
    if not text.strip():
        return "", [], "Enter some text."
    own = client is None
    client = client or httpx.Client(base_url=GATEWAY_URL, timeout=120)
    try:
        r = client.post("/redact", json={"text": text, "policy": policy})
        if r.status_code != 200:
            return "", [], f"Gateway answered HTTP {r.status_code}: {r.json().get('detail')}"
        body = r.json()
        health = client.get("/health").json()
    except httpx.HTTPError as e:
        return "", [], f"Gateway unreachable at {client.base_url}: {type(e).__name__}"
    finally:
        if own:
            client.close()
    highlighted, pos = [], 0
    for e in sorted(body["entities"], key=lambda e: e["start"]):
        if e["start"] > pos:
            highlighted.append((text[pos : e["start"]], None))
        highlighted.append((text[e["start"] : e["end"]], f"{e['label']} ({e['action']})"))
        pos = e["end"]
    if pos < len(text):
        highlighted.append((text[pos:], None))
    status = (f"{len(body['entities'])} entities · policy {body['policy']} · "
              f"detector {health.get('detector')}")  # fmt: skip
    return body["redacted"], highlighted, status


def build_ui():
    import gradio as gr

    with gr.Blocks(title="PII redaction gateway") as ui:
        gr.Markdown(
            "# PII redaction gateway\n"
            f"Paste customer-support text. The gateway at `{GATEWAY_URL}` finds personal data "
            "(M5 LLM + deterministic validators when served on a GPU; validators only on CPU) "
            "and masks, pseudonymises, hashes or keeps it according to the policy."
        )
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Input", lines=8, value=EXAMPLE)
                policy = gr.Dropdown(POLICIES, value="support", label="Policy")
                go = gr.Button("Redact", variant="primary")
            with gr.Column():
                out = gr.Textbox(label="Redacted", lines=8)
                spans = gr.HighlightedText(label="What was found", combine_adjacent=False)
                status = gr.Markdown()
        go.click(redact, inputs=[text, policy], outputs=[out, spans, status])
    return ui


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=int(os.environ.get("DEMO_PORT", "7860")))
