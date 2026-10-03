"""Hosted demo of the PII redaction gateway with the M5 model (Hugging Face Space).

The page runs the gateway pipeline in-process (pii_gateway.pipeline.Gateway): validators on
the Space's CPU, and M5 served by a RunPod Serverless vLLM endpoint that scales to zero when
idle. Configuration comes from Space secrets, never from this repo:

  PII_LLM_URL     https://api.runpod.ai/v2/<endpoint-id>/openai/v1
  PII_LLM_KEY     RunPod API key (only this Space and the endpoint owner see it)
  PII_LLM_MODEL   served LoRA name (default "m5")
  DEMO_RATE_PER_MIN / DEMO_DAILY_CAP   optional guardrail overrides (default 10 / 300)

Guardrails for a public page: inputs capped at 2,000 characters, a per-visitor rate limit,
a global daily cap, queue concurrency 2. If the model can't be reached, the page says so and
shows validator-only results, labelled as such. Nothing is stored or logged by the app.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("PII_POLICY_DIR", str(HERE / "configs" / "policy"))

from pii_gateway.pipeline import Gateway  # noqa: E402
from pii_gateway.policy import Policy  # noqa: E402
from pii_gateway.vault import Vault  # noqa: E402

MAX_CHARS = 2000
REPO = "https://github.com/Othocs/QLoRA-PII-Redaction"
POLICIES = ["support", "analytics", "strict"]
EXAMPLES = [
    "Hi, this is Dr. Amara Okafor. My card 4111 1111 1111 1111 was charged twice for order "
    "88213409 - please call me back on oh seven seven double oh, nine double oh, three one two.",
    "cust Saoirse Kerrigan, 34F, DOB 120390, moved to 41 Larkspur Rise, Harrogate HG2 8QT. "
    "new email saoirse.k@example.co.uk, NOK: brother Cormac (07700 900 518).",
    "Hello,\n\nPlease update my National Insurance number to QQ 74 19 22 B and send the "
    "refund to IBAN GB82 WEST 1234 5698 7654 32.\n\nThanks,\nRev. Tomasz Wierzbicki",
    "my order #55120931 still hasnt shipped and the gift card 6034 9321 0075 5510 shows zero, "
    "can someone call 415-555-0147? -- Jordan",
]
NOTICE = (
    "**Use fake data only.** Text you submit is sent to a GPU endpoint on RunPod to run the "
    "model and is not stored or logged by this app. The first request after a quiet period "
    "usually takes **3-5 minutes** (occasionally ~10 when GPUs are busy) while a GPU is found "
    "and the model loads; later ones take a second or two."
)


class Limiter:
    """Per-visitor sliding-window rate limit plus a global daily cap (in memory)."""

    def __init__(self, per_min: int, daily_cap: int, clock=time.time) -> None:
        self.per_min, self.daily_cap, self.clock = per_min, daily_cap, clock
        self.hits: dict[str, deque] = {}
        self.day, self.count = None, 0
        self.lock = threading.Lock()

    def check(self, visitor: str) -> str | None:
        """None if allowed (and counted), else a message for the visitor."""
        now = self.clock()
        today = datetime.fromtimestamp(now, UTC).date()
        with self.lock:
            if today != self.day:
                self.day, self.count = today, 0
            if self.count >= self.daily_cap:
                return "The demo's daily limit is reached - please try again tomorrow."
            q = self.hits.setdefault(visitor, deque())
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_min:
                return "Too many requests - please wait a minute."
            q.append(now)
            self.count += 1
        return None


def build_gateways():
    """(model gateway or None, validator-only gateway). One in-memory vault for both."""
    vault = Vault(os.urandom(32))
    fallback = Gateway(None, vault)
    if not os.environ.get("PII_LLM_URL"):
        return None, fallback
    from pii_gateway.detectors.registry import build

    return Gateway(build("lora_remote"), vault), fallback


def highlights(text: str, entities: list[dict]) -> list[tuple[str, str | None]]:
    out, pos = [], 0
    for e in sorted(entities, key=lambda e: e["start"]):
        if e["start"] > pos:
            out.append((text[pos : e["start"]], None))
        out.append((text[e["start"] : e["end"]], f"{e['label']} ({e['action']})"))
        pos = e["end"]
    if pos < len(text):
        out.append((text[pos:], None))
    return out


def run(text: str, policy_name: str, visitor: str, model_gw, fallback_gw, limiter: Limiter):
    """(redacted text, highlights, status markdown). Never returns unredacted text as output."""
    text = (text or "")[:MAX_CHARS]
    if not text.strip():
        return "", [], "Enter some text, or pick an example."
    if (msg := limiter.check(visitor)) is not None:
        return "", [], f"⏳ {msg}"
    policy = Policy.load(policy_name)
    t0 = time.perf_counter()
    used, note = "M5 + validators", ""
    try:
        if model_gw is None:
            raise RuntimeError("model endpoint not configured")
        result = model_gw.redact(text, policy)
    except Exception as e:  # noqa: BLE001 - degrade to validators, clearly labelled
        result = fallback_gw.redact(text, policy)
        used = "validators only"
        code = getattr(getattr(e, "response", None), "status_code", None)  # httpx errors
        why = f"{type(e).__name__}, HTTP {code}" if code else type(e).__name__
        note = (f" ⚠️ The model is unavailable right now ({why}); these results "
                "come from the deterministic validators alone, so names, addresses and dates "
                "are **not** detected.")  # fmt: skip
    dt = time.perf_counter() - t0
    status = (f"**{len(result.entities)} entities** · detector: {used} · policy: {policy.name} · "
              f"{dt:.1f} s.{note}")  # fmt: skip
    return result.text, highlights(text, result.entities), status


def build_ui():
    import gradio as gr

    model_gw, fallback_gw = build_gateways()
    limiter = Limiter(int(os.environ.get("DEMO_RATE_PER_MIN", "10")),
                      int(os.environ.get("DEMO_DAILY_CAP", "300")))  # fmt: skip

    def handler(text, policy, request: gr.Request):
        fwd = (request.headers.get("x-forwarded-for") or "") if request else ""
        visitor = fwd.split(",")[0].strip() or (request.client.host if request else "anon")
        return run(text, policy, visitor, model_gw, fallback_gw, limiter)

    with gr.Blocks(title="PII redaction gateway - M5 demo") as ui:
        gr.Markdown(
            "# PII redaction gateway\n"
            "Paste customer-support text. **M5**, a Qwen3-1.7B model fine-tuned with QLoRA, finds "
            "the personal data together with deterministic validators (Luhn, IBAN, phone, email, "
            "IP); a policy then masks, pseudonymises or keeps each value.\n\n"
            f"[Code and report on GitHub]({REPO}) · [Model card]({REPO}/blob/main/MODEL_CARD.md)"
            f" · [Evaluation]({REPO}/blob/main/docs/EVALUATION.md)"
        )
        gr.Markdown(NOTICE)
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label=f"Input (up to {MAX_CHARS} characters)", lines=9,
                                  max_length=MAX_CHARS, value=EXAMPLES[0])  # fmt: skip
                info = ("support: mask, keep city · analytics: pseudonymise identities, "
                        "keep age/city · strict: mask everything")  # fmt: skip
                policy = gr.Dropdown(POLICIES, value="support", label="Policy", info=info)
                go = gr.Button("Redact", variant="primary")
                gr.Examples(EXAMPLES, inputs=text, label="Examples (all fictional)")
            with gr.Column():
                out = gr.Textbox(label="Redacted", lines=9)
                spans = gr.HighlightedText(label="What was found", combine_adjacent=False)
                status = gr.Markdown()
        gr.Markdown(
            "Known limits: M5 is strongest on structured and synthetic text; on real chat "
            "transcripts encoder models such as GLiNER-PII still miss less (see the "
            f"[project report]({REPO}/blob/main/docs/REPORT.md)). English only."
        )
        go.click(handler, inputs=[text, policy], outputs=[out, spans, status])
    ui.queue(default_concurrency_limit=2, max_size=20)
    return ui


if __name__ == "__main__":
    build_ui().launch()
