"""FastAPI gateway: /redact, /restore (separate key scope, audited), /proxy, /health.
Fails closed; never logs values.

    PII_VAULT_KEY=... PII_API_KEY=... PII_RESTORE_KEY=... \\
        uv run uvicorn --factory pii_gateway.api:create_app

Pipeline per text: normalise (NFKC, zero-width, look-alikes) -> detect on the normalised text
(validators always; the LoRA model too when PII_DETECTOR=lora) -> map spans back to the
original -> recall-first union -> policy (mask / pseudonymize / hash / keep).

  POST /redact   text -> redacted text + entities (offsets, label, action; never values)
  POST /restore  pseudonymised text -> original values (needs X-Restore-Key; audit-logged)
  POST /proxy    chat messages -> each redacted -> OpenAI-compatible upstream LLM -> the
                 reply's pseudonym tokens restored. The only route that makes outbound calls;
                 404 unless PII_UPSTREAM_URL is set. The upstream only ever sees redacted text.
  GET  /health

Environment:
  PII_DETECTOR        validators (default, CPU only) | lora (vLLM on a GPU host)
  PII_ADAPTER         LoRA adapter path, for PII_DETECTOR=lora
  PII_VAULT_KEY       base64 32-byte AES-256 key (pseudonymize / restore / proxy restore)
  PII_API_KEY         required in X-API-Key for /redact and /proxy when set
  PII_RESTORE_KEY     required in X-Restore-Key for /restore; /restore is disabled if unset
  PII_UPSTREAM_URL    base URL of an OpenAI-compatible API (enables /proxy)
  PII_UPSTREAM_KEY    bearer token for the upstream (optional)
  PII_UPSTREAM_MODEL  model name sent upstream when the request names none

Fail closed: if detection fails, /redact and /proxy answer 503 and never return or forward
the input; if the upstream fails, /proxy answers 502. Request-validation errors never echo
the submitted text. Logs carry counts, labels, policy, ids and the upstream host only.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import uuid
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pii_gateway.detectors.base import Detector
from pii_gateway.detectors.validators import ValidatorDetector
from pii_gateway.merge import recall_first_union
from pii_gateway.normalize import normalize, spans_to_original
from pii_gateway.policy import Policy, apply
from pii_gateway.vault import Vault, VaultError

log = logging.getLogger("pii_gateway")
MAX_CHARS = 100_000


class RedactRequest(BaseModel):
    text: str = Field(max_length=MAX_CHARS)
    policy: str = "support"
    tenant: str = "default"
    conversation_id: str | None = None


class RestoreRequest(BaseModel):
    text: str = Field(max_length=MAX_CHARS)
    tenant: str = "default"
    conversation_id: str


class ChatMessage(BaseModel):
    role: str = Field(max_length=32)
    content: str = Field(max_length=MAX_CHARS)


class ProxyRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    policy: str = "analytics"  # pseudonymises, so the reply's tokens can be restored
    tenant: str = "default"
    conversation_id: str | None = None
    model: str | None = None


def _env_upstream() -> httpx.Client | None:
    url = os.environ.get("PII_UPSTREAM_URL")
    if not url:
        return None
    key = os.environ.get("PII_UPSTREAM_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=120)


def _env_detector() -> Detector | None:
    if os.environ.get("PII_DETECTOR", "validators") != "lora":
        return None
    from pii_gateway.detectors.registry import build

    return build("lora", adapter=os.environ["PII_ADAPTER"])


def create_app(detector: Detector | None = None, vault: Vault | None = None,
               use_env: bool = True, upstream: httpx.Client | None = None) -> FastAPI:  # fmt: skip
    """`detector` is the model (None = validators only); `upstream` the /proxy client.
    Tests pass stubs (and an httpx.MockTransport client) with use_env=False."""
    if use_env:
        detector = detector or _env_detector()
        if vault is None and os.environ.get("PII_VAULT_KEY"):
            vault = Vault.from_env()
        upstream = upstream or _env_upstream()
    upstream_model = os.environ.get("PII_UPSTREAM_MODEL", "default") if use_env else "default"
    validators = ValidatorDetector()
    model_lock = threading.Lock()  # sync endpoints run in a thread pool; vLLM is not thread-safe
    hash_key = vault.subkey("policy-hash") if vault else None
    api_key = os.environ.get("PII_API_KEY") if use_env else None
    restore_key = os.environ.get("PII_RESTORE_KEY") if use_env else None
    app = FastAPI(title="PII redaction gateway", version="0.1.0")
    app.state.restore_key = restore_key
    app.state.api_key = api_key

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 body echoes the submitted input; return locations only
        errors = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")}
                  for e in exc.errors()]  # fmt: skip
        return JSONResponse({"detail": errors}, status_code=422)

    def redact_text(text: str, policy: Policy, tenant: str, conversation: str):
        """The detection pipeline for one text; raises on any failure (callers fail closed)."""
        norm = normalize(text)
        found = validators.detect(norm.text)
        if detector is not None:
            with model_lock:
                found = found + detector.detect(norm.text)
        spans = recall_first_union(text, spans_to_original(norm, text, found))
        return apply(text, spans, policy, vault=vault, tenant=tenant,
                     conversation=conversation, hash_key=hash_key)  # fmt: skip

    def load_policy(name: str) -> Policy:
        try:
            return Policy.load(name)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    def check_api_key(x_api_key: str | None = Header(default=None)) -> None:
        key = app.state.api_key
        if key and not (x_api_key and hmac.compare_digest(x_api_key, key)):
            raise HTTPException(401, "invalid API key")

    def check_restore_key(x_restore_key: str | None = Header(default=None)) -> None:
        key = app.state.restore_key
        if not key:
            raise HTTPException(403, "restore is disabled")
        if not (x_restore_key and hmac.compare_digest(x_restore_key, key)):
            raise HTTPException(403, "restore needs the restore key")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "detector": getattr(detector, "name", None) or "validators",
                "vault": vault is not None}  # fmt: skip

    @app.post("/redact", dependencies=[Depends(check_api_key)])
    def redact(req: RedactRequest) -> dict:
        policy = load_policy(req.policy)
        conversation = req.conversation_id or uuid.uuid4().hex
        try:
            result = redact_text(req.text, policy, req.tenant, conversation)
        except Exception as e:  # noqa: BLE001 - fail closed, whatever went wrong
            log.error("redact failed: %s (tenant=%s)", type(e).__name__, req.tenant)
            raise HTTPException(503, "detection unavailable; nothing was returned") from None
        log.info("redact tenant=%s conversation=%s policy=%s entities=%d labels=%s",
                 req.tenant, conversation, policy.name, len(result.entities),
                 sorted({e["label"] for e in result.entities}))  # fmt: skip
        return {"redacted": result.text, "entities": result.entities,
                "policy": policy.name, "conversation_id": conversation}  # fmt: skip

    @app.post("/restore", dependencies=[Depends(check_restore_key)])
    def restore(req: RestoreRequest) -> dict:
        if vault is None:
            raise HTTPException(503, "no vault configured")
        try:
            text, n = vault.restore(req.tenant, req.conversation_id, req.text)
        except VaultError:
            raise HTTPException(409, "vault error") from None
        log.info("AUDIT restore tenant=%s conversation=%s tokens=%d",
                 req.tenant, req.conversation_id, n)  # fmt: skip
        return {"restored": text, "tokens_restored": n}

    @app.post("/proxy", dependencies=[Depends(check_api_key)])
    def proxy(req: ProxyRequest) -> dict:
        if upstream is None:
            raise HTTPException(404, "proxy is disabled (PII_UPSTREAM_URL not set)")
        policy = load_policy(req.policy)
        conversation = req.conversation_id or uuid.uuid4().hex
        try:  # redact everything first; nothing leaves the gateway if any message fails
            results = [redact_text(m.content, policy, req.tenant, conversation)
                       for m in req.messages]  # fmt: skip
        except Exception as e:  # noqa: BLE001 - fail closed: the upstream is never called
            log.error("proxy redact failed: %s (tenant=%s)", type(e).__name__, req.tenant)
            raise HTTPException(503, "detection unavailable; nothing was forwarded") from None
        body = {"model": req.model or upstream_model,
                "messages": [{"role": m.role, "content": r.text}
                             for m, r in zip(req.messages, results, strict=True)]}  # fmt: skip
        host = urlparse(str(upstream.base_url)).netloc
        try:
            resp = upstream.post("/chat/completions", json=body)
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001 - upstream down or malformed answer
            log.error("proxy upstream failed: %s (host=%s)", type(e).__name__, host)
            raise HTTPException(502, "upstream LLM unavailable") from None
        restored, n = vault.restore(req.tenant, conversation, reply) if vault else (reply, 0)
        entities = [r.entities for r in results]
        log.info("AUDIT proxy tenant=%s conversation=%s messages=%d entities=%d "
                 "tokens_restored=%d upstream=%s", req.tenant, conversation, len(results),
                 sum(len(e) for e in entities), n, host)  # fmt: skip
        return {"reply": restored, "conversation_id": conversation, "policy": policy.name,
                "entities": entities}  # fmt: skip

    return app
