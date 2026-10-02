"""FastAPI gateway: /redact, /restore (separate key scope, audited), /health. Fails closed;
never logs values.

    PII_VAULT_KEY=... PII_API_KEY=... PII_RESTORE_KEY=... \\
        uv run uvicorn --factory pii_gateway.api:create_app

Pipeline per request: normalise (NFKC, zero-width, look-alikes) -> detect on the normalised
text (validators always; the LoRA model too when PII_DETECTOR=lora) -> map spans back to the
original -> recall-first union -> policy (mask / pseudonymize / hash / keep).

Environment:
  PII_DETECTOR      validators (default, CPU only) | lora (vLLM on a GPU host)
  PII_ADAPTER       LoRA adapter path, for PII_DETECTOR=lora
  PII_VAULT_KEY     base64 32-byte AES-256 key (pseudonymize / restore)
  PII_API_KEY       required in X-API-Key for /redact when set
  PII_RESTORE_KEY   required in X-Restore-Key for /restore; /restore is disabled if unset

Fail closed: if detection fails, /redact answers 503 and never returns the input. Logs carry
counts, labels, policy and ids only, never text or values.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
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


def _env_detector() -> Detector | None:
    if os.environ.get("PII_DETECTOR", "validators") != "lora":
        return None
    from pii_gateway.detectors.registry import build

    return build("lora", adapter=os.environ["PII_ADAPTER"])


def create_app(detector: Detector | None = None, vault: Vault | None = None,
               use_env: bool = True) -> FastAPI:  # fmt: skip
    """`detector` is the model (None = validators only). Tests pass stubs and use_env=False."""
    if use_env:
        detector = detector or _env_detector()
        if vault is None and os.environ.get("PII_VAULT_KEY"):
            vault = Vault.from_env()
    validators = ValidatorDetector()
    model_lock = threading.Lock()  # sync endpoints run in a thread pool; vLLM is not thread-safe
    hash_key = vault.subkey("policy-hash") if vault else None
    api_key = os.environ.get("PII_API_KEY") if use_env else None
    restore_key = os.environ.get("PII_RESTORE_KEY") if use_env else None
    app = FastAPI(title="PII redaction gateway", version="0.1.0")
    app.state.restore_key = restore_key
    app.state.api_key = api_key

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
        try:
            policy = Policy.load(req.policy)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        conversation = req.conversation_id or uuid.uuid4().hex
        try:
            norm = normalize(req.text)
            found = validators.detect(norm.text)
            if detector is not None:
                with model_lock:
                    found = found + detector.detect(norm.text)
            spans = recall_first_union(req.text, spans_to_original(norm, req.text, found))
            result = apply(req.text, spans, policy, vault=vault, tenant=req.tenant,
                           conversation=conversation, hash_key=hash_key)  # fmt: skip
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

    return app
