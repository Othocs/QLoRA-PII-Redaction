"""Encrypted token -> original mapping (AES-256-GCM), scoped per tenant and conversation,
deleted after a TTL.

`token()` hands out stable placeholders ("<GIVENNAME_1>") for pseudonymisation: the same value
in the same conversation always gets the same token. Values are never stored in clear: the
lookup key is an HMAC of the value, and the value itself is AES-256-GCM encrypted with the
tenant, conversation and token as associated data, so a ciphertext moved to another scope
or edited fails to decrypt. The store is in memory; restarting the gateway forgets every
mapping, which is the safe failure for a redaction service.

The key is 32 bytes, base64 in PII_VAULT_KEY (generate: `python -c "import os,base64;
print(base64.b64encode(os.urandom(32)).decode())"`).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

TOKEN = re.compile(r"<([A-Z]+)_(\d+)>")


class VaultError(Exception):
    """A mapping is missing, expired or failed authentication."""


@dataclass
class _Scope:
    created: float
    by_digest: dict[bytes, str] = field(default_factory=dict)  # HMAC(value) -> token
    by_token: dict[str, bytes] = field(default_factory=dict)  # token -> nonce + ciphertext
    counters: dict[str, int] = field(default_factory=dict)  # label -> last number


class Vault:
    def __init__(self, key: bytes, ttl_s: float = 24 * 3600,
                 clock: Callable[[], float] = time.time) -> None:  # fmt: skip
        if len(key) != 32:
            raise ValueError("vault key must be 32 bytes (AES-256)")
        self._aead = AESGCM(key)
        self._mac_key = hashlib.sha256(b"pii-vault-lookup" + key).digest()
        self.ttl_s, self._clock = ttl_s, clock
        self._scopes: dict[tuple[str, str], _Scope] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_env(cls, var: str = "PII_VAULT_KEY", **kw) -> Vault:
        raw = os.environ.get(var)
        if not raw:
            raise VaultError(f"{var} is not set")
        return cls(base64.b64decode(raw), **kw)

    def subkey(self, purpose: str) -> bytes:
        """A key derived from the vault key for another use (e.g. the policy's hash action)."""
        return hmac.new(self._mac_key, purpose.encode(), "sha256").digest()

    def _scope(self, tenant: str, conversation: str, create: bool) -> _Scope | None:
        self.purge()
        key = (tenant, conversation)
        if key not in self._scopes and create:
            self._scopes[key] = _Scope(created=self._clock())
        return self._scopes.get(key)

    def token(self, tenant: str, conversation: str, label: str, value: str) -> str:
        digest = hmac.new(self._mac_key, f"{label}\x00{value}".encode(), "sha256").digest()
        with self._lock:
            scope = self._scope(tenant, conversation, create=True)
            if digest in scope.by_digest:
                return scope.by_digest[digest]
            n = scope.counters.get(label, 0) + 1
            scope.counters[label] = n
            tok = f"<{label}_{n}>"
            nonce = os.urandom(12)
            aad = f"{tenant}\x00{conversation}\x00{tok}".encode()
            scope.by_token[tok] = nonce + self._aead.encrypt(nonce, value.encode(), aad)
            scope.by_digest[digest] = tok
            return tok

    def reveal(self, tenant: str, conversation: str, tok: str) -> str:
        with self._lock:
            scope = self._scope(tenant, conversation, create=False)
            blob = scope.by_token.get(tok) if scope else None
        if blob is None:
            raise VaultError("unknown or expired token")
        aad = f"{tenant}\x00{conversation}\x00{tok}".encode()
        try:
            return self._aead.decrypt(blob[:12], blob[12:], aad).decode()
        except InvalidTag as e:
            raise VaultError("token failed authentication") from e

    def restore(self, tenant: str, conversation: str, text: str) -> tuple[str, int]:
        """Replace every known token in `text` by its value; unknown tokens stay as they are."""
        count = 0

        def sub(m: re.Match) -> str:
            nonlocal count
            try:
                value = self.reveal(tenant, conversation, m.group(0))
            except VaultError:
                return m.group(0)
            count += 1
            return value

        return TOKEN.sub(sub, text), count

    def purge(self) -> int:
        """Drop every scope older than the TTL; returns how many were dropped."""
        now = self._clock()
        with self._lock:
            old = [k for k, s in self._scopes.items() if now - s.created > self.ttl_s]
            for k in old:
                del self._scopes[k]
        return len(old)
