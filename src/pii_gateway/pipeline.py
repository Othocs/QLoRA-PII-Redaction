"""The redaction pipeline shared by the API (pii_gateway.api) and the hosted demo (space/).

    gw = Gateway(detector=None, vault=Vault(key))     # detector None = validators only
    result = gw.redact("Card 4111 1111 1111 1111", Policy.load("support"))

normalise (NFKC, zero-width, look-alikes) -> validators + optional model on the normalised
text -> spans mapped back to the original -> recall-first union -> policy. Any exception
propagates: callers fail closed (the API answers 503, the demo shows an error).
"""

from __future__ import annotations

import threading
import uuid

from pii_gateway.detectors.base import Detector
from pii_gateway.detectors.validators import ValidatorDetector
from pii_gateway.merge import recall_first_union
from pii_gateway.normalize import normalize, spans_to_original
from pii_gateway.policy import Policy, Redaction, apply
from pii_gateway.vault import Vault


class Gateway:
    def __init__(self, detector: Detector | None = None, vault: Vault | None = None) -> None:
        self.detector, self.vault = detector, vault
        self.validators = ValidatorDetector()
        self.hash_key = vault.subkey("policy-hash") if vault else None
        # sync API endpoints run in a thread pool; one in-process vLLM engine must not interleave
        self._model_lock = threading.Lock()

    @property
    def detector_name(self) -> str:
        return getattr(self.detector, "name", None) or "validators"

    def detect(self, text: str):
        norm = normalize(text)
        found = self.validators.detect(norm.text)
        if self.detector is not None:
            with self._model_lock:
                found = found + self.detector.detect(norm.text)
        return recall_first_union(text, spans_to_original(norm, text, found))

    def redact(self, text: str, policy: Policy, tenant: str = "default",
               conversation: str | None = None) -> Redaction:  # fmt: skip
        return apply(text, self.detect(text), policy, vault=self.vault, tenant=tenant,
                     conversation=conversation or uuid.uuid4().hex,
                     hash_key=self.hash_key)  # fmt: skip
