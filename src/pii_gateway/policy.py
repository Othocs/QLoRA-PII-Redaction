"""Per-tenant, per-label actions from YAML: mask, pseudonymize, hash or keep.

    default: mask            # action for any label not listed
    labels:
      CITY: keep
      GIVENNAME: pseudonymize

configs/policy/ ships three: support (mask, keep CITY), analytics (pseudonymise identities,
keep coarse attributes such as CITY, AGE, SEX) and strict (mask everything).

  mask          "[LABEL]"
  pseudonymize  "<LABEL_n>", stable per value within a conversation, reversible via the vault
  hash          "<LABEL:9f2c1a7b04de>", keyed HMAC-SHA256 (same value -> same hash, tenant-wide)
  keep          the value stays
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pii_gateway.spans import Span
from pii_gateway.vault import Vault

ACTIONS = ("mask", "pseudonymize", "hash", "keep")
POLICY_DIR = Path(
    os.environ.get("PII_POLICY_DIR", Path(__file__).resolve().parents[2] / "configs" / "policy")
)


@dataclass
class Policy:
    name: str
    default: str = "mask"
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bad = {a for a in [self.default, *self.labels.values()] if a not in ACTIONS}
        if bad:
            raise ValueError(f"policy {self.name!r}: unknown action(s) {sorted(bad)}")

    @classmethod
    def load(cls, name: str, directory: Path = POLICY_DIR) -> Policy:
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid policy name {name!r}")
        path = directory / f"{name}.yaml"
        if not path.exists():
            raise ValueError(f"unknown policy {name!r}")
        cfg = yaml.safe_load(path.read_text()) or {}
        return cls(name, cfg.get("default", "mask"), dict(cfg.get("labels") or {}))

    def action(self, label: str) -> str:
        return self.labels.get(label, self.default)


@dataclass
class Redaction:
    text: str
    entities: list[dict]  # start/end in the ORIGINAL text, label, action; never the value


def apply(
    text: str,
    spans: list[Span],
    policy: Policy,
    *,
    vault: Vault | None = None,
    tenant: str = "default",
    conversation: str = "default",
    hash_key: bytes | None = None,
) -> Redaction:
    """Replace each span per the policy, right to left so earlier offsets stay valid."""
    entities, out = [], text
    for s in sorted(spans, key=lambda s: s.start, reverse=True):
        act = policy.action(s.label)
        value = text[s.start : s.end]
        if act == "keep":
            repl = value
        elif act == "mask":
            repl = f"[{s.label}]"
        elif act == "pseudonymize":
            if vault is None:
                raise ValueError("pseudonymize needs a vault")
            repl = vault.token(tenant, conversation, s.label, value)
        else:
            if not hash_key:
                raise ValueError("hash needs a hash key")
            mac = hmac.new(hash_key, f"{tenant}\x00{s.label}\x00{value}".encode(), hashlib.sha256)
            repl = f"<{s.label}:{mac.hexdigest()[:12]}>"
        out = out[: s.start] + repl + out[s.end :]
        entities.append({"start": s.start, "end": s.end, "label": s.label, "action": act})
    return Redaction(out, entities[::-1])
