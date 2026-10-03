"""Benchmark a running gateway end to end: single-request latency, correctness, restore.

    uv run python scripts/bench_gateway.py --url http://127.0.0.1:8000 [--testset support_desk_val]

Waits for /health, sends every message of a validation set (support_desk_val by default, never
a test set) to /redact one at a time with the analytics policy (pseudonymise), and records:
  - latency per request (seconds, and seconds per 1,000 characters): p50 / p95 / max
  - HTTP status codes; gold values still present in the redacted output (a leak proxy)
  - /restore round-trip: the restored text must equal the original exactly
The restore key is read from PII_RESTORE_KEY. Message text is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.run_eval import load_testset  # noqa: E402


def wait_ready(client: httpx.Client, timeout_s: float = 900) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = client.get("/health")
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise SystemExit("gateway did not become healthy")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--testset", default="support_desk_val")
    ap.add_argument("--policy", default="analytics")
    ap.add_argument("--out", default="results/phase4/gateway_latency.json")
    args = ap.parse_args(argv)
    _, examples = load_testset(args.testset, None, gateway_labels=True)
    restore_key = os.environ.get("PII_RESTORE_KEY", "")
    client = httpx.Client(base_url=args.url, timeout=300)
    health = wait_ready(client)
    client.post("/redact", json={"text": "warm-up: call 020 7946 0958", "policy": "strict"})

    secs, per_1k, codes, leaks, gold_total, restored_ok = [], [], {}, 0, 0, 0
    for i, ex in enumerate(examples):
        conv = f"bench-{i}"
        t0 = time.perf_counter()
        r = client.post("/redact", json={"text": ex.text, "policy": args.policy,
                                         "conversation_id": conv})  # fmt: skip
        dt = time.perf_counter() - t0
        codes[r.status_code] = codes.get(r.status_code, 0) + 1
        if r.status_code != 200:
            continue
        secs.append(dt)
        per_1k.append(dt * 1000 / max(1, len(ex.text)))
        red = r.json()["redacted"]
        for s in ex.spans:
            if s.label == "IGNORE":
                continue
            gold_total += 1
            leaks += ex.text[s.start : s.end] in red
        back = client.post("/restore", json={"text": red, "conversation_id": conv},
                           headers={"X-Restore-Key": restore_key})  # fmt: skip
        restored_ok += back.status_code == 200 and back.json()["restored"] == ex.text

    def pct(v, q):
        return float(np.percentile(v, q)) if v else None

    out = {
        "testset": args.testset,
        "policy": args.policy,
        "health": health,
        "requests": len(examples),
        "status_codes": codes,
        "latency_s": {
            "p50": pct(secs, 50),
            "p95": pct(secs, 95),
            "max": max(secs, default=None),
            "mean": statistics.fmean(secs) if secs else None,
        },  # fmt: skip
        "latency_s_per_1k_chars": {"p50": pct(per_1k, 50), "p95": pct(per_1k, 95)},
        "gold_values_left_in_output": leaks,
        "gold_values": gold_total,
        "restore_exact": restored_ok,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
