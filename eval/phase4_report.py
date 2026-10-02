# ruff: noqa: E501  (markdown table rows)
"""Phase 4 tables: M5 (3 seeds) vs baselines on the final test sets, with document bootstraps.

    python -m eval.phase4_report [--n 1000]

Reads per-document predictions (results/runs/) and writes results/phase4/phase4.json and
results/phase4/phase4.md:
  - M5, model alone: seed-averaged character leakage, document leakage and over-redaction,
    each with a 95% CI (documents resampled; per-document counts averaged over the 3 seeds),
    plus the seed spread
  - M5 full gateway (model + validators, eval.gateway_eval) vs the model alone under the same
    gateway label scope (IBAN / IPADDRESS counted), paired
  - baselines (model-only scope), and the paired leakage difference M5 - baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.bootstrap import bootstrap, ci, doc_counts, ratios

SEEDS = ["m5_targeted", "m5_s42", "m5_s3407"]  # seeds 13, 42, 3407
SETS = ["tab", "support_desk_300", "test_id"]
BASELINES = ["gliner_nvidia", "openmed", "presidio", "validators"]


def summary(counts: np.ndarray, n: int) -> dict:
    s = bootstrap(counts, n)
    leak, over, doc = ratios(counts)
    pct = lambda lo_hi: [round(100 * x, 2) for x in lo_hi]  # noqa: E731
    return {"leak": round(100 * leak, 2), "leak_ci": pct(ci(s[:, 0])),
            "doc_leak": round(100 * doc, 2), "doc_leak_ci": pct(ci(s[:, 2])),
            "over": round(100 * over, 2), "over_ci": pct(ci(s[:, 1])), "docs": len(counts)}  # fmt: skip


def paired(a: np.ndarray, b: np.ndarray, n: int) -> dict:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = np.array([[x - y for x, y in zip(ratios(a[i]), ratios(b[i]), strict=True)] for i in idx])
    pa, pb = ratios(a), ratios(b)
    return {"leak_diff": round(100 * (pa[0] - pb[0]), 2),
            "leak_diff_ci": [round(100 * x, 2) for x in ci(d[:, 0])],
            "over_diff": round(100 * (pa[1] - pb[1]), 2),
            "over_diff_ci": [round(100 * x, 2) for x in ci(d[:, 1])]}  # fmt: skip


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    R = Path(args.results)
    out: dict = {"n_resamples": args.n, "seeds": SEEDS, "sets": {}}
    for ts in SETS:
        per_seed = {t: doc_counts(t, ts, R) for t in SEEDS}
        model = np.mean(list(per_seed.values()), axis=0)
        gl = np.mean([doc_counts(f"{t}_gl", ts, R, True) for t in SEEDS], axis=0)
        gw = np.mean([doc_counts(f"{t}_gw", ts, R, True) for t in SEEDS], axis=0)
        entry = {
            "m5_model": summary(model, args.n),
            "m5_seed_leak": {t: round(100 * ratios(c)[0], 2) for t, c in per_seed.items()},
            "m5_model_gateway_scope": summary(gl, args.n),
            "m5_gateway": summary(gw, args.n),
            "gateway_vs_model": paired(gw, gl, args.n),
            "baselines": {},
        }
        for b in BASELINES:
            if not (R / "runs" / f"{b}__{ts}.jsonl").exists():
                continue
            cb = doc_counts(b, ts, R)
            entry["baselines"][b] = {
                **summary(cb, args.n),
                "m5_minus_baseline": paired(model, cb, args.n),
            }
        out["sets"][ts] = entry
    lat = R / "phase4" / "gateway_latency.json"
    if lat.exists():
        out["gateway_latency"] = json.loads(lat.read_text())
    (R / "phase4").mkdir(exist_ok=True)
    (R / "phase4" / "phase4.json").write_text(json.dumps(out, indent=2) + "\n")
    md = render(out)
    (R / "phase4" / "phase4.md").write_text(md)
    print(md)
    return out


def _ci(x: list) -> str:
    return f"[{x[0]:.2f}, {x[1]:.2f}]"


def render(out: dict) -> str:
    lines = ["## Phase 4: final blind evaluation", "",
             f"M5 = Qwen3-1.7B + QLoRA r=16, lr 4e-4, train_mix_32k; seeds 13, 42, 3407 (per-document "
             f"counts averaged over seeds). 95% CIs: {out['n_resamples']} document resamples. "
             "Each test set was scored once per system.", ""]  # fmt: skip
    for ts, e in out["sets"].items():
        m, gw, gl = e["m5_model"], e["m5_gateway"], e["m5_model_gateway_scope"]
        lines += [f"### {ts} ({m['docs']} documents)", "",
                  "| System | Char leakage (%) | 95% CI | Doc leakage (%) | 95% CI | Over-redaction (%) | 95% CI |",
                  "| --- | ---: | --- | ---: | --- | ---: | --- |"]  # fmt: skip
        row = lambda name, s: (f"| {name} | {s['leak']:.2f} | {_ci(s['leak_ci'])} | {s['doc_leak']:.2f} | "  # noqa: E731
                               f"{_ci(s['doc_leak_ci'])} | {s['over']:.2f} | {_ci(s['over_ci'])} |")  # fmt: skip
        lines.append(row("**M5, model alone**", m))
        for b, s in e["baselines"].items():
            lines.append(row(b, s))
        seeds = ", ".join(f"{v:.2f}" for v in e["m5_seed_leak"].values())
        lines += ["", f"Seed leakage (13, 42, 3407): {seeds}.", ""]
        if e["baselines"]:
            lines += ["| M5 − baseline | Leakage Δ (pt) | 95% CI | Over-redaction Δ (pt) | 95% CI |",
                      "| --- | ---: | --- | ---: | --- |"]  # fmt: skip
            for b, s in e["baselines"].items():
                p = s["m5_minus_baseline"]
                lines.append(f"| {b} | {p['leak_diff']:+.2f} | {_ci(p['leak_diff_ci'])} | "
                             f"{p['over_diff']:+.2f} | {_ci(p['over_diff_ci'])} |")  # fmt: skip
            lines.append("")
        g = e["gateway_vs_model"]
        lines += ["Gateway label scope (IBAN, IPADDRESS counted):", "",
                  "| System | Char leakage (%) | 95% CI | Doc leakage (%) | Over-redaction (%) |",
                  "| --- | ---: | --- | ---: | ---: |",
                  f"| M5 model alone | {gl['leak']:.2f} | {_ci(gl['leak_ci'])} | {gl['doc_leak']:.2f} | {gl['over']:.2f} |",
                  f"| **M5 + validators (gateway)** | {gw['leak']:.2f} | {_ci(gw['leak_ci'])} | {gw['doc_leak']:.2f} | {gw['over']:.2f} |",
                  "", f"Gateway − model: leakage {g['leak_diff']:+.2f} pt {_ci(g['leak_diff_ci'])}, "
                  f"over-redaction {g['over_diff']:+.2f} pt {_ci(g['over_diff_ci'])}.", ""]  # fmt: skip
    if "gateway_latency" in out:
        lt = out["gateway_latency"]
        lines += ["### Live gateway (M5 + validators, one A40, single requests)", "",
                  f"{lt['requests']} support_desk_val messages, HTTP {lt['status_codes']}: latency "
                  f"p50 {lt['latency_s']['p50']:.2f} s, p95 {lt['latency_s']['p95']:.2f} s per request; "
                  f"restore exact {lt['restore_exact']}/{lt['requests']}.", ""]  # fmt: skip
    return "\n".join(lines)


if __name__ == "__main__":
    main()
