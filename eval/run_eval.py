"""Run systems x test sets and write one results JSON per pair.

    python -m eval.run_eval --systems presidio,gliner_knowledgator --testsets dev --limit 200

Writes results/{system}__{testset}.json (metrics, latency, environment) and
results/runs/{system}__{testset}.jsonl (per-example predictions, gitignored), then
refreshes results/SUMMARY.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval.metrics import aggregate
from pii_gateway.detectors.base import Detector
from pii_gateway.detectors.registry import build
from pii_gateway.spans import Example, Span, read_examples

TESTSETS: dict[str, str] = {
    "dev": "data/processed/dev.jsonl",
    "test_id": "data/processed/test_id.jsonl",
    "test_holdout_regions": "data/processed/test_holdout_regions.jsonl",
    "support_desk": "data/support_desk/support_desk.jsonl",
    # out-of-distribution sets (data/prepare_eval_sets.py)
    "nemotron": "data/processed/nemotron.jsonl",
    "tab": "data/processed/tab.jsonl",
    "gretel_en": "data/processed/gretel_en.jsonl",
    "gretel_xx": "data/processed/gretel_xx.jsonl",
    "openpii_xx": "data/processed/openpii_xx.jsonl",
    "fixture": "tests/fixtures/openpii_sample.jsonl",
}


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10  # bytes on macOS, KiB on Linux


def load_testset(name_or_path: str, limit: int | None) -> tuple[str, list[Example]]:
    path = Path(TESTSETS.get(name_or_path, name_or_path))
    name = name_or_path if name_or_path in TESTSETS else path.stem
    if not path.exists():
        raise FileNotFoundError(f"test set {name!r} not found at {path} (run `make data`?)")
    return name, list(read_examples(path, limit))


def run_batched(
    det, examples: list[Example], latency_sample: int
) -> tuple[dict[str, list[Span]], dict]:
    """Detectors with detect_batch (the LLM): one batched pass for the predictions,
    then single-document calls on a sample to measure per-request latency."""
    t0 = time.perf_counter()
    outs = det.detect_batch([ex.text for ex in examples])
    total_s = time.perf_counter() - t0
    preds = {ex.id: spans for ex, spans in zip(examples, outs, strict=True)}
    stats = dict(det.stats) if hasattr(det, "stats") else None
    raw_log = list(getattr(det, "raw_log", []))
    ms_per_1k = []
    for ex in examples[:latency_sample]:
        t1 = time.perf_counter()
        det.detect(ex.text)
        ms_per_1k.append((time.perf_counter() - t1) * 1e6 / max(1, len(ex.text)))
    if stats is not None:  # format stats and raw log describe the batched pass only
        det.stats = stats
    if hasattr(det, "raw_log"):
        det.raw_log = raw_log
    total_chars = sum(len(ex.text) for ex in examples)
    latency = {
        "p50_ms_per_1k_chars": float(np.percentile(ms_per_1k, 50)) if ms_per_1k else None,
        "p95_ms_per_1k_chars": float(np.percentile(ms_per_1k, 95)) if ms_per_1k else None,
        "latency_sample": len(ms_per_1k),
        "chars_per_sec": total_chars / total_s if total_s else None,
        "batched": True,
        "n_errors": 0,
        "errors": [],
    }
    return preds, latency


def run_one(det: Detector, examples: list[Example]) -> tuple[dict[str, list[Span]], dict]:
    if examples:  # warm-up: first call pays for lazy init / graph compilation
        det.detect(examples[0].text)
    preds: dict[str, list[Span]] = {}
    ms_per_1k: list[float] = []
    errors: list[dict] = []
    total_chars, total_s = 0, 0.0
    for ex in examples:
        t0 = time.perf_counter()
        try:
            preds[ex.id] = det.detect(ex.text)
        except Exception as e:  # noqa: BLE001 - a failing doc counts as "predicted nothing"
            errors.append({"id": ex.id, "error": repr(e)[:300]})
            continue
        dt = time.perf_counter() - t0
        total_s += dt
        total_chars += len(ex.text)
        ms_per_1k.append(dt * 1000 / max(1, len(ex.text)) * 1000)
    latency = {
        "p50_ms_per_1k_chars": float(np.percentile(ms_per_1k, 50)) if ms_per_1k else None,
        "p95_ms_per_1k_chars": float(np.percentile(ms_per_1k, 95)) if ms_per_1k else None,
        "chars_per_sec": total_chars / total_s if total_s else None,
        "n_errors": len(errors),
        "errors": errors[:10],
    }
    return preds, latency


def main(argv: list[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--systems", default="presidio,gliner_knowledgator,gliner_nvidia")
    ap.add_argument("--testsets", default="dev")
    ap.add_argument("--limit", type=int, default=None, help="first N examples of each test set")
    ap.add_argument("--out", default="results")
    ap.add_argument(
        "--device", default=None, help="cpu | mps | cuda (default: each detector's own)"
    )
    ap.add_argument("--spacy-model", default=None, help="override Presidio's spaCy model")
    ap.add_argument("--base-model", default=None, help="LLM systems: base model id")
    ap.add_argument("--adapter", default=None, help="system 'lora': path to the LoRA adapter")
    ap.add_argument("--backend", default=None, help="LLM systems: vllm (default) | hf")
    ap.add_argument(
        "--tag", default=None, help="name results by this tag instead of the system name"
    )
    ap.add_argument("--latency-sample", type=int, default=50)
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    testsets = [load_testset(t.strip(), args.limit) for t in args.testsets.split(",") if t.strip()]
    written = []
    for sys_name in [s.strip() for s in args.systems.split(",") if s.strip()]:
        opts = {
            "device": args.device,
            "spacy_model": args.spacy_model,
            "base_model": args.base_model,
            "adapter": args.adapter,
            "backend": args.backend,
        }
        kw = {k: v for k, v in opts.items() if v}
        result_name = args.tag or sys_name
        t0 = time.perf_counter()
        det = build(sys_name, **kw)
        load_s = time.perf_counter() - t0
        for ts_name, examples in testsets:
            print(f"[{result_name}] {ts_name}: {len(examples)} examples ...", file=sys.stderr)
            if hasattr(det, "detect_batch"):
                preds, latency = run_batched(det, examples, args.latency_sample)
            else:
                preds, latency = run_one(det, examples)
            metrics = aggregate(examples, preds)
            if getattr(det, "log_raw", False):  # LLM diagnostics: raw outputs per chunk
                with open(out / "runs" / f"{result_name}__{ts_name}.raw.jsonl", "w") as f:
                    for entry in det.raw_log:
                        entry["id"] = examples[entry.pop("text_index")].id
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                det.raw_log = []
            if hasattr(det, "stats"):  # LLM: format reliability
                st = dict(det.stats)
                st["valid_json_rate"] = st["valid_json"] / st["outputs"] if st["outputs"] else None
                metrics["llm_output"] = st
                det.stats = {k: 0 for k in det.stats}
            result = {
                "system": result_name,
                "detector": sys_name,
                "adapter": args.adapter,
                "detector_config": getattr(det, "config", None),
                "testset": ts_name,
                "n_examples": len(examples),
                "limit": args.limit,
                "metrics": metrics,
                "latency": latency,
                "env": {
                    "device": args.device or "default",
                    "platform": f"{platform.system()} {platform.machine()}",
                    "processor": platform.processor(),
                    "python": platform.python_version(),
                    "git_sha": git_sha(),
                    "model_load_s": round(load_s, 2),
                    "peak_rss_mb": round(peak_rss_mb(), 1),
                    "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            }
            path = out / f"{result_name}__{ts_name}.json"
            path.write_text(json.dumps(result, indent=2) + "\n")
            run_path = out / "runs" / f"{result_name}__{ts_name}.jsonl"
            with open(run_path, "w", encoding="utf-8") as f:
                for ex in examples:
                    spans = [s.to_dict(full=True) for s in preds.get(ex.id, [])]
                    f.write(json.dumps({"id": ex.id, "pred": spans}, ensure_ascii=False) + "\n")
            written.append(path)
            print(
                f"  leakage {metrics['leakage_chars']:.3f}  strict F1 {metrics['strict']['f1']:.3f}"
                f"  p95 {latency['p95_ms_per_1k_chars'] or float('nan'):.0f} ms/1k chars",
                file=sys.stderr,
            )
        del det

    from eval.summarize import summarize

    # Only the canonical results/ directory feeds the README table.
    is_main = out.resolve() == Path("results").resolve()
    summarize(out, Path("README.md") if is_main else None)
    return written


if __name__ == "__main__":
    main()
