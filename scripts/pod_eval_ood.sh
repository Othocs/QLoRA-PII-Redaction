#!/usr/bin/env bash
# Week 3 out-of-distribution runs on a GPU pod. Two environments, because vLLM and
# GLiNER pin different torch versions:
#   .venv      (data, train, llm extras; built by pod_setup.sh) -> lora, base_llm
#   venv-base  (data, presidio, presidio-lg, gliner extras)       -> GLiNER, OpenMed, Presidio
# A failing system is logged and skipped; the others still run.
#   ADAPTER=outputs/r16_10k_1.7b TAG=lora_r16_10k_1.7b bash scripts/pod_eval_ood.sh
set -uo pipefail
cd "${WORKDIR:-/workspace}/pii-gateway"
export PATH="$HOME/.local/bin:$PATH" HF_HOME="${WORKDIR:-/workspace}/hf-cache" \
       UV_CACHE_DIR="${WORKDIR:-/workspace}/uv-cache" UV_NO_SYNC=1
ALL="${TESTSETS:-test_holdout_regions,nemotron,tab,gretel_en,gretel_xx,openpii_xx}"
EN="${TESTSETS_EN:-test_holdout_regions,nemotron,tab,gretel_en}"
LIMIT="${LIMIT:+--limit $LIMIT}"
OUT="${OUT:-results}"
mkdir -p logs
run() {  # name, then run_eval args
  local name=$1; shift
  echo "=== $name $(date -u +%H:%M:%S)"
  if uv run python -m eval.run_eval --out "$OUT" --latency-sample 20 $LIMIT "$@" \
       > "logs/eval_${name}.log" 2>&1; then
    grep -aE "^\[|leakage" "logs/eval_${name}.log"
  else
    echo "FAILED: $name (see logs/eval_${name}.log)"; tail -5 "logs/eval_${name}.log"
  fi
}

# --- LLMs (vLLM, constrained JSON)
run lora --systems lora --adapter "${ADAPTER:?set ADAPTER}" --tag "${TAG:-lora}" --testsets "$ALL"
run base_llm --systems base_llm --testsets "$ALL"

# --- encoder baselines + Presidio
export UV_PROJECT_ENVIRONMENT="${WORKDIR:-/workspace}/venv-base"
if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  UV_NO_SYNC=0 uv sync -q --extra data --extra presidio --extra presidio-lg --extra gliner
fi
run gliner_nvidia --systems gliner_nvidia --device cuda --testsets "$ALL"
run gliner_knowledgator --systems gliner_knowledgator --device cuda --testsets "$ALL"
run openmed --systems openmed --device cuda --testsets "$ALL"
run presidio --systems presidio --testsets "$EN"
echo "=== done $(date -u +%H:%M:%S)"
