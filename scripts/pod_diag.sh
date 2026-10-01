#!/usr/bin/env bash
# Diagnostic LoRA runs with raw-output logging (week 3). Config A = previous settings,
# config B = 1,200-char chunks (training-length) and a 2,048-token output limit.
#   ADAPTER=outputs/r16_10k_1.7b bash scripts/pod_diag.sh
set -uo pipefail
cd "${WORKDIR:-/workspace}/pii-gateway"
export PATH="$HOME/.local/bin:$PATH" HF_HOME="${WORKDIR:-/workspace}/hf-cache" UV_NO_SYNC=1 \
       PII_LLM_LOG_RAW=1
SETS="${TESTSETS:-nemotron,tab,gretel_en,gretel_xx}"
mkdir -p logs
run() {  # tag testsets
  echo "=== $1 $(date -u +%H:%M:%S)"
  if uv run python -m eval.run_eval --systems lora --adapter "${ADAPTER:?set ADAPTER}" --tag "$1" \
       --testsets "$2" --latency-sample 0 > "logs/diag_$1.log" 2>&1; then
    grep -aE "^\[|leakage" "logs/diag_$1.log"
  else
    echo "FAILED $1"; tail -5 "logs/diag_$1.log"
  fi
}

PII_LLM_MAX_CHARS=2000 PII_LLM_MAX_NEW_TOKENS=1024 run diagA_chunk2000_tok1024 "$SETS"
PII_LLM_MAX_CHARS=1200 PII_LLM_MAX_NEW_TOKENS=2048 run diagB_chunk1200_tok2048 "dev,$SETS"
echo "=== done $(date -u +%H:%M:%S)"
