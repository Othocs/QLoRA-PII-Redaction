#!/usr/bin/env bash
# Encoder baselines + Presidio on a GPU pod, in their own venv (GLiNER pins a different torch
# than vLLM). The baseline half of pod_eval_ood.sh, driven by TESTSETS.
#   TESTSETS=support_desk_300 bash scripts/pod_eval_baselines.sh
set -uo pipefail
cd "${WORKDIR:-/workspace}/pii-gateway"
export PATH="$HOME/.local/bin:$PATH" HF_HOME="${WORKDIR:-/workspace}/hf-cache" \
       UV_CACHE_DIR="${WORKDIR:-/workspace}/uv-cache" UV_NO_SYNC=1 \
       UV_PROJECT_ENVIRONMENT="${WORKDIR:-/workspace}/venv-base"
SETS="${TESTSETS:?set TESTSETS}"
mkdir -p logs
if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  UV_NO_SYNC=0 uv sync -q --extra data --extra presidio --extra presidio-lg --extra gliner
fi
for sys in gliner_nvidia openmed presidio; do
  dev=""; [ "$sys" != presidio ] && dev="--device cuda"
  echo "=== baseline $sys $(date -u +%H:%M:%S)"
  # shellcheck disable=SC2086
  if uv run python -m eval.run_eval --systems "$sys" $dev --testsets "$SETS" --latency-sample 20 \
       > "logs/eval_${sys}.log" 2>&1 < /dev/null; then
    grep -aE "^\[|leakage" "logs/eval_${sys}.log"
  else
    echo "FAILED baseline $sys"; tail -5 "logs/eval_${sys}.log"
  fi
done
