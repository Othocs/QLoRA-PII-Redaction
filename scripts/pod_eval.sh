#!/usr/bin/env bash
# Evaluate the zero-shot base model and a trained adapter on the pod (vLLM, constrained JSON).
#   ADAPTER=outputs/r16_10k_1.7b TAG=lora_r16_10k_1.7b TESTSETS=dev bash scripts/pod_eval.sh
set -euo pipefail
cd "${WORKDIR:-/workspace}/pii-gateway"
export PATH="$HOME/.local/bin:$PATH" HF_HOME="${WORKDIR:-/workspace}/hf-cache"
# Plain `uv run` re-syncs without the llm extra and upgrades numpy past what numba supports.
export UV_NO_SYNC=1
TESTSETS="${TESTSETS:-dev}"
if [ "${SKIP_BASE:-0}" != 1 ]; then
  uv run python -m eval.run_eval --systems base_llm --testsets "$TESTSETS"
fi
if [ -n "${ADAPTER:-}" ]; then
  uv run python -m eval.run_eval --systems lora --adapter "$ADAPTER" --tag "${TAG:-lora}" --testsets "$TESTSETS"
fi
