#!/usr/bin/env bash
# Week 3 part B on a GPU pod: train M1-M3 and evaluate M0-M3 on every set, interleaved
# so results arrive early. A failed step is logged and the rest continue.
#   bash scripts/pod_partb.sh     (expects outputs/r16_10k_1.7b = M0 copied from the Mac)
set -uo pipefail
cd "${WORKDIR:-/workspace}/pii-gateway"
export PATH="$HOME/.local/bin:$PATH" HF_HOME="${WORKDIR:-/workspace}/hf-cache" UV_NO_SYNC=1
SETS="dev,gretel_dev,test_holdout_regions,openpii_xx,nemotron,tab,gretel_en,gretel_xx"
mkdir -p logs

step() {  # name, command...
  local name=$1; shift
  echo "=== $name $(date -u +%H:%M:%S)"
  if "$@" > "logs/partb_${name}.log" 2>&1; then
    grep -aE "^\[|leakage|wall_clock_min|cost_usd|adapter_mb" "logs/partb_${name}.log"
  else
    echo "FAILED: $name"; tail -5 "logs/partb_${name}.log"
  fi
}
train() { step "train_$1" uv run python training/train_lora.py "configs/train/$1.yaml"; }
evaluate() {  # tag adapter
  step "eval_$1" uv run python -m eval.run_eval --systems lora --adapter "$2" --tag "$1" \
    --testsets "$SETS" --latency-sample 0
}

evaluate m0_openpii10k outputs/r16_10k_1.7b
train m1_clean10k && evaluate m1_clean10k outputs/m1_clean10k
train m2_mix10k   && evaluate m2_mix10k   outputs/m2_mix10k
train m3_mix20k   && evaluate m3_mix20k   outputs/m3_mix20k
echo "=== done $(date -u +%H:%M:%S)"
