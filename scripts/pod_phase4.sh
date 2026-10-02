#!/usr/bin/env bash
# Phase 4 driver for one pod: the sharded sweep, then this pod's extra role, then a marker.
#   ROLE=baselines|gateway SHARD=0/2 RUNS=configs/sweeps/phase4_final.txt bash scripts/pod_phase4.sh
set -uo pipefail
cd "${WORKDIR:-/workspace}/pii-gateway"
export PATH="$HOME/.local/bin:$PATH" HF_HOME="${WORKDIR:-/workspace}/hf-cache" UV_NO_SYNC=1 \
       VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs results/phase4
bash scripts/pod_sweep.sh
case "${ROLE:-}" in
  baselines) TESTSETS="${BASELINE_SETS:-support_desk_300}" bash scripts/pod_eval_baselines.sh ;;
  gateway)
    # live gateway with M5 on this GPU; throwaway keys generated on the pod
    export PII_DETECTOR=lora PII_ADAPTER=outputs/hpo/m5_targeted
    PII_VAULT_KEY=$(python3 -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())")
    PII_RESTORE_KEY=$(python3 -c "import secrets;print(secrets.token_hex(16))")
    export PII_VAULT_KEY PII_RESTORE_KEY
    echo "=== gateway $(date -u +%H:%M:%S)"
    uv run uvicorn --factory pii_gateway.api:create_app --port 8000 > logs/gateway_server.log 2>&1 &
    srv=$!
    uv run python scripts/bench_gateway.py --url http://127.0.0.1:8000 \
      --out results/phase4/gateway_latency.json > logs/gateway_bench.log 2>&1
    tail -25 logs/gateway_bench.log
    kill "$srv"; wait "$srv" 2>/dev/null
    echo "server log lines: $(wc -l < logs/gateway_server.log)" ;;
esac
echo "=== phase4 done $(date -u +%H:%M:%S)"
