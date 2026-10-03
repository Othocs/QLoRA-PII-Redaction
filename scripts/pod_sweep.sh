#!/usr/bin/env bash
# Train + evaluate every run in a run list (configs/sweeps/*.txt), optionally sharded
# across pods. Line format:  tag | base config | overrides (space-separated) | eval sets
# Adapters go to outputs/hpo/<tag>, results to results/<tag>__<set>.json. Finished runs
# (adapter + every result present) are skipped, so a re-launch resumes. A failed step is
# logged and the next run starts.
#   RUNS=configs/sweeps/phase1_lr.txt SHARD=0/3 bash scripts/pod_sweep.sh
set -uo pipefail
cd "${WORKDIR:-/workspace}/pii-gateway"
export PATH="$HOME/.local/bin:$PATH" HF_HOME="${WORKDIR:-/workspace}/hf-cache" UV_NO_SYNC=1
RUNS="${RUNS:?set RUNS to a run list}"
SHARD="${SHARD:-0/1}"; me="${SHARD%/*}"; of="${SHARD#*/}"
mkdir -p logs outputs/hpo
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; echo "${s%"${s##*[![:space:]]}"}"; }

k=-1
while IFS='|' read -r tag cfg overrides sets; do
  tag=$(trim "$tag"); [ -z "$tag" ] || [ "${tag:0:1}" = "#" ] && continue
  k=$((k + 1)); [ $((k % of)) -eq "$me" ] || continue
  cfg=$(trim "$cfg"); overrides=$(trim "$overrides"); sets=$(trim "$sets")
  out="outputs/hpo/$tag"
  done_all=1
  for s in ${sets//,/ }; do [ -f "results/${tag}__${s}.json" ] || done_all=0; done
  if [ -f "$out/adapter_model.safetensors" ] && [ "$done_all" = 1 ]; then
    echo "=== skip $tag (done)"; continue
  fi
  if [ ! -f "$out/adapter_model.safetensors" ]; then
    echo "=== train $tag $(date -u +%H:%M:%S) [$overrides]"
    # shellcheck disable=SC2086
    if ! uv run python training/train_lora.py "$cfg" --set output_dir="$out" $overrides \
         > "logs/sweep_train_${tag}.log" 2>&1 < /dev/null; then
      echo "FAILED train $tag"; tail -3 "logs/sweep_train_${tag}.log"; continue
    fi
    grep -aE '"(wall_clock_min|cost_usd|trainable_params)"' "logs/sweep_train_${tag}.log"
  fi
  echo "=== eval $tag $(date -u +%H:%M:%S) [$sets]"
  if uv run python -m eval.run_eval --systems lora --adapter "$out" --tag "$tag" \
       --testsets "$sets" --latency-sample "${LATENCY_SAMPLE:-0}" \
       > "logs/sweep_eval_${tag}.log" 2>&1 < /dev/null; then
    grep -aE '^\[|leakage' "logs/sweep_eval_${tag}.log"
  else
    echo "FAILED eval $tag"; tail -3 "logs/sweep_eval_${tag}.log"
  fi
done < "$RUNS"
echo "=== done $(date -u +%H:%M:%S) shard $SHARD"
