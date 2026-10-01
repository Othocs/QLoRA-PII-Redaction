#!/usr/bin/env bash
# Set up a RunPod GPU pod: clone the repo, install the training + vLLM extras, build the data.
# Run over `ssh -A` (agent forwarding) so the private repo clones with your local key,
# without copying any credential to the pod.
#
#   ssh -A root@<pod-ip> -p <port> 'bash -s' < scripts/pod_setup.sh
set -euo pipefail

REPO="${REPO:-git@github.com:Othocs/pii-gateway.git}"
BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-/workspace}"

mkdir -p ~/.ssh && ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd "$WORKDIR"
if [ -d pii-gateway ]; then
  git -C pii-gateway fetch -q && git -C pii-gateway checkout -q "$BRANCH" && git -C pii-gateway pull -q
else
  git clone -q -b "$BRANCH" "$REPO" pii-gateway
fi
cd pii-gateway

export HF_HOME="$WORKDIR/hf-cache" UV_CACHE_DIR="$WORKDIR/uv-cache"
uv sync --extra data --extra train --extra llm
export UV_NO_SYNC=1  # later `uv run`s must not re-sync without the extras
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run python -c "import torch, vllm, trl, peft, bitsandbytes; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| vllm', vllm.__version__, '| trl', trl.__version__)"
[ -f data/processed/train_10k.jsonl ] || make data
echo "setup done: $(pwd)"
