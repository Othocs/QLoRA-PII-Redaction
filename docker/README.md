# Docker

Two images, one compose file. Both run as a non-root user, and secrets are passed at runtime through `docker/gateway.env` (gitignored; see `gateway.env.example`), never baked into an image.

| Image | Dockerfile | What runs | Hardware |
| --- | --- | --- | --- |
| `pii-gateway:cpu` | `Dockerfile.cpu` (python:3.12-slim, ~620 MB) | Gateway with validators only: email, Luhn-checked cards, IBAN, phone numbers, IPs | Any CPU |
| `pii-gateway:gpu` | `Dockerfile.gpu` (CUDA 13 runtime + vLLM) | Gateway + the LoRA model in process + validators | NVIDIA GPU, CUDA 13 driver |

**No weights in the images.**

- The adapter is mounted read-only at `/models/adapter`: by default `outputs/hpo/m5_targeted` from a training run.
- The base model (`Qwen/Qwen3-1.7B`) downloads into the `hf-cache` volume on first start, which takes a few minutes. Model loading is covered by a 10-minute healthcheck start period.

```bash
cp docker/gateway.env.example docker/gateway.env    # fill in the keys
docker compose -f docker/compose.yaml --profile cpu up --build
docker compose -f docker/compose.yaml --profile gpu up --build
curl -s localhost:8000/redact -H 'Content-Type: application/json' \
  -d '{"text": "Hi, I am Ann Lee, card 4111 1111 1111 1111", "policy": "support"}'
```

## Security

- The CPU service runs with a read-only root filesystem, `cap_drop: ALL` and `no-new-privileges`. The GPU service has the same capability limits, but needs a writable `/cache`.
- Ports bind to `127.0.0.1` only; put a TLS proxy in front for remote clients.
- `/restore` is disabled unless `PII_RESTORE_KEY` is set.

## Testing

| Image | Tested where | What runs |
| --- | --- | --- |
| CPU | Locally and in CI (`.github/workflows/ci.yml`, job `docker`) | Builds and runs; `/health`, `/redact` and `/restore` are checked |
| GPU | CI | Builds, and its Python stack imports (vLLM, the gateway). It is never *run* there, because the runner has no GPU |

The GPU image's exact command (uvicorn + `PII_DETECTOR=lora`) was run against M5 on a RunPod A40 outside Docker. Its latency is in `results/phase4/gateway_latency.json`.
