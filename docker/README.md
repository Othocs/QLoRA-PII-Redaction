# Docker (week 5)

- `Dockerfile.gpu`: vLLM serving the base model plus the LoRA adapter.
- `Dockerfile.cpu`: llama.cpp with a 4-bit GGUF build of the 1.7B model, for the on-premises story.
- `compose.yaml`: the API and the demo.

Both images run as a non-root user with the weights baked in. The API container has no outbound network except the `/proxy` route.
