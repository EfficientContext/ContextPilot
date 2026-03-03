# Docker

All-in-one images that run ContextPilot + an inference engine in a single container.

## Build

```bash
docker build -t contextpilot-sglang -f docker/Dockerfile.sglang .
docker build -t contextpilot-vllm   -f docker/Dockerfile.vllm .
```

Pin a specific engine version:

```bash
docker build -t contextpilot-sglang -f docker/Dockerfile.sglang --build-arg SGLANG_VERSION=v0.5.0 .
docker build -t contextpilot-vllm   -f docker/Dockerfile.vllm   --build-arg VLLM_VERSION=v0.8.5 .
```

## Run

### SGLang

```bash
docker run --gpus all --shm-size 32g --ipc=host \
  -p 30000:30000 -p 8765:8765 \
  -e HF_TOKEN=$HF_TOKEN \
  contextpilot-sglang \
  --model-path meta-llama/Llama-3.1-8B-Instruct --schedule-policy lpm
```

### vLLM

```bash
docker run --gpus all --ipc=host \
  -p 8000:8000 -p 8765:8765 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  contextpilot-vllm \
  Qwen/Qwen2.5-7B-Instruct --enable-prefix-caching
```

Everything after the image name is passed to the engine. Defaults are `Qwen/Qwen2.5-7B-Instruct` for both images.

## GPU Selection

```bash
docker run --gpus '"device=2,3"' ...
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CONTEXTPILOT_PORT` | `8765` | ContextPilot HTTP server port |
| `SGLANG_PORT` | `30000` | SGLang serving port |
| `VLLM_PORT` | `8000` | vLLM serving port |
| `HF_TOKEN` | -- | HuggingFace token (SGLang) |
| `HUGGING_FACE_HUB_TOKEN` | -- | HuggingFace token (vLLM) |

## Verify

```bash
curl http://localhost:8765/health
curl http://localhost:30000/health  # SGLang
curl http://localhost:8000/health   # vLLM
```

## Architecture

The entrypoint starts the ContextPilot HTTP server in the background, then `exec`s the engine as PID 1. This means:

- `docker stop` sends SIGTERM to the engine for graceful shutdown
- The ContextPilot server dies automatically when the container exits
- The `.pth` hook auto-activates monkey-patching since `CONTEXTPILOT_INDEX_URL` is set in the image
