# Docker

## Option A: Standalone ContextPilot Server

Run ContextPilot as its own container, install the hook into your existing engine container separately.

### Build & Run

```bash
docker build -t contextpilot -f docker/Dockerfile .
docker run -p 8765:8765 contextpilot --infer-api-url http://<engine-host>:30000
```

### Install the hook in your engine container

One-liner — no ContextPilot clone needed:

```bash
# Inside your SGLang/vLLM container:
curl -sL https://raw.githubusercontent.com/EfficientContext/ContextPilot/main/contextpilot/install_standalone.py | python3 -
```

Then launch the engine with `CONTEXTPILOT_INDEX_URL` pointing at the CP server:

```bash
CONTEXTPILOT_INDEX_URL=http://<contextpilot-host>:8765 python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 30000
```

Or add it to your engine Dockerfile:

```dockerfile
RUN curl -sL https://raw.githubusercontent.com/EfficientContext/ContextPilot/main/contextpilot/install_standalone.py | python3 -
ENV CONTEXTPILOT_INDEX_URL=http://<contextpilot-host>:8765
```

## Option B: All-in-One (Engine + ContextPilot)

Single container with both the engine and ContextPilot server.

### Build

```bash
docker build -t contextpilot-sglang  -f docker/Dockerfile.sglang .
docker build -t contextpilot-vllm    -f docker/Dockerfile.vllm .
docker build -t contextpilot-lmcache -f docker/Dockerfile.lmcache .
```

Pin a specific engine version:

```bash
docker build -t contextpilot-sglang -f docker/Dockerfile.sglang --build-arg SGLANG_VERSION=v0.5.0 .
docker build -t contextpilot-vllm   -f docker/Dockerfile.vllm   --build-arg VLLM_VERSION=v0.8.5 .
```

### Run

**SGLang:**

```bash
docker run --gpus all --shm-size 32g --ipc=host \
  -p 30000:30000 -p 8765:8765 \
  -e HF_TOKEN=$HF_TOKEN \
  contextpilot-sglang \
  --model-path meta-llama/Llama-3.1-8B-Instruct --schedule-policy lpm
```

**vLLM:**

```bash
docker run --gpus all --ipc=host \
  -p 8000:8000 -p 8765:8765 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  contextpilot-vllm \
  Qwen/Qwen3.5-2B --enable-prefix-caching
```

Everything after the image name is passed to the engine. Defaults are `Qwen/Qwen3.5-2B` for both images.

**vLLM + LMCache (KV cache CPU offloading):**

[LMCache](https://github.com/LMCache/LMCache) offloads KV cache to CPU/disk so evicted prefixes can be restored without recomputation. ContextPilot works with LMCache out of the box — the `BlockPool` hook is unaffected.

```bash
docker build -t contextpilot-lmcache -f docker/Dockerfile.lmcache .
```

Pin a specific LMCache version:

```bash
docker build -t contextpilot-lmcache -f docker/Dockerfile.lmcache --build-arg LMCACHE_VERSION=latest .
```

Run:

```bash
docker run --gpus all --ipc=host \
  -p 8000:8000 -p 8765:8765 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  contextpilot-lmcache
```

Override the model or LMCache config:

```bash
docker run --gpus all --ipc=host \
  -p 8000:8000 -p 8765:8765 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  contextpilot-lmcache \
  Qwen/Qwen3-4B --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## GPU Selection

```bash
docker run --gpus '"device=2,3"' ...
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CONTEXTPILOT_PORT` | `8765` | ContextPilot HTTP server port |
| `SGLANG_PORT` | `30000` | SGLang serving port (all-in-one only) |
| `VLLM_PORT` | `8000` | vLLM serving port (all-in-one only) |
| `HF_TOKEN` | -- | HuggingFace token (SGLang) |
| `HUGGING_FACE_HUB_TOKEN` | -- | HuggingFace token (vLLM) |

## Verify

```bash
curl http://localhost:8765/health          # ContextPilot
curl http://localhost:30000/health         # SGLang
curl http://localhost:8000/health          # vLLM
```

## Architecture

**All-in-one images:** The entrypoint starts the ContextPilot HTTP server in the background, then `exec`s the engine as PID 1. `docker stop` sends SIGTERM to the engine for graceful shutdown. The `.pth` hook auto-activates monkey-patching since `CONTEXTPILOT_INDEX_URL` is set in the image.

**Standalone image:** Runs only the ContextPilot server as PID 1. The hook is installed separately in the engine environment via the one-liner above.
