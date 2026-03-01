# Installation

This guide covers installing ContextPilot and its dependencies.

## Requirements

- Python >= 3.10
- CUDA 12.x (optional — for GPU-accelerated distance computation; not required on Mac)
- An inference engine: [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), [llama.cpp](https://github.com/ggerganov/llama.cpp), or any OpenAI-compatible server

## Install ContextPilot

**CPU (Mac / Apple Silicon or no CUDA):**
```bash
pip install contextpilot
```

**GPU (Linux + CUDA 12.x):**
```bash
pip install "contextpilot[gpu]"
```

The `[gpu]` extra installs `cupy-cuda12x` for GPU-accelerated distance computation. Without it, ContextPilot falls back to the CPU backend automatically.

Or install from source:
```bash
git clone https://github.com/EfficientContext/ContextPilot.git
cd ContextPilot
pip install -e .          # CPU
pip install -e ".[gpu]"   # GPU (CUDA 12.x)
```

This installs the core dependencies:

| Package | Purpose |
|---------|--------|
| `fastapi[all]` | HTTP server |
| `aiohttp` | Async inference engine proxy |
| `scipy` | Hierarchical clustering |
| `transformers` | Tokenizer / chat templates |
| `cupy-cuda12x` | GPU distance computation (`[gpu]` extra only) |
| `elasticsearch` | BM25 retriever (optional) |
| `datasets` | Loading benchmark datasets |

## Install an Inference Engine

**SGLang:**
```bash
pip install "sglang==0.5.6"
```

**vLLM:**
```bash
pip install vllm
```

For eviction sync, set `CONTEXTPILOT_INDEX_URL` when launching your engine and apply the corresponding patch (see [Online Usage Guide](../guides/online_usage.md#inference-engine-integration)).

**llama.cpp (Mac / Apple Silicon — no CUDA required):**
```bash
brew install llama.cpp
```

Then download a GGUF model and start llama-server with prefix caching enabled. See the [Mac + llama.cpp guide](../guides/mac_llama_cpp.md) for the full setup.

## Verify Installation

```bash
python -c "import contextpilot; print('ContextPilot', contextpilot.__version__)"
```
