# Installation

This guide covers installing ContextPilot and its dependencies.

## Requirements

- Python >= 3.10
- CUDA 12.x (for GPU-accelerated distance computation)
- An inference engine: [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), or any OpenAI-compatible server

## Install ContextPilot

```bash
pip install contextpilot
```

Or from source (development):
```bash
git clone https://github.com/EfficientContext/ContextPilot.git
cd ContextPilot
pip install -e .
python -m contextpilot.install_hook   # one-time: enables automatic SGLang integration
```

This installs the core dependencies:

| Package | Purpose |
|---------|--------|
| `fastapi[all]` | HTTP server |
| `aiohttp` | Async inference engine proxy |
| `scipy` | Hierarchical clustering |
| `transformers` | Tokenizer / chat templates |
| `cupy-cuda12x` | GPU distance computation |
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

For SGLang eviction sync, just set `CONTEXTPILOT_INDEX_URL` when launching — no patches needed. For vLLM, apply the manual patch. See [Online Usage Guide](../guides/online_usage.md#inference-engine-integration).

## Verify Installation

```bash
python -c "import contextpilot; print('ContextPilot', contextpilot.__version__)"
```
