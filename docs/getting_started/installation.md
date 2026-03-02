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
python -m contextpilot.install_hook   # one-time: enables automatic SGLang + vLLM hooks
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
pip install "sglang>=0.5"
```

**vLLM:**
```bash
pip install vllm
```

Both engines are supported via zero-patch runtime hooks — just set `CONTEXTPILOT_INDEX_URL` when launching. See [Online Usage Guide](../guides/online_usage.md#inference-engine-integration).

## Distributed Setup

If the ContextPilot index server and the inference engine run in **separate Python environments** (e.g., different virtualenvs or containers), the engine environment won't have the `contextpilot` package. Use the standalone hook instead:

```bash
# In the engine's Python environment (one command, no clone needed):
pip install requests
curl -sL https://raw.githubusercontent.com/EfficientContext/ContextPilot/main/contextpilot/install_standalone.py | python -
```

The installer downloads the hook from GitHub and installs it into site-packages. No `contextpilot` clone or install needed — just `requests` as a runtime dependency.

## Verify Installation

```bash
python -c "import contextpilot; print('ContextPilot', contextpilot.__version__)"
```
