# Installation

This guide covers installing ContextPilot and its dependencies.

## Requirements

- Python >= 3.10
- CUDA 12.x (for GPU-accelerated distance computation)
- An inference engine: [SGLang](https://github.com/sgl-project/sglang) (recommended) or any OpenAI-compatible server

## Install ContextPilot

```bash
git clone https://github.com/SecretSettler/ContextPilot.git
cd ContextPilot
pip install -e .
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

```bash
pip install "sglang==0.5.6"
```

For eviction sync with SGLang, set `CONTEXTPILOT_INDEX_URL` when launching (see [Online Usage Guide](../guides/online_usage.md#sglang-integration)).

## Verify Installation

```bash
python -c "import contextpilot; print('ContextPilot', contextpilot.__version__)"
```
