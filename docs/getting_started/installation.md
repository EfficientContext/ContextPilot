# Installation

This guide covers installing ContextPilot and its dependencies.

## Requirements

- Python >= 3.10
- CUDA 12.x (for GPU-accelerated distance computation)
- An inference engine: [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), or any OpenAI-compatible server

## Install ContextPilot

```bash
git clone https://github.com/SecretSettler/ContextPilot.git
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

> **Note:** flash-attn and flashinfer are backend-specific dependencies not installed by ContextPilot.
> Install them separately per your CUDA / PyTorch version:
> - flash-attn: https://github.com/Dao-AILab/flash-attention/releases
> - flashinfer: https://docs.flashinfer.ai/installation.html

## Verify Installation

```bash
python -c "import contextpilot; print('ContextPilot', contextpilot.__version__)"
```
