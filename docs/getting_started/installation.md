# Installation

This guide covers installing ContextPilot and its dependencies.

## Requirements

- Python >= 3.10
- An inference engine: SGLang (recommended), vLLM, or LMCache

## Install ContextPilot

```bash
git clone https://github.com/SecretSettler/ContextPilot.git
cd ContextPilot
pip install -e .
```

## Install Inference Engine

### SGLang (Recommended)

```bash
pip install "sglang[all]"
```

See also: [SGLang Installation Guide](https://docs.sglang.ai/get_started/install.html)

### vLLM

```bash
pip install vllm
```

See also: [vLLM Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation/)

## Optional: Install FAISS

FAISS is required for semantic search with embeddings.

```bash
# GPU support (recommended)
conda install conda-forge::faiss-gpu

# CPU only
conda install conda-forge::faiss-cpu
```

## Docker

For containerized deployment:

```bash
docker pull seanjiang01/contextpilot-sgl-v0.5.5:latest
docker run -d --gpus all --name contextpilot seanjiang01/contextpilot-sgl-v0.5.5:latest
docker exec -it contextpilot bash
```

## Verify Installation

```python
from contextpilot.pipeline import RAGPipeline
print("ContextPilot installed successfully!")
```

## Next Steps

- [Quick Start](quickstart.md) - Run your first ContextPilot pipeline
- [Offline Usage](../guides/offline_usage.md) - Batch processing examples
