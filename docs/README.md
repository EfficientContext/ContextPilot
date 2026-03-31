# ContextPilot Documentation

## Getting Started

| Guide | Description |
|-------|-------------|
| [Installation](getting_started/installation.md) | System requirements and pip install |
| [Quick Start](getting_started/quickstart.md) | Your first ContextPilot pipeline in 5 minutes |
| [Docker](getting_started/docker.md) | Container deployment |

## Guides

| Guide | Description |
|-------|-------------|
| [OpenClaw Integration](guides/openclaw.md) | Proxy setup for OpenClaw agents |
| [How It Works](guides/how_it_works.md) | Reorder and deduplication explained |
| [Cache Synchronization](guides/cache_sync.md) | Self-hosted (eviction callbacks) vs cloud (TTL) |
| [Offline Usage](guides/offline_usage.md) | Batch processing without server |
| [Online Usage](guides/online_usage.md) | Index server (stateless and stateful modes) |
| [Multi-Turn Conversations](guides/multi_turn.md) | Context deduplication across turns |
| [Mem0 Integration](guides/mem0.md) | Memory-augmented chat with LoCoMo benchmark |
| [PageIndex Integration](guides/pageindex.md) | Tree-structured documents |
| [Mac + llama.cpp](guides/mac_llama_cpp.md) | Apple Silicon deployment |

## Benchmarks

| Benchmark | Description |
|-----------|-------------|
| [OpenClaw](benchmarks/openclaw.md) | 60 enterprise document analysis tasks on RTX 5090 |
| [RAG](benchmarks/rag.md) | MultihopRAG and NarrativeQA on Qwen3-32B and DeepSeek-R1 |

## Reference

| Document | Description |
|----------|-------------|
| [API Reference](reference/api.md) | Pipeline, InferenceConfig, HTTP endpoints |
| [Benchmarks](reference/benchmarks.md) | GPU vs CPU performance methodology |

## Quick Links

- [Examples](../examples/)
- [Paper](https://arxiv.org/abs/2511.03475)
