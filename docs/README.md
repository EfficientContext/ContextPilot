# ContextPilot Documentation

Welcome to the ContextPilot documentation. This guide covers everything you need to get started and make the most of ContextPilot.

## Getting Started

| Guide | Description |
|-------|-------------|
| [Installation](getting_started/installation.md) | System requirements and pip install |
| [Quick Start](getting_started/quickstart.md) | Your first ContextPilot pipeline in 5 minutes |

## User Guides

| Guide | Description |
|-------|-------------|
| [Offline Usage](guides/offline_usage.md) | Batch processing without server |
| [Online Usage](guides/online_usage.md) | Index server (stateless & stateful modes) |
| [Eviction Patches](guides/online_usage.md#inference-engine-integration) | **Required for stateful mode** — eviction callback for KV cache sync (SGLang & vLLM) |
| [Multi-Turn Conversations](guides/multi_turn.md) | Context deduplication across turns (30-60% savings) |
| [PageIndex Integration](guides/pageindex.md) | Tree-structured documents → ContextPilot scheduling |
| [mem0 Integration](guides/mem0.md) | LoCoMo benchmark with mem0 memory backend |

## Reference

| Document | Description |
|----------|-------------|
| [API Reference](reference/api.md) | Pipeline, InferenceConfig, HTTP endpoints |
| [Benchmarks](reference/benchmarks.md) | GPU vs CPU performance analysis and methodology |


## Quick Links

- [Examples](../examples/)
