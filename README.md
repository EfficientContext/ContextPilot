<div align="center">
  <img src="assets/about.png" alt="ContextPilot Logo" width="800"/>

  <h1><strong>ContextPilot: Efficient Long Context Inference with Context Reuse</strong></h1>

  [![Python](https://img.shields.io/badge/python-≥3.10-blue)](https://www.python.org/)
  [![PyPI](https://img.shields.io/pypi/v/contextpilot)](https://pypi.org/project/contextpilot/)
  [![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

</div>

--------------------------------------------------------------------------------

| [**Documentation**](docs/README.md) | [**Examples**](examples/) | [**Benchmarks**](docs/reference/benchmarks.md) |

## News

- [2026/02] ContextPilot v0.3.2 released, supporting [PageIndex](https://github.com/VectifyAI/PageIndex) and [Mem0](https://github.com/mem0ai/mem0).
- [2026/01] ContextPilot has been accepted to MLSys 2026 🎉! See you in Bellevue, WA, USA.
- [2025/12] ContextPilot v0.2.0 released.

## About

ContextPilot is a fast optimization system on context engineering layer for agentic workloads:
1. **High Throughput & Cache Hit Ratio**: Boosting prefill throughput and prefix cache hit ratio with intelligent context reuse.
2. **Strong Compatibility**: Strong compatibility with existing popular RAG libraries (PageIndex), Agentic memory layer (Mem0), KV cache optimization engine (LMCache), and Inference engines (vLLM and SGLang).
3. **Negligible Accuracy Loss**: Achieving significant performance improvements with minimal to no accuracy degradation across various benchmarks.
3. **Widely Tested**: Tested with a wide range of RAG and Agentic AI applications.

## Target Workloads

1. **Trending Topic QA** — Search and generation for breaking news and hot topics beyond model knowledge
2. **Closed-Domain Long-Context QA** — QA over specialized corpora (novels, financial reports, legal documents) with retrieval or in-context search
3. **Large-Batch Long-Context Execution** — High-throughput inference where many requests share overlapping contexts; ContextPilot maximizes prefix reuse regardless of the search method
4. **Multi-Turn Conversations with Long-Term Memory** — Persistent context reuse across turns (e.g. [Mem0](https://github.com/mem0ai/mem0))

## Benchmark and Performance

### System Performance

<div align="center">
<img src="assets/deepseek_r1_results.png" alt="Benchmark Results" width="600"/>
</div>

ContextPilot (Stateless) on DeepSeek-R1 maintains accuracy compared to SGLang, achieving 64.68% vs 64.15% F1 on MultihopRAG and 41.08% vs 40.20% F1 on NarrativeQA.

### Accuracy on MT-RAG Benchmark (Online Scheduling)

<div align="center">

| Method | Qwen3-4B | Llama3.1-8B | Qwen3-30B-A3B |
|--------|----------|-------------|-----------|
| LMCache | 62.56 | **68.46** | 75.12 |
| CacheBlend | 50.33 | 56.52 | X |
| RadixCache | 62.56 | **68.46** | 75.12 |
| **ContextPilot** | **64.27** | 68.12 | **75.81** |

</div>

ContextPilot delivers **4-13x** improvements in cache hit rates and **1.5-3.5x** reductions in prefill latency for large-batch RAG workloads, while maintaining or improving accuracy.

**Furthermore**, ContextPilot has been tested to reduce input token costs by around **36%** with GPT-5.2.

See [Benchmarks](docs/reference/benchmarks.md) in the documentation for GPU vs CPU performance analysis and detailed benchmark methodology.

## Getting Started

### Installation

**Requirements:** Python >= 3.10

```bash
pip install contextpilot
```

Or install from source:
```bash
git clone https://github.com/Edinburgh-AgenticAI/ContextPilot.git
cd ContextPilot
pip install -e .
```

More [detailed installation instructions](docs/getting_started/installation.md) are available in the docs.

### Quick Start

**Stateful** — `ContextPilot` tracks cached state across turns so
overlapping documents are moved to the prefix for KV-cache reuse:

```python
from openai import OpenAI
import contextpilot as cp

client = OpenAI(base_url="http://localhost:30000/v1", api_key="...")
cp_live = cp.ContextPilot(use_gpu=False)

# Simulated per-turn memory search (e.g. from mem0)
# Each turn retrieves different but partially overlapping documents
turn_memories = [
    ["Transformers use self-attention", "GPT is based on transformers", "BERT is bidirectional"],
    ["RNNs use hidden states", "GPT is based on transformers", "LSTMs solve vanishing gradients"],
    ["Attention computes QKV", "Transformers use self-attention", "GPT is based on transformers"],
]
queries = ["What are transformers?", "How do RNNs compare?", "Explain attention in detail."]

for turn_idx, (query, mems) in enumerate(zip(queries, turn_memories)):
    # 1. Reorder for prefix sharing (handles cold start & incremental)
    [ctx], order = cp_live.reorder([mems])   # single request per turn
    # Turn 2: "GPT is based on transformers" ← moved to prefix (shared with turn 1)
    # Turn 3: "Transformers …", "GPT …"     ← both moved to prefix

    # 2. Generate answer with reordered context
    docs_section = "\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(ctx))
    importance_ranking = ">".join(
        str(ctx.index(doc) + 1) for doc in mems if doc in ctx
    )
    response = client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=[
            {"role": "system", "content": (
                f"Answer the question based on the provided documents.\n\n"
                f"<documents>\n{docs_section}\n</documents>\n\n"
                f"Read the documents in this importance ranking: {importance_ranking}\n"
                f"Prioritize information from higher-ranked documents."
            )},
            {"role": "user", "content": query},
        ],
    )
    print(f"[Turn {turn_idx+1}] Q: {query}")
    print(f"A: {response.choices[0].message.content}\n")
```

> **Note:** Stateful mode works without eviction sync — `ContextPilot` tracks the previous ordering and reorders new contexts to maximize prefix cache hits. For production deployments with limited KV-cache capacity, install the [SGLang eviction patch](docs/guides/online_usage.md#sglang-integration) to keep the index in sync. See the [online usage guide](docs/guides/online_usage.md) for HTTP server setup.

**Offline / Online Stateless** — same API, just pass the full batch at once:

```python
from openai import OpenAI
import contextpilot as cp

client = OpenAI(base_url="http://localhost:30000/v1", api_key="...") # Your inference engine URL and API key
cp_batch = cp.ContextPilot(use_gpu=False)

queries = ["What is AI?", "Explain neural networks", "What is deep learning?"]
all_contexts = [
    ["Doc about AI", "Doc about ML", "Doc about computing"],
    ["Doc about neural nets", "Doc about deep learning"],
    ["Doc about ML", "Doc about AI", "Doc about deep learning basics"],
]

# One call: builds index, reorders docs for prefix sharing, and schedules execution order
reordered, order = cp_batch.reorder(all_contexts)

# Build all prompts in optimized order
messages_batch = []
for ctx, orig_idx in zip(reordered, order):
    docs_section = "\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(ctx))
    importance_ranking = ">".join(
        str(ctx.index(doc) + 1) for doc in all_contexts[orig_idx] if doc in ctx
    )
    messages_batch.append({
        "model": "Qwen/Qwen3-4B",
        "messages": [
            {"role": "system", "content": (
                f"Answer the question based on the provided documents.\n\n"
                f"<documents>\n{docs_section}\n</documents>\n\n"
                f"Read the documents in this importance ranking: {importance_ranking}\n"
                f"Prioritize information from higher-ranked documents."
            )},
            {"role": "user", "content": queries[orig_idx]},
        ],
    })

# Send concurrently — inference engine processes them in order for max cache reuse
import asyncio, openai

async def generate_all(batch):
    aclient = openai.AsyncOpenAI(base_url="http://localhost:30000/v1", api_key="...")
    tasks = [aclient.chat.completions.create(**req) for req in batch]
    return await asyncio.gather(*tasks)

responses = asyncio.run(generate_all(messages_batch))
for resp, orig_idx in zip(responses, order):
    print(f"Q: {queries[orig_idx]}\nA: {resp.choices[0].message.content}\n")
```

> For online stateless scheduling via HTTP server, see the [online usage guide](docs/guides/online_usage.md).

## Documentation

Check out the ContextPilot [documentation](docs/README.md) for comprehensive guides.

## Examples

Go hands-on with our [examples](examples/), demonstrating how to address different use cases with ContextPilot.

## Contributing

We welcome and value all contributions! Please feel free to submit issues and pull requests.

## Citation
We will include the paper citation soon!
