<div align="center">
  <img src="assets/about.png" alt="ContextPilot Logo" width="600"/>

  <h2><strong>ContextPilot: Fast Long-Context Inference via Context Reuse</strong></h2>

  [![Python](https://img.shields.io/badge/python-≥3.10-blue)](https://www.python.org/)
  [![PyPI](https://img.shields.io/pypi/v/contextpilot)](https://pypi.org/project/contextpilot/)
  [![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

  <p><strong>4–13× cache hits | 1.5–3× faster prefill | ~36% token savings</strong> across vLLM, SGLang, RAG, AI Agents, and more.</p>

</div>

--------------------------------------------------------------------------------

| [**Documentation**](docs/README.md) | [**Examples**](examples/) | [**Benchmarks**](docs/reference/benchmarks.md) |

## News

- [2026/02] ContextPilot v0.3.2 released, supporting [PageIndex](https://github.com/VectifyAI/PageIndex) and [Mem0](https://github.com/mem0ai/mem0).
- [2026/01] ContextPilot has been accepted to MLSys 2026 🎉! See you in Bellevue, WA, USA.

## About

Long-context workloads (RAG, memory chat, tool-augmented agents) prepend many context blocks. Across requests, these blocks often overlap but get reordered or duplicated, changing token prefixes and triggering cache misses and redundant KV recomputation. Common examples include (1) Trending Topic QA, (2) Closed-Domain Long-Context QA, (3) Batched Long-Context Inference, (4) multi-turn conversations with long-term memory and many more.

ContextPilot sits between context assembly and inference to maximize prefix reuse and remove duplicates:

1. **Higher throughput & cache hits** — boosts prefill throughput and prefix cache hit ratio via context reuse.  
2. **Drop-in solutions** — works with [PageIndex](https://github.com/VectifyAI/PageIndex), [Mem0](https://github.com/mem0ai/mem0), [LMCache](https://github.com/LMCache/LMCache), and backends like [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang).  
3. **No compromise in reasoning quality** — can even improve with extremely long contexts.
4. **Widely tested** — validated across diverse RAG and agentic workloads.

It maintains a **Context Index** of cached content, then per request applies **Reorder** (align shared blocks into a common prefix) and/or **Deduplicate** (replace repeats with reference hints), plus **cache-aware scheduling** to maximize prefix sharing. The optimized prompt is sent via the OpenAI-compatible API; `POST /evict` keeps the index synced when KV cache is reclaimed. See its design overview below.

<div align="center">
<img src="assets/system_description.png" alt="ContextPilot Architecture" width="600"/>
</div>

> For more design details, see [Paper](https://arxiv.org/abs/2511.03475) and [Documentation](docs/README.md).

## Performance at a Glance

ContextPilot is validated across three representative settings: single-node academic RAG, multi-node production MoE inference, and multi-turn memory-augmented chat. In every case it delivers significant speedups with comparable answer quality.

**Qwen3-32B on 4×A6000** — single-node academic RAG with a 32B model on consumer GPUs.

| Benchmark | Method | Prefill TP (tok/s) | Cache Hit | F1 (%) |
|-----------|--------|--------------------|-----------|--------|
| MultihopRAG | SGLang | 7,290 | 4.64% | 60.42 |
| MultihopRAG | **ContextPilot (Ours)** | **14,214** | **33.97%** | **64.39** |
| NarrativeQA | SGLang | 7,921 | 5.91% | 28.41 |
| NarrativeQA | **ContextPilot (Ours)** | **12,117** | **20.82%** | **29.64** |

**~2× prefill throughput** and **4–7× cache hit ratio** with improved F1 on both benchmarks.

**DeepSeek-R1-671B on 16×H20** — production-scale 671B MoE inference on a multi-node GPU cluster.

| Benchmark | Method | Prefill TP (tok/s) | Cache Hit | F1 (%) |
|-----------|--------|--------------------|-----------|--------|
| MultihopRAG | SGLang | 9,636 | 5.12% | 64.15 |
| MultihopRAG | **ContextPilot (Ours)** | **17,498** | **60.37%** | **64.68** |
| NarrativeQA | SGLang | 8,687 | 6.08% | 40.20 |
| NarrativeQA | **ContextPilot (Ours)** | **13,201** | **38.24%** | **41.08** |

**1.5–1.8× prefill throughput** and **6–12× cache hit ratio** with comparable or better F1.

**Qwen3-4B on 1×A6000** — multi-turn memory chat with [Mem0](https://github.com/mem0ai/mem0) on the [LoCoMo](https://github.com/snap-research/locomo) benchmark.

| Context Size | Method | TTFT (s) | LLM Judge |
|--------------|--------|----------|-----------|
| 20 memories | SGLang | 0.0377 | 0.440 |
| 20 memories | **ContextPilot (Ours)** | **0.0315** | **0.460** |
| 100 memories | SGLang | 0.1012 | 0.437 |
| 100 memories | **ContextPilot (Ours)** | **0.0554** | 0.420 |

**1.2–1.8× faster time-to-first-token** with comparable answer quality. We include the 100-memory setting to evaluate long-context performance, as the full LoCoMo conversation contains only ~26k tokens of memory context. Judge scores at 100 memories are lower than at 20 for both methods, as more memories introduces noise that dilutes relevant context. The 0.420 judge score is without annotation.

## Installation

**Requirements:** Python >= 3.10

```bash
pip install contextpilot
```

Or install from source:
```bash
git clone https://github.com/EfficientContext/ContextPilot.git
cd ContextPilot
pip install -e .
```

More [detailed installation instructions](docs/getting_started/installation.md) are available in the docs.

## Getting Started

### Quick Start with Context Ordering

Add **one call** (`cp_instance.optimize()`) before inference to rearrange context blocks so that shared content aligns into a common prefix, enabling cache reuse. An importance ranking in the prompt preserves accuracy.

| Mode | When to Use | How It Works |
|------|-------------|--------------|
| **Online** | Multi-turn (e.g., chatbot + [Mem0](https://github.com/mem0ai/mem0)) | Tracks previously cached blocks; moves overlapping ones to the prefix each turn |
| **Offline** | Batch / single-shot | Globally reorders and schedules all requests for maximum prefix sharing |

Both modes work with any OpenAI-compatible endpoint (vLLM, SGLang, etc.) — no changes to your inference deployment. They support both direct API calls (shown below) and HTTP server deployment (see the [online usage guide](docs/guides/online_usage.md)).

---

#### Accelerating Online Inference

Multi-turn chatbot with Mem0 or RAG where each turn's context blocks partially overlap. `cp_instance.optimize()` moves shared blocks to the prefix so the engine reuses cached KV states.

```python
from openai import OpenAI
# Step 1: Import ContextPilot
import contextpilot as cp

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
# Step 2: Create a ContextPilot instance
cp_instance = cp.ContextPilot(use_gpu=False)

for query in queries:
    contexts = get_contexts(query)                         # Mem0, Retriever, ...
    # Step 3: Optimize context ordering and build ready-to-use messages
    messages = cp_instance.optimize(contexts, query)

    response = client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=messages,
    )
    print(f"Q: {query}\nA: {response.choices[0].message.content}\n")
```

> **Note:** When the engine evicts KV-cache entries under memory pressure, ContextPilot's index can go stale. Install the eviction patch for [SGLang](docs/guides/online_usage.md#sglang-integration) or [vLLM](docs/guides/online_usage.md#vllm-integration) to keep the index in sync. See the [online usage guide](docs/guides/online_usage.md).

---

#### Accelerating Offline Inference

Batch of requests with overlapping context blocks. `cp_instance.optimize_batch()` globally reorders blocks and schedules execution order so queries with similar contexts run consecutively, maximizing cache reuse. See the [offline usage guide](docs/guides/offline_usage.md) for details. Offline mode can also be deployed as an HTTP server without eviction sync — see [Stateless Mode](docs/guides/online_usage.md#stateless-mode).

```python
import asyncio
import openai
# Step 1: Import ContextPilot
import contextpilot as cp

BASE_URL = "http://localhost:30000/v1"
# Step 2: Create a ContextPilot instance
cp_instance = cp.ContextPilot(use_gpu=False)

all_contexts = [get_contexts(q) for q in queries]          # Mem0, Retriever, ...
# Step 3: Optimize — reorder, schedule, and build prompts in one call
messages_batch, order = cp_instance.optimize_batch(all_contexts, queries)

# Send all requests concurrently
async def generate_all():
    ac = openai.AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY")
    return await asyncio.gather(*[ac.chat.completions.create(
        model="Qwen/Qwen3-4B", messages=m
    ) for m in messages_batch])

for resp, idx in zip(asyncio.run(generate_all()), order):
    print(f"Q: {queries[idx]}\nA: {resp.choices[0].message.content}\n")
```

For a detailed walkthrough with concrete examples, see the [Quick Start Guide](docs/getting_started/quickstart.md). For more fine-grained control, you can also use `cp_instance.reorder()` and `cp_instance.deduplicate()` directly — see the [API reference](docs/reference/api.md) and [multi-turn deduplication guide](docs/guides/multi_turn.md).

### Adoption Examples

See many useful adoption examples: [Mem0 integration](docs/guides/mem0.md), [PageIndex RAG](docs/guides/pageindex.md), [offline batch scheduling](docs/guides/offline_usage.md), and [multi-turn deduplication](docs/guides/multi_turn.md).

## Citation
```bibtex
@inproceedings{contextpilot2026,
  title     = {ContextPilot: Fast Long-Context Inference via Context Reuse},
  author    = {Jiang, Yinsicheng and Huang, Yeqi and Cheng, Liang and Deng, Cheng and Sun, Xuan and Mai, Luo},
  booktitle = {Proceedings of the 9th Conference on Machine Learning and Systems (MLSys 2026)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2511.03475}
}
```

## Contributing

We welcome and value all contributions! Please feel free to submit issues and pull requests.
