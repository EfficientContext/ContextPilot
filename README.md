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

<div align="center">
<img src="assets/ds_r1_result_horizontal.png" alt="Benchmark Results" width="800"/>
</div>

ContextPilot significantly speeds up DeepSeek-R1-671B offline inference on a GPU cluster with minimal accuracy impact: **64.68% vs 64.15% F1** on MultihopRAG and **41.08% vs 40.20% F1** on NarrativeQA. 

On consumer-grade or professional-grade GPUs (e.g., 4090, A6000), ContextPilot delivers consistent speedups across popular LLMs and long-context workloads—see the Evaluation section of the [Paper](https://arxiv.org/abs/2511.03475) for full performance results.

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

ContextPilot offers two core optimizations—**reorder** and **deduplicate**—to reduce long-context inefficiencies.

### Context Ordering

`cp.reorder()` places **shared blocks at the beginning** of the prompt so consecutive requests share the longest possible common prefix, enabling KV-cache reuse. To preserve answer quality, ContextPilot injects an **importance ranking** so the model still prioritizes blocks in their original relevance order.

### Context Deduplication

In multi-turn conversations, successive turns frequently gather **many of the same context blocks**, wasting tokens and compute.

`cp.deduplicate()` compares the current turn's context blocks against prior turns (tracked by `conversation_id`). Duplicate blocks are replaced with lightweight **reference hints** (e.g., *"See Doc 3 from previous context"*); only genuinely new blocks are sent in full — typically reducing duplicated tokens by **30-60%**. See [automatic context deduplication](docs/guides/multi_turn.md).

### Quick Start with Context Ordering

Add **one call** (`cp.reorder()`) before inference to rearrange context blocks so that shared content aligns into a common prefix, enabling cache reuse. An importance ranking in the prompt preserves accuracy.

| Mode | When to Use | How It Works |
|------|-------------|--------------|
| **Online** | Multi-turn (e.g., chatbot + [Mem0](https://github.com/mem0ai/mem0)) | Tracks previously cached blocks; moves overlapping ones to the prefix each turn |
| **Offline** | Batch / single-shot | Globally reorders and schedules all requests for maximum prefix sharing |

Both modes work with any OpenAI-compatible endpoint (vLLM, SGLang, etc.) — no changes to your inference deployment. They support both direct API calls (shown below) and HTTP server deployment (see the [online usage guide](docs/guides/online_usage.md)).

---

#### Accelerating Online Inference

Multi-turn chatbot with Mem0 or RAG where each turn's context blocks partially overlap. `cp.reorder()` moves shared blocks to the prefix so the engine reuses cached KV states.

```python
from openai import OpenAI
import contextpilot as cp

# vLLM default: http://localhost:8000/v1 | SGLang default: http://localhost:30000/v1
client = OpenAI(base_url="http://localhost:30000/v1", api_key="...")
cp_live = cp.ContextPilot(use_gpu=False)

# Simulated per-turn context blocks from Mem0 — partially overlapping across turns
turn_contexts = [
    ["Transformers use self-attention", "GPT is based on transformers", "BERT is bidirectional"],
    ["RNNs use hidden states", "GPT is based on transformers", "LSTMs solve vanishing gradients"],
    ["Attention computes QKV", "Transformers use self-attention", "GPT is based on transformers"],
]
queries = ["What are transformers?", "How do RNNs compare?", "Explain attention in detail."]

for turn_idx, (query, blocks) in enumerate(zip(queries, turn_contexts)):
    reordered, indices = cp_live.reorder(blocks)  # ← reorder for prefix sharing
    ctx = reordered[0]
    # Turn 2: "GPT is based on transformers" moves to prefix (cache hit)
    # Turn 3: "Transformers …", "GPT …" both move to prefix

    docs_section = "\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(ctx))
    pos = {doc: i + 1 for i, doc in enumerate(ctx)}
    importance_ranking = ">".join(str(pos[doc]) for doc in blocks if doc in pos)

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

> **Note:** When the engine evicts KV-cache entries under memory pressure, ContextPilot's index can go stale. Install the eviction patch for [SGLang](docs/guides/online_usage.md#sglang-integration) or [vLLM](docs/guides/online_usage.md#vllm-integration) to keep the index in sync. See the [online usage guide](docs/guides/online_usage.md).

---

#### Accelerating Offline Inference

Batch of requests with overlapping context blocks. `cp.reorder()` globally reorders blocks and schedules execution order so queries with similar contexts run consecutively, maximizing cache reuse. See the [offline usage guide](docs/guides/offline_usage.md) for details. Offline mode can also be deployed as an HTTP server without eviction sync — see [Stateless Mode](docs/guides/online_usage.md#stateless-mode).

```python
import asyncio
import openai
import contextpilot as cp

# vLLM default: http://localhost:8000/v1 | SGLang default: http://localhost:30000/v1
BASE_URL = "http://localhost:30000/v1"
cp_batch = cp.ContextPilot(use_gpu=False)

queries = ["What is AI?", "Explain neural networks", "What is deep learning?"]
all_contexts = [
    ["Doc about AI", "Doc about ML", "Doc about computing"],
    ["Doc about neural nets", "Doc about deep learning"],
    ["Doc about ML", "Doc about AI", "Doc about deep learning basics"],
]

reordered_ctx, order = cp_batch.reorder(all_contexts)  # ← global reorder + schedule

# Build requests in the scheduled order
messages_batch = []
for ctx, orig_idx in zip(reordered_ctx, order):
    docs = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(ctx))
    pos = {d: i + 1 for i, d in enumerate(ctx)}
    ranking = ">".join(str(pos[d]) for d in all_contexts[orig_idx] if d in pos)
    messages_batch.append(dict(
        model="Qwen/Qwen3-4B",
        messages=[
            {"role": "system", "content": f"Answer based on the documents.\n\n"
             f"<documents>\n{docs}\n</documents>\n\n"
             f"Importance ranking: {ranking}"},
            {"role": "user", "content": queries[orig_idx]},
        ],
    ))

# Send all requests concurrently
async def generate_all():
    ac = openai.AsyncOpenAI(base_url=BASE_URL, api_key="...")
    return await asyncio.gather(*[ac.chat.completions.create(**r) for r in messages_batch])

for resp, idx in zip(asyncio.run(generate_all()), order):
    print(f"Q: {queries[idx]}\nA: {resp.choices[0].message.content}\n")
```

### API Documentation

See [documentation](docs/README.md) for more APIs.

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
