# Offline Usage

Offline mode is best for **batch processing** where you have all queries upfront and want to maximize KV-cache reuse across them — no server required.

## How ContextPilot Optimizes Batches

ContextPilot performs **two levels of optimization** to maximize KV-cache prefix sharing:

1. **Inter-Context Reordering**: Queries with overlapping context blocks are scheduled together
2. **Intra-Context Reordering**: Context blocks within each query are reordered so shared blocks appear first as a common prefix

For example, if Query A has `["block_C", "block_A", "block_D", "block_B"]` and Query B has `["block_B", "block_E", "block_A", "block_C"]`, after optimization:
- Query A: `["block_A", "block_B", "block_C", "block_D"]` (shared blocks first)
- Query B: `["block_A", "block_B", "block_C", "block_E"]` (same prefix — cache hit!)

## Prerequisites

Start your inference engine:

```bash
# SGLang:
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 30000

# or vLLM:
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 30000 \
    --enable-prefix-caching
```

---

## Using `cp.optimize_batch()` (Simplest)

Pass your context blocks and queries — ContextPilot handles reordering and returns ready-to-use OpenAI messages in the optimal execution order.

```python
import asyncio
import openai
import contextpilot as cp

BASE_URL = "http://localhost:30000/v1"
engine = cp.ContextPilot(use_gpu=False)

queries = ["What is AI?", "Explain neural networks", "What is deep learning?"]
all_contexts = [
    ["AI is the simulation of human intelligence", "Machine learning is a subset of AI", "Deep learning uses neural networks"],
    ["Neural networks are inspired by the brain", "Machine learning is a subset of AI", "Backpropagation trains neural networks"],
    ["Deep learning uses neural networks", "Machine learning is a subset of AI", "GPUs accelerate deep learning training"],
]

# Returns messages in scheduled order + the original index mapping
messages_batch, order = engine.optimize_batch(all_contexts, queries)

async def generate_all():
    client = openai.AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY")
    return await asyncio.gather(*[
        client.chat.completions.create(model="Qwen/Qwen2.5-7B-Instruct", messages=m)
        for m in messages_batch
    ])

for resp, idx in zip(asyncio.run(generate_all()), order):
    print(f"Q: {queries[idx]}\nA: {resp.choices[0].message.content}\n")
```

`messages_batch[i]` corresponds to `queries[order[i]]` — send them in this order to the inference engine for maximum prefix sharing, then use `order` to map results back.

---

## Using `cp.reorder()` (Manual Control)

Use `reorder()` when you need full control over prompt construction — it returns reordered context blocks and the execution order, and you build the prompts yourself.

```python
import asyncio
import openai
import contextpilot as cp

BASE_URL = "http://localhost:30000/v1"
engine = cp.ContextPilot(use_gpu=False)

queries = ["What is AI?", "Explain neural networks", "What is deep learning?"]
all_contexts = [
    ["AI is the simulation of human intelligence", "Machine learning is a subset of AI", "Deep learning uses neural networks"],
    ["Neural networks are inspired by the brain", "Machine learning is a subset of AI", "Backpropagation trains neural networks"],
    ["Deep learning uses neural networks", "Machine learning is a subset of AI", "GPUs accelerate deep learning training"],
]

# reordered[i] = reordered blocks for the i-th scheduled query
# order[i]     = index into the original queries list
reordered, order = engine.reorder(all_contexts)

def build_prompt(query, blocks):
    context_text = "\n".join(f"[{i+1}] {b}" for i, b in enumerate(blocks))
    return [
        {"role": "system", "content": f"Answer based on the context:\n{context_text}"},
        {"role": "user", "content": query},
    ]

messages_batch = [build_prompt(queries[order[i]], reordered[i]) for i in range(len(order))]

async def generate_all():
    client = openai.AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY")
    return await asyncio.gather(*[
        client.chat.completions.create(model="Qwen/Qwen2.5-7B-Instruct", messages=m)
        for m in messages_batch
    ])

results = [None] * len(queries)
for resp, idx in zip(asyncio.run(generate_all()), order):
    results[idx] = resp.choices[0].message.content

for q, a in zip(queries, results):
    print(f"Q: {q}\nA: {a}\n")
```

---

## RAG Pipeline (with Built-in Retrieval)

If you have a document corpus and want ContextPilot to handle retrieval + optimization in one call, use `RAGPipeline`:

```python
from contextpilot.pipeline import RAGPipeline, InferenceConfig

pipeline = RAGPipeline(
    retriever="bm25",          # or "faiss" for semantic search
    corpus_path="corpus.jsonl",
    use_contextpilot=True,
    inference=InferenceConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:30000",
        max_tokens=256,
    )
)

results = pipeline.run(
    queries=["What is machine learning?", "Explain neural networks", "What is deep learning?"],
    top_k=20,
    generate_responses=True,
)

for gen_result in results["generation_results"]:
    if gen_result["success"]:
        print(gen_result["generated_text"][:200])
```

See the [API Reference](../reference/api.md) for full `RAGPipeline` options including FAISS retrieval, step-by-step control, and saving results.

---

## Next Steps

- [Online Usage](online_usage.md) - Live index server for stateful cache tracking
- [Multi-Turn](multi_turn.md) - Context deduplication across conversation turns
- [API Reference](../reference/api.md) - Full API documentation
