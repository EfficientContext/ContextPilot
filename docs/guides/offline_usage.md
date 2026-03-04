# Offline Usage

Offline mode is best for **batch processing** where you have all queries upfront and want to maximize KV-cache prefix sharing across the entire batch — no server, no live state.

## How It Works

ContextPilot performs two levels of optimization:

1. **Intra-context reordering** — shared documents are moved to the front of each context so adjacent requests have a common token prefix
2. **Inter-context reordering** — requests sharing the most documents are scheduled consecutively so the prefix is already cached when they run

For example, if Query A retrieves docs `[5, 1, 8, 2]` and Query B retrieves `[2, 9, 1, 5]`, after optimization:
- Query A: `[1, 2, 5, 8]` (shared IDs first)
- Query B: `[1, 2, 5, 9]` (same prefix `[1, 2, 5]`!)

---

## Prerequisites

Start your inference engine with prefix caching enabled:

```bash
# SGLang
sglang serve --model-path Qwen/Qwen3-4B --port 30000

# vLLM
vllm serve Qwen/Qwen3-4B --port 30000 --enable-prefix-caching

# llama.cpp (brew install llama.cpp)
contextpilot-llama-server -m models/Qwen3-4B-Q4_K_M.gguf --port 30000 --cache-reuse 256
```

---

## Two-Line Integration

The simplest way to add ContextPilot to an existing pipeline:

```python
import contextpilot as cp
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

# Single request — reorders docs for max prefix sharing
docs = get_retrieved_docs(query)   # your RAG retriever
messages = cp.optimize(docs, query)
response = client.chat.completions.create(model="Qwen/Qwen3-4B", messages=messages)
```

---

## Batch Processing

`optimize_batch` schedules an entire batch in the globally optimal execution order — queries sharing the most documents execute consecutively:

```python
import contextpilot as cp
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

queries = ["What is machine learning?", "Explain neural networks", "What is deep learning?"]
all_docs = [get_retrieved_docs(q) for q in queries]   # your RAG retriever

# Reorder entire batch at once
messages_batch, original_indices = cp.optimize_batch(all_docs, queries)

print(f"Scheduled order: {original_indices}")  # e.g. [0, 2, 1] — most overlap runs first

# Run inference in scheduled order
answers = [""] * len(queries)
for messages, orig_idx in zip(messages_batch, original_indices):
    response = client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=messages,
        max_tokens=256,
    )
    answers[orig_idx] = response.choices[0].message.content

# answers[i] corresponds to queries[i]
```

---

## Multi-Turn Conversations

Pass a stable `conversation_id` so ContextPilot deduplicates documents already seen in earlier turns:

```python
import uuid
import contextpilot as cp

conversation_id = f"conv-{uuid.uuid4().hex[:8]}"

for query in conversation_turns:
    docs = get_retrieved_docs(query)
    messages = cp.optimize(docs, query, conversation_id=conversation_id)
    response = client.chat.completions.create(model="Qwen/Qwen3-4B", messages=messages)
```

---

## Pipeline API

For a higher-level interface that wires retrieval + optimization + inference in one call, use `RAGPipeline`:

```python
from contextpilot.pipeline import RAGPipeline, InferenceConfig

pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_contextpilot=True,
    inference=InferenceConfig(
        model_name="Qwen/Qwen3-4B",
        base_url="http://localhost:30000",
        max_tokens=256,
    )
)

results = pipeline.run(
    queries=["What is machine learning?", "Explain neural networks"],
    top_k=20,
    generate_responses=True,
)

for gen in results["generation_results"]:
    if gen["success"]:
        print(gen["generated_text"])
```

Supported retrievers: `"bm25"`, `"faiss"`, `"mem0"`, `"pageindex"`.

---

## Complete End-to-End Example

See **[examples/stateless_sglang_e2e.py](../../examples/stateless_sglang_e2e.py)** for a full working example covering document retrieval, tokenization, ContextPilot scheduling, prompt building, inference, and result reordering.

---

## Next Steps

- [Online Usage](online_usage.md) — Live index server with eviction tracking (stateful mode)
- [Multi-Turn](multi_turn.md) — Cross-turn deduplication in detail
- [API Reference](../reference/api.md) — Full API documentation
