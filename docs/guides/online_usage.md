# Online Usage

Online mode uses the **ContextPilot Index Server** for live scheduling. Two modes are available:

| Mode | Use Case | Cache Tracking |
|------|----------|----------------|
| **Stateless** | Per-batch scheduling, no state management | ❌ |
| **Stateful** | Live cache sync with the inference engine, eviction tracking | ✅ |

---

## How Scheduling Works

ContextPilot performs **two levels of optimization** to maximize KV-cache prefix sharing:

### 1. Inter-Context Reordering
Contexts with similar document IDs are scheduled together. For example:
```
Original order: [Context0, Context1, Context2, Context3]
Scheduled:      [Context0, Context2, Context1, Context3]  # 0 and 2 share docs, now adjacent
```

### 2. Intra-Context Reordering  
Within each context, document IDs are reordered so that **shared IDs appear first** as a common prefix:
```
Original:
  Context 0: [999, 1, 888, 2, 777, 3]   # IDs scattered
  Context 2: [3, 666, 1, 555, 2, 444]   # Same IDs, different order

After scheduling:
  Context 0: [1, 2, 3, 999, 888, 777]   # Shared {1,2,3} moved to front
  Context 2: [1, 2, 3, 666, 555, 444]   # Same prefix [1,2,3]!
```

This ensures adjacent contexts have **identical prefixes** that can be cached and reused.

> **Important:** Use `scheduled_contexts` (not `original_indices`) when building prompts to get the reordered document IDs.

---

## Stateless Mode

Stateless mode provides **optimal batch ordering** without tracking cache state. Each `/schedule` call is independent.

**Best for:** Simple batch scheduling, microservices architecture, per-request optimization.

### Start the Server

```bash
python -m contextpilot.server.http_server --port 8765 --stateless --infer-api-url http://localhost:30000
```

### Client Usage

```python
import requests

# Prepare contexts (each context = list of doc IDs for a query)
contexts = [
    [1, 5, 10, 15, 20],   # Query 0: uses docs 1, 5, 10, 15, 20
    [2, 5, 11, 16, 21],   # Query 1: shares doc 5 with Query 0
    [1, 5, 12, 17, 22],   # Query 2: shares docs 1, 5 with Query 0
    [3, 6, 13, 18, 23],   # Query 3: completely different
]

# Get optimal scheduling
response = requests.post("http://localhost:8765/schedule", json={
    "contexts": contexts,
    "alpha": 0.005,
    "use_gpu": False,
    "linkage_method": "average"
})
result = response.json()

print(f"Optimal order: {result['original_indices']}")  # e.g., [0, 2, 1, 3]
print(f"Number of groups: {result['num_groups']}")

# IMPORTANT: Use scheduled_contexts for building prompts!
# These have BOTH:
#   1. Contexts reordered (similar ones adjacent)
#   2. IDs within each context reordered (shared IDs as prefix)
scheduled_contexts = result['scheduled_contexts']
scheduled_order = result['original_indices']

# Build prompts using the reordered contexts
# scheduled_contexts[i] corresponds to original query at scheduled_order[i]
for i, reordered_ids in enumerate(scheduled_contexts):
    original_query_idx = scheduled_order[i]
    # Build prompt with reordered document IDs for maximum prefix sharing
    # prompt = build_prompt(queries[original_query_idx], reordered_ids)
    # response = sglang_client.generate(prompt)
    pass

# After inference, map results back to original order
# final_results[scheduled_order[i]] = sglang_results[i]
```

### Using the Python Client

```python
from contextpilot.server.http_client import ContextPilotIndexClient

client = ContextPilotIndexClient("http://localhost:8765")

result = client.schedule(
    contexts=[[1, 5, 10], [2, 5, 11], [1, 5, 12]],
    alpha=0.005,
    use_gpu=False
)

print(f"Scheduled order: {result['original_indices']}")
client.close()
```

---

## Stateful Mode

Stateful mode maintains a **live index** that tracks tokens and synchronizes with the inference engine's cache.

**Best for:** Long-running services, cache-aware scheduling, inference engine integration.

### Architecture

```
┌─────────────┐         ┌─────────────────────┐         ┌─────────────────┐
│   Client    │ ──────► │  ContextPilot Index │ ──────► │ Inference Engine│
│             │         │  Server (8765)      │         │ (30000)         │
└─────────────┘         └─────────────────────┘         └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │ LiveContextIndex│
                        │- Token tracking │
                        │- LRU eviction   │
                        └─────────────────┘
```

### Start the Server

```bash
python -m contextpilot.server.http_server \
    --port 8765 \
    --infer-api-url http://localhost:30000
```

### Step 1: Build the Index

```python
import requests

response = requests.post("http://localhost:8765/build", json={
    "contexts": [
        [1, 5, 10, 15, 20],
        [2, 5, 11, 16, 21],
        [1, 5, 12, 17, 22],
    ],
    "initial_tokens_per_context": 0,
    "alpha": 0.005,
    "use_gpu": False
})

result = response.json()
request_ids = result["request_ids"]
print(f"Built index with {len(request_ids)} contexts")
```

### Step 2: Send Requests via Proxy

The server proxies requests to the inference engine and tracks tokens automatically:

```python
response = requests.post("http://localhost:8765/v1/completions", json={
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "What is machine learning?",
    "max_tokens": 100,
    "rid": request_ids[0]
})

result = response.json()
print(result["choices"][0]["text"])
```

### Step 3: Eviction Sync

When using an inference engine with ContextPilot integration (e.g. SGLang with the `CONTEXTPILOT_INDEX_URL` env var), eviction sync is **automatic**. The engine's cache calls the `/evict` endpoint with evicted `request_ids` via a callback.

If you need manual eviction (e.g., for testing), use:

```python
requests.post("http://localhost:8765/evict", json={
    "request_ids": ["request_id_1", "request_id_2"]
})
```

---

## SGLang Integration

Stateful mode requires patching SGLang so its radix cache notifies ContextPilot on eviction.

### Install the SGLang Patch

```bash
# Automatic (recommended)
bash patches/sglang/apply_patch.sh

# Or manually:
SGLANG_PATH=$(python -c "import sglang; print(sglang.__path__[0])")

# Backup originals
cp $SGLANG_PATH/srt/mem_cache/radix_cache.py $SGLANG_PATH/srt/mem_cache/radix_cache.py.bak
cp $SGLANG_PATH/srt/mem_cache/common.py $SGLANG_PATH/srt/mem_cache/common.py.bak
cp $SGLANG_PATH/srt/mem_cache/cache_init_params.py $SGLANG_PATH/srt/mem_cache/cache_init_params.py.bak

# Copy patched files
cp patches/sglang/*.py $SGLANG_PATH/srt/mem_cache/
```

The patch adds an eviction callback to `RadixCache` that POSTs evicted `request_ids` to the ContextPilot server. Compatible with SGLang **0.5.x**. See [patches/sglang/README.md](../../patches/sglang/README.md) for details.

### Start SGLang with ContextPilot

```bash
# Start ContextPilot server first
python -m contextpilot.server.http_server \
    --port 8765 \
    --infer-api-url http://localhost:30000

# Start SGLang with ContextPilot integration enabled
CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B \
    --port 30000
```

### How It Works

When `CONTEXTPILOT_INDEX_URL` is set, SGLang integrates with ContextPilot at eviction time:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Eviction Sync Flow                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Cache Full → RadixCache.evict() → Callback invoked                 │
│                                      │                              │
│                                      ▼                              │
│                     POST /evict {"request_ids": ["rid1", "rid2"]}   │
│                                      │                              │
│                                      ▼                              │
│                     ContextPilot removes evicted requests from index│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **Request ID Tracking**: Each request has a unique `request_id` (e.g., `contextpilot_abc123`)
2. **Eviction Callback**: When the engine evicts cache entries, it notifies ContextPilot
3. **Index Sync**: ContextPilot removes evicted requests from its live index

---

## Server Endpoints

| Endpoint | Method | Mode | Description |
|----------|--------|------|-------------|
| `/health` | GET | Both | Health check |
| `/schedule` | POST | Stateless | Schedule a batch |
| `/build` | POST | Stateful | Build live index |
| `/deduplicate` | POST | Stateful | Multi-turn deduplication (lightweight) |
| `/evict` | POST | Stateful | Remove evicted requests |
| `/reset` | POST | Stateful | Reset index and conversation tracker |
| `/stats` | GET | Stateful | Get index statistics |

---

## Complete End-to-End Example with SGLang

For a complete working example that shows the entire workflow from documents → context scheduling → prompt building → SGLang inference, see:

📄 **[examples/stateless_sglang_e2e.py](../../examples/stateless_sglang_e2e.py)**

This example demonstrates:

1. **Document Retrieval** - Simulated RAG retrieval of relevant documents
2. **Context Tokenization** - Converting text to token IDs for scheduling
3. **ContextPilot Scheduling** - Optimal ordering for prefix sharing
4. **Prompt Building** - Constructing RAG prompts with context
5. **SGLang Inference** - Batch inference in scheduled order
6. **Result Reordering** - Mapping results back to original order

### Quick Preview

```python
from contextpilot.server.http_client import ContextPilotIndexClient
import requests

# 1. Tokenize your contexts
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
contexts = [tokenizer.encode(doc, add_special_tokens=False) for doc in documents]

# 2. Schedule with ContextPilot (stateless mode)
client = ContextPilotIndexClient("http://localhost:8765")
result = client.schedule(contexts=contexts, alpha=0.005)
scheduled_order = result["original_indices"]

# 3. Reorder and build prompts
scheduled_prompts = [build_rag_prompt(query, docs[i]) for i in scheduled_order]

# 4. Send to SGLang in scheduled order (maximizes prefix sharing)
sglang_response = requests.post("http://localhost:30000/generate", json={
    "text": scheduled_prompts,  # batch of prompts
    "sampling_params": {"max_new_tokens": 256}
})
scheduled_results = sglang_response.json()

# 5. Reorder results back to original order
final_results = [None] * len(scheduled_order)
for new_idx, orig_idx in enumerate(scheduled_order):
    final_results[orig_idx] = scheduled_results[new_idx]
```

### Running the Example

```bash
# Terminal 1: Start SGLang
python -m sglang.launch_server --model Qwen/Qwen3-4B --port 30000

# Terminal 2: Start ContextPilot  
python -m contextpilot.server.http_server --port 8765

# Terminal 3: Run the example
python examples/stateless_sglang_e2e.py
```

---

## Next Steps

- [Multi-Turn](multi_turn.md) - Conversation handling with deduplication
- [API Reference](../reference/api.md) - Full API documentation
