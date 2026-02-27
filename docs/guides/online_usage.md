# Online Usage

Online mode uses the **ContextPilot Index Server** for live reordering. Both modes use a single `POST /reorder` endpoint:

| Mode | Use Case | Cache Tracking |
|------|----------|----------------|
| **Stateless** (`--stateless`) | Per-batch reordering, no state management | ❌ |
| **Stateful** (default) | Live cache sync with the inference engine, eviction tracking | ✅ |

The server auto-dispatches based on its mode — your client code is the same either way.

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

> **Important:** Use `reordered_contexts` (not `original_indices`) when building prompts to get the reordered document IDs.

---

## Stateless Mode

Stateless mode provides **optimal batch ordering** without tracking cache state. Each `/reorder` call is independent.

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

# One call — reorder for optimal prefix sharing
response = requests.post("http://localhost:8765/reorder", json={
    "contexts": contexts
})
result = response.json()

# Use the reordered contexts and execution order
scheduled_contexts = result['reordered_contexts']
scheduled_order = result['original_indices']
print(f"Optimal order: {scheduled_order}")  # e.g., [0, 2, 1, 3]

# Build prompts using the reordered contexts
# scheduled_contexts[i] corresponds to original query at scheduled_order[i]
for i, reordered_ids in enumerate(scheduled_contexts):
    original_query_idx = scheduled_order[i]
    # Build prompt with reordered document IDs for maximum prefix sharing
    # prompt = build_prompt(queries[original_query_idx], reordered_ids)
    # response = inference_client.generate(prompt)
    pass

# After inference, map results back to original order
# final_results[scheduled_order[i]] = results[i]
```

### Using the Python Client

```python
from contextpilot.server.http_client import ContextPilotIndexClient

client = ContextPilotIndexClient("http://localhost:8765")

reordered, order = client.reorder(
    contexts=[[1, 5, 10], [2, 5, 11], [1, 5, 12]]
)

print(f"Scheduled order: {order}")
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
                        │ ContextPilot     │
                        │- Token tracking  │
                        │- LRU eviction    │
                        └─────────────────┘
```

### Inference Engine Integration

Stateful mode requires the inference engine to notify ContextPilot when KV-cache entries are evicted. **No engine patches are needed** — ContextPilot automatically hooks into SGLang at runtime via a `.pth` import hook.

#### SGLang (automatic, zero-patch)

Just set the `CONTEXTPILOT_INDEX_URL` environment variable when starting SGLang:

```bash
CONTEXTPILOT_INDEX_URL=http://localhost:8765 sglang serve --model-path Qwen/Qwen3-4B --port 30000
```

That's it. ContextPilot's `.pth` hook automatically monkey-patches SGLang's `RadixCache` at import time to add eviction tracking. You will see this in the SGLang logs:

```
[ContextPilot] Applying monkey-patches to SGLang RadixCache …
[ContextPilot] SGLang RadixCache monkey-patched successfully
```

**Requirements:** If you used `pip install -e .`, run `python -m contextpilot.install_hook` once to install the `.pth` file.

**Distributed setup** (SGLang and ContextPilot on different machines): You don't need to install the full `contextpilot` package on the SGLang machine. Copy the standalone install script instead:

```bash
# On the SGLang machine:
wget https://raw.githubusercontent.com/EfficientContext/ContextPilot/main/scripts/install_engine_hook.py
wget https://raw.githubusercontent.com/EfficientContext/ContextPilot/main/contextpilot/_sglang_hook.py
python install_engine_hook.py

# Then start SGLang pointing to the remote ContextPilot server:
CONTEXTPILOT_INDEX_URL=http://<contextpilot-host>:8765 sglang serve --model-path Qwen/Qwen3-4B
```

Compatible with SGLang **0.5.x**.

#### vLLM (automatic, zero-patch)

Same approach — just set the environment variable:

```bash
CONTEXTPILOT_INDEX_URL=http://localhost:8765 vllm serve Qwen/Qwen3-4B --port 30000 --enable-prefix-caching
```

ContextPilot's `.pth` hook automatically monkey-patches vLLM's `BlockPool` at import time. You will see this in the vLLM logs:

```
[ContextPilot] Applying monkey-patches to vLLM BlockPool …
[ContextPilot] vLLM BlockPool monkey-patched successfully
```

Compatible with vLLM **0.15.x** (v1 block manager architecture).

#### How It Works

When `CONTEXTPILOT_INDEX_URL` is set, the inference engine integrates with ContextPilot at eviction time:

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

### Start the Servers

```bash
# Terminal 1: Start ContextPilot server
python -m contextpilot.server.http_server \
    --port 8765 \
    --infer-api-url http://localhost:30000

# Terminal 2: Start your inference engine with ContextPilot integration enabled

# Option A: SGLang
CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B \
    --port 30000

# Option B: vLLM
CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --port 30000 \
    --enable-prefix-caching
```

### Step 1: Reorder Contexts (Builds Index Automatically)

```python
import requests

response = requests.post("http://localhost:8765/reorder", json={
    "contexts": [
        [1, 5, 10, 15, 20],
        [2, 5, 11, 16, 21],
        [1, 5, 12, 17, 22],
    ]
})

result = response.json()
reordered = result["reordered_contexts"]
order = result["original_indices"]
request_ids = result["request_ids"]
print(f"Reordered {len(request_ids)} contexts, mode: {result['mode']}")
# mode="initial" on first call, "incremental" on subsequent calls
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

When using a patched inference engine with the `CONTEXTPILOT_INDEX_URL` env var, eviction sync is **automatic**. The engine's cache calls the `/evict` endpoint with evicted `request_ids` via the callback.

If you need manual eviction (e.g., for testing), use the HTTP API directly:

```python
# Direct HTTP request
requests.post("http://localhost:8765/evict", json={
    "request_ids": ["contextpilot_abc123", "contextpilot_def456"]
})

# Or using the Python client
from contextpilot.server.http_client import ContextPilotIndexClient

client = ContextPilotIndexClient("http://localhost:8765")
result = client.evict(["contextpilot_abc123", "contextpilot_def456"])
print(f"Removed {result['removed_count']} requests")
print(f"Cleared {result['conversations_cleared']} conversations")
```

---

## Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reorder` | POST | **Primary** — reorder contexts (auto-dispatches stateless / stateful) |
| `/health` | GET | Health check |
| `/deduplicate` | POST | Multi-turn deduplication (lightweight, stateful only) |
| `/evict` | POST | Remove evicted requests (stateful only) |
| `/reset` | POST | Reset index and conversation tracker (stateful only) |
| `/stats` | GET | Get index statistics (stateful only) |
| `/build` | POST | _Deprecated alias → `/reorder`_ |
| `/schedule` | POST | _Deprecated alias → `/reorder` (always stateless)_ |

---

## Complete End-to-End Example

For a complete working example that shows the entire workflow from documents → context scheduling → prompt building → inference, see:

📄 **[examples/stateless_sglang_e2e.py](../../examples/stateless_sglang_e2e.py)**

This example demonstrates:

1. **Document Retrieval** - Simulated RAG retrieval of relevant documents
2. **Context Tokenization** - Converting text to token IDs for scheduling
3. **ContextPilot Scheduling** - Optimal ordering for prefix sharing
4. **Prompt Building** - Constructing RAG prompts with context
5. **Inference** - Batch inference in scheduled order (works with any OpenAI-compatible engine)
6. **Result Reordering** - Mapping results back to original order

### Quick Preview

```python
from contextpilot.server.http_client import ContextPilotIndexClient
import requests

# 1. Tokenize your contexts
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
contexts = [tokenizer.encode(doc, add_special_tokens=False) for doc in documents]

# 2. Reorder with ContextPilot (stateless mode)
client = ContextPilotIndexClient("http://localhost:8765")
reordered_contexts, scheduled_order = client.reorder(contexts=contexts)

# 3. Build prompts using reordered contexts
scheduled_prompts = [build_rag_prompt(query, reordered_contexts[i]) for i in range(len(reordered_contexts))]

# 4. Send to inference engine in scheduled order (maximizes prefix sharing)
response = requests.post("http://localhost:30000/v1/completions", json={
    "prompt": scheduled_prompts,
    "max_tokens": 256
})
scheduled_results = response.json()

# 5. Reorder results back to original order
final_results = [None] * len(scheduled_order)
for new_idx, orig_idx in enumerate(scheduled_order):
    final_results[orig_idx] = scheduled_results[new_idx]
```

### Running the Example

```bash
# Terminal 1: Start your inference engine (SGLang or vLLM)
python -m sglang.launch_server --model Qwen/Qwen3-4B --port 30000
# or: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B --port 30000 --enable-prefix-caching

# Terminal 2: Start ContextPilot
python -m contextpilot.server.http_server --port 8765

# Terminal 3: Run the example
python examples/stateless_sglang_e2e.py
```

---

## Next Steps

- [Multi-Turn](multi_turn.md) - Conversation handling with deduplication
- [API Reference](../reference/api.md) - Full API documentation
