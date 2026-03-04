# API Reference

Complete API documentation for ContextPilot.

## ContextPilot

The core class for context optimization.

### Constructor

```python
import contextpilot as cp

engine = cp.ContextPilot(use_gpu=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | `0.001` | Distance computation parameter |
| `use_gpu` | bool | `False` | Use GPU for distance computation |
| `linkage_method` | str | `"average"` | Clustering method |
| `batch_size` | int | `128` | Batch size for distance computation |

### Methods

#### `optimize()`

Reorder contexts and return ready-to-use OpenAI messages. This is the simplest way to use ContextPilot — it handles reordering and prompt assembly in one call.

```python
messages = engine.optimize(
    docs=["Doc about ML", "Doc about AI", "Doc about DL"],
    query="What is ML?",
    conversation_id="user_42",        # optional, for multi-turn
    system_instruction="Be concise.", # optional, prepended to system message
)
# messages is a list of dicts ready for client.chat.completions.create()
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `docs` | List[str] | Required | List of document strings |
| `query` | str | Required | The user question |
| `conversation_id` | str \| None | `None` | Conversation key for multi-turn deduplication |
| `system_instruction` | str \| None | `None` | Extra instruction prepended to the system message |

**Returns:** `List[Dict[str, str]]` — a list of message dicts (`role` / `content`) ready for the OpenAI chat completions API.

#### `optimize_batch()`

Batch-optimize contexts and return messages in scheduled execution order.

```python
messages_batch, order = engine.optimize_batch(
    all_docs=[["Doc A", "Doc B"], ["Doc B", "Doc C"]],
    all_queries=["What is A?", "What is C?"],
)
# messages_batch[i] corresponds to all_queries[order[i]]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `all_docs` | List[List[str]] | Required | Documents for each query |
| `all_queries` | List[str] | Required | One query per entry in `all_docs` |
| `system_instruction` | str \| None | `None` | Extra instruction prepended to every system message |

**Returns:** `(messages_batch, original_indices)` — a 2-tuple where `messages_batch[i]` corresponds to `all_queries[original_indices[i]]`.

#### `reorder()`

Reorder contexts for optimal KV-cache prefix sharing. Use this when you need full control over prompt construction.

```python
reordered, indices = engine.reorder(
    contexts,
    conversation_id="user_42",  # optional, for multi-turn dedup
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contexts` | List[List] | Required | List of contexts (doc IDs or strings) |
| `initial_tokens_per_context` | int | `0` | Initial token budget per context |
| `conversation_id` | str \| None | `None` | Conversation key for deduplication tracking |

**Returns:** `(reordered_contexts, original_indices)` — a 2-tuple where `reordered_contexts[i]` corresponds to `contexts[original_indices[i]]`.

#### `deduplicate()`

Remove already-seen documents from follow-up conversation turns.

> **`conversation_id` is required** — prevents cross-contamination between concurrent users.

```python
results = engine.deduplicate(
    contexts,
    conversation_id="user_42",        # REQUIRED
    hint_template="See Doc {doc_id}", # optional
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contexts` | List[List] | Required | List of contexts to deduplicate |
| `conversation_id` | str | **Required** | Must match ID from a prior `.reorder()` call |
| `hint_template` | str \| None | `None` | Custom hint template with `{doc_id}` placeholder |

**Returns:** `List[Dict]` — one dict per context with keys:

| Key | Type | Description |
|-----|------|-------------|
| `new_docs` | List | Documents not seen in prior turns |
| `overlapping_docs` | List | Documents already sent |
| `reference_hints` | List[str] | Hint strings for overlapping docs |
| `deduplicated_docs` | List | Alias for `new_docs` |

**Raises:**
- `TypeError` if `conversation_id` is not provided
- `ValueError` if `conversation_id` is empty or has no prior `.reorder()` history

---

## HTTP Server Endpoints

ContextPilot provides an HTTP server for live index management with inference engine integration.

### Root Endpoint

```
GET /
```

Root health check endpoint with basic server information.

**Response:**
```json
{
    "status": "ready",
    "mode": "stateful",
    "index_initialized": true,
    "timestamp": "2025-02-15T10:30:00Z"
}
```

### Health Check

```
GET /health
```

Detailed health check with index statistics.

**Response:**
```json
{
    "status": "ready",
    "mode": "stateful",
    "eviction_enabled": true,
    "current_tokens": 12500,
    "utilization": 0.45,
    "index_stats": {
        "total_nodes": 42,
        "leaf_nodes": 20,
        "total_docs": 150
    }
}
```

### Reorder

```
POST /reorder
```

Reorder contexts. Auto-detects mode based on whether an index exists:
- **Stateless** (no index): computes reordering without maintaining state.
- **Stateful** (index present): builds or incrementally updates the index. Call `POST /reset` to force a fresh build.

**Request:**
```json
{
    "contexts": [[1, 2, 3], [2, 3, 4]],
    "initial_tokens_per_context": 0,
    "alpha": 0.001,
    "use_gpu": false,
    "linkage_method": "average",
    "deduplicate": false,
    "parent_request_ids": [null, null],
    "hint_template": "Refer to Doc {doc_id} from Turn {turn_number}"
}
```

Accepts both integer doc IDs and string documents — the server auto-detects input type.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contexts` | List[List] | Required | List of contexts (doc IDs or strings) |
| `initial_tokens_per_context` | int | `0` | Initial token count per context |
| `alpha` | float | `0.001` | Distance computation parameter |
| `use_gpu` | bool | `false` | Use GPU for distance computation |
| `linkage_method` | str | `"average"` | Clustering method |
| `deduplicate` | bool | `false` | Enable multi-turn deduplication |
| `parent_request_ids` | List[str\|null] | `null` | Parent request IDs for deduplication |
| `hint_template` | str | `null` | Custom template for reference hints |

**Response:**
```json
{
    "status": "success",
    "mode": "initial",
    "input_type": "integer",
    "num_contexts": 2,
    "matched_count": 0,
    "inserted_count": 2,
    "request_ids": ["contextpilot_abc123", "contextpilot_def456"],
    "reordered_contexts": [[2, 3, 1], [2, 3, 4]],
    "original_indices": [0, 1],
    "stats": {}
}
```

### Deduplicate

```
POST /deduplicate
```

Deduplicate contexts for multi-turn conversations without index operations.

**Recommended flow:**
1. **Turn 1**: Call `/reorder` (builds index, registers request in tracker)
2. **Turn 2+**: Call `/deduplicate` (deduplicates only, no index ops)

**Request:**
```json
{
    "contexts": [[4, 3, 2], [10, 20, 30]],
    "parent_request_ids": ["req_turn1_a", "req_turn1_b"],
    "hint_template": "See Doc {doc_id} from Turn {turn_number}"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contexts` | List[List[int]] | Required | List of contexts to deduplicate |
| `parent_request_ids` | List[str\|null] | Required | Parent request IDs (null = new conversation) |
| `hint_template` | str | `null` | Custom template for reference hints |

**Response:**
```json
{
    "status": "success",
    "request_ids": ["dedup_abc123", "dedup_def456"],
    "results": [
        {
            "request_id": "dedup_abc123",
            "parent_request_id": "req_turn1_a",
            "original_docs": [4, 3, 2],
            "deduplicated_docs": [2],
            "overlapping_docs": [4, 3],
            "new_docs": [2],
            "reference_hints": ["Refer to Doc 4...", "Refer to Doc 3..."],
            "is_new_conversation": false
        }
    ]
}
```

### Evict

```
POST /evict
```

Notify ContextPilot that KV cache entries have been evicted by the inference engine.

**Request:**
```json
{
    "request_ids": ["req-abc", "req-def"]
}
```

### Reset

```
POST /reset
```

Reset the index and conversation tracker. Clears all state and frees memory.

**Response:**
```json
{
    "status": "success",
    "message": "Index reset to initial state",
    "conversation_tracker": "reset"
}
```

### Stats

```
GET /stats
```

Get detailed index statistics.

**Response:**
```json
{
    "index_stats": {
        "total_nodes": 256,
        "leaf_nodes": 128,
        "total_docs": 1500,
        "unique_docs": 450,
        "tree_depth": 8
    },
    "total_tokens": 50000,
    "num_contexts": 100,
    "cache_utilization": 0.75
}
```

### Get Requests

```
GET /requests
```

Get all tracked request IDs in the index.

**Response:**
```json
{
    "request_ids": ["contextpilot_abc123", "contextpilot_def456"],
    "count": 2
}
```

### Search Context

```
POST /search
```

Search for a context in the index and return its location.

**Request:**
```json
{
    "context": [1, 2, 3, 4],
    "update_access": true
}
```

**Response:**
```json
{
    "status": "success",
    "search_path": [0, 1, 5, 12],
    "node_id": 12,
    "prefix_length": 3
}
```

### Insert Context

```
POST /insert
```

Insert a new context into the index at a specific location.

**Request:**
```json
{
    "context": [1, 2, 3, 4],
    "search_path": [0, 1, 5],
    "total_tokens": 256
}
```

**Response:**
```json
{
    "status": "success",
    "node_id": 42,
    "search_path": [0, 1, 5, 42],
    "request_id": "contextpilot_xyz789"
}
```

---

## ContextPilotIndexClient

Python client for the HTTP server.

```python
from contextpilot.server.http_client import ContextPilotIndexClient

client = ContextPilotIndexClient("http://localhost:8765", timeout=1.0)

# Primary API
reordered, order = client.reorder(contexts)       # returns (reordered_contexts, original_indices)
result = client.reorder_raw(contexts)              # returns full server response dict

# Stateful options
result = client.reorder_raw(
    contexts, deduplicate=True, parent_request_ids=[None]
)
client.deduplicate(contexts, parent_request_ids, hint_template=None)
client.evict(request_ids)
client.reset()

# Queries
client.search(context, update_access=True)
client.insert(context, search_path, total_tokens=0)
client.get_stats()
client.get_requests()
client.health()
client.is_ready()

client.close()
```

### Methods

| Method | Description |
|--------|-------------|
| `reorder(contexts, ...)` | Reorder contexts, returns `(reordered_contexts, original_indices)` |
| `reorder_raw(contexts, ...)` | Reorder contexts, returns full server response dict |
| `deduplicate(contexts, parent_request_ids, hint_template)` | Deduplicate contexts (Turn 2+) |
| `evict(request_ids)` | Remove requests from index |
| `reset()` | Reset index and conversation tracker |
| `get_stats()` | Get index statistics |
| `get_requests()` | Get all tracked request IDs |
| `health()` | Health check |
| `is_ready()` | Check if server is ready |
| `close()` | Close connection |

### Example

```python
from contextpilot.server.http_client import ContextPilotIndexClient

client = ContextPilotIndexClient("http://localhost:8765")

# Turn 1: build index
turn1 = client.reorder_raw(
    contexts=[[4, 3, 1]],
    deduplicate=True,
    parent_request_ids=[None]
)
turn1_id = turn1["request_ids"][0]

# Turn 2+: lightweight deduplication
turn2 = client.deduplicate(
    contexts=[[4, 3, 2]],
    parent_request_ids=[turn1_id]
)
result = turn2["results"][0]
print(f"Overlapping: {result['overlapping_docs']}")  # [4, 3]
print(f"New docs: {result['new_docs']}")             # [2]

client.close()
```

---

## Module-Level Convenience Functions

These use a shared singleton `ContextPilot` instance internally — no need to create one yourself.

```python
import contextpilot as cp

messages = cp.optimize(docs, query, conversation_id="user_42")
messages_batch, order = cp.optimize_batch(all_docs, all_queries)
```

Signatures and parameters are identical to [`ContextPilot.optimize()`](#optimize) and [`ContextPilot.optimize_batch()`](#optimize_batch) above.

---

## InferenceConfig

Configuration for LLM generation used by `RAGPipeline`.

```python
from contextpilot.pipeline import InferenceConfig

config = InferenceConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    backend="sglang",
    base_url="http://localhost:30000",
    max_tokens=256,
    temperature=0.0
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | Required | Model name/path |
| `backend` | str | `"sglang"` | Inference backend (`"sglang"` or `"vllm"`) |
| `base_url` | str | Required | Server URL |
| `max_tokens` | int | `256` | Maximum generation tokens |
| `temperature` | float | `0.0` | Sampling temperature |

---

## RAGPipeline

High-level pipeline combining retrieval and ContextPilot optimization.

### Constructor

```python
from contextpilot.pipeline import RAGPipeline, InferenceConfig

pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_contextpilot=True,
    use_gpu=False,
    inference=InferenceConfig(...)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retriever` | str | Required | Retriever type: `"bm25"`, `"faiss"`, or custom |
| `corpus_path` | str | Required | Path to corpus JSONL file |
| `use_contextpilot` | bool | `True` | Enable ContextPilot optimization |
| `use_gpu` | bool | `False` | Use GPU for distance computation |
| `inference` | InferenceConfig | `None` | Configuration for LLM generation |
| `index_path` | str | `None` | Path to FAISS index (faiss retriever only) |
| `embedding_model` | str | `None` | Embedding model name (faiss retriever only) |
| `embedding_base_url` | str | `None` | Embedding server URL (faiss retriever only) |

### Methods

#### `run()`

```python
results = pipeline.run(queries=["What is ML?"], top_k=20, generate_responses=True)
```

**Returns:**
```python
{
    "retrieval_results": [...],
    "optimized_batch": [...],
    "generation_results": [...],
    "metadata": {"num_queries": 1, "num_groups": 1, "total_time": 1.5}
}
```

#### `retrieve()`

```python
results = pipeline.retrieve(queries=["What is ML?"], top_k=20)
```

#### `optimize()`

```python
optimized = pipeline.optimize(retrieval_results)
```

#### `generate()`

```python
generation_results = pipeline.generate(optimized)
```

#### `save_results()`

```python
pipeline.save_results(results, "output.jsonl")
```

#### `process_conversation_turn()`

```python
result = pipeline.process_conversation_turn(
    conversation_id="session_123",
    query="What is ML?",
    top_k=10,
    enable_deduplication=True,
    generate_response=True
)
```

#### `reset_conversation()`

```python
pipeline.reset_conversation("session_123")
```

#### `reset_all_conversations()`

```python
pipeline.reset_all_conversations()
```
