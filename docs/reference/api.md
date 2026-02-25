# API Reference

Complete API documentation for ContextPilot.

## RAGPipeline

The main interface for ContextPilot.

### Constructor

```python
from contextpilot.pipeline import RAGPipeline, InferenceConfig

pipeline = RAGPipeline(
    retriever="bm25",              # "bm25", "faiss", or custom retriever
    corpus_path="corpus.jsonl",    # Path to corpus file
    use_contextpilot=True,             # Enable/disable ContextPilot optimization
    use_gpu=False,                 # GPU for distance computation
    inference=InferenceConfig(...) # Optional: for LLM generation
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retriever` | str | Required | Retriever type: "bm25", "faiss", or custom |
| `corpus_path` | str | Required | Path to corpus JSONL file |
| `use_contextpilot` | bool | `True` | Enable ContextPilot optimization |
| `use_gpu` | bool | `False` | Use GPU for distance computation |
| `inference` | InferenceConfig | `None` | Configuration for LLM generation |
| `index_path` | str | `None` | Path to FAISS index (for faiss retriever) |
| `embedding_model` | str | `None` | Embedding model name (for faiss) |
| `embedding_base_url` | str | `None` | Embedding server URL (for faiss) |

### Methods

#### `run()`

Run the complete pipeline.

```python
results = pipeline.run(
    queries=["What is ML?", "What is AI?"],
    top_k=20,
    generate_responses=True
)
```

**Returns:**
```python
{
    "retrieval_results": [...],
    "optimized_batch": [...],
    "generation_results": [...],
    "metadata": {
        "num_queries": 2,
        "num_groups": 1,
        "total_time": 1.5
    }
}
```

#### `retrieve()`

Retrieve documents only.

```python
results = pipeline.retrieve(queries=["What is ML?"], top_k=20)
```

#### `optimize()`

Optimize retrieved results.

```python
optimized = pipeline.optimize(retrieval_results)
```

#### `generate()`

Generate responses from optimized results.

```python
generation_results = pipeline.generate(optimized)
```

#### `save_results()`

Save results to file.

```python
pipeline.save_results(results, "output.jsonl")
```

#### `process_conversation_turn()`

Process a multi-turn conversation turn.

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

Reset a conversation's history.

```python
pipeline.reset_conversation("session_123")
```

#### `reset_all_conversations()`

Reset all conversation histories.

```python
pipeline.reset_all_conversations()
```

---

## ContextPilot

The core class for context reordering and multi-turn deduplication.

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
    conversation_id="user_42",       # optional, for multi-turn
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
    conversation_id="user_42",       # REQUIRED
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

## Module-Level Convenience Functions

These use a shared singleton `ContextPilot` instance internally — no need to create one yourself.

```python
import contextpilot as cp

# Single query
messages = cp.optimize(docs, query, conversation_id="user_42")

# Batch
messages_batch, order = cp.optimize_batch(all_docs, all_queries)
```

Signatures and parameters are identical to [`ContextPilot.optimize()`](#optimize) and [`ContextPilot.optimize_batch()`](#optimize_batch) above.

---

## InferenceConfig

Configuration for LLM generation.

```python
from contextpilot.pipeline import InferenceConfig

config = InferenceConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    backend="sglang",              # "sglang" or "vllm"
    base_url="http://localhost:30000",
    max_tokens=256,
    temperature=0.0
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | Required | Model name/path |
| `backend` | str | `"sglang"` | Inference backend |
| `base_url` | str | Required | Server URL |
| `max_tokens` | int | `256` | Maximum generation tokens |
| `temperature` | float | `0.0` | Sampling temperature |

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

### Schedule / Reorder (Stateless)

```
POST /reorder
POST /schedule   # deprecated alias
```

Compute optimal context reordering without maintaining state.

**Request (with integer doc IDs):**
```json
{
    "contexts": [[1, 2, 3], [2, 3, 4]]
}
```

**Request (with string documents):**
```json
{
    "contexts": [
        ["Document text A", "Document text B", "Document text C"],
        ["Document text B", "Document text C", "Document text D"]
    ]
}
```

The server auto-detects input type and handles string-to-ID mapping internally.

**Response:**
```json
{
    "status": "success",
    "mode": "stateless",
    "input_type": "integer",
    "num_contexts": 2,
    "num_groups": 1,
    "reordered_contexts": [[2, 3, 1], [2, 3, 4]],
    "original_indices": [0, 1],
    "groups": [...],
    "stats": {...}
}
```

### Build Index (Stateful)

```
POST /reorder          # primary (auto-detects stateful mode)
POST /build            # deprecated alias
```

Build the index or incrementally update an existing one. Auto-detects mode: empty index → initial build, existing index → incremental update. Call `POST /reset` to force initial build. Supports multi-turn deduplication.

The response always includes `reordered_contexts` and `original_indices`.

**Request (integer doc IDs):**
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

**Request (string documents):**
```json
{
    "contexts": [
        ["Full text of doc A", "Full text of doc B"],
        ["Full text of doc B", "Full text of doc C"]
    ],
    "alpha": 0.001
}
```

Identical strings are automatically mapped to the same internal ID.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contexts` | List[List[int]] | Required | List of contexts (doc ID lists) |
| `initial_tokens_per_context` | int | `0` | Initial token count per context |
| `alpha` | float | `0.001` | Distance computation parameter |
| `use_gpu` | bool | `false` | Use GPU for distance computation |
| `linkage_method` | str | `"average"` | Clustering method |
| `deduplicate` | bool | `false` | Enable multi-turn deduplication |
| `parent_request_ids` | List[str\|null] | `null` | Parent request IDs for deduplication |
| `hint_template` | str | `null` | Custom template for reference hints |

**Response (initial build):**
```json
{
    "status": "success",
    "message": "Index built successfully",
    "mode": "initial",
    "input_type": "integer",
    "num_contexts": 2,
    "matched_count": 0,
    "inserted_count": 2,
    "request_id_mapping": {...},
    "request_ids": ["contextpilot_abc123", "contextpilot_def456"],
    "reordered_contexts": [[2, 3, 1], [2, 3, 4]],
    "original_indices": [0, 1],
    "stats": {...}
}
```

**Response (incremental build — index already exists):**
```json
{
    "status": "success",
    "message": "Incremental build completed",
    "mode": "incremental",
    "input_type": "integer",
    "num_contexts": 1,
    "matched_count": 1,
    "merged_count": 0,
    "request_ids": ["contextpilot_ghi789"],
    "reordered_contexts": [[2, 3, 5]],
    "original_indices": [0],
    "groups": [...],
    "stats": {...}
}
```

**Response (with deduplication):**
```json
{
    "status": "success",
    "request_ids": ["contextpilot_abc123", "contextpilot_def456"],
    "deduplication": {
        "enabled": true,
        "results": [
            {
                "request_id": "contextpilot_abc123",
                "original_docs": [1, 2, 3],
                "deduplicated_docs": [3],
                "overlapping_docs": [1, 2],
                "new_docs": [3],
                "reference_hints": ["Refer to Doc 1...", "Refer to Doc 2..."]
            }
        ],
        "total_docs_deduplicated": 2
    }
}
```

### Deduplicate (Lightweight)

```
POST /deduplicate
```

Deduplicate contexts for multi-turn conversations **without index operations**.

This is a lightweight endpoint designed for **Turn 2+** in multi-turn conversations. It only performs deduplication against conversation history - no index build or search operations.

**Recommended Flow:**
1. **Turn 1**: Call `/reorder` (builds index, registers request in tracker)
2. **Turn 2+**: Call `/deduplicate` (just deduplicates, no index ops)

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
    "message": "Deduplication completed",
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
    ],
    "summary": {
        "total_contexts": 2,
        "new_conversations": 0,
        "continued_conversations": 2,
        "total_docs_deduplicated": 4
    }
}
```

### Evict (Stateful)

```
POST /evict
```

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

**When to call `/reset`:**
- Before starting a completely new batch session
- When a user's multi-turn conversation is fully complete
- During scheduled maintenance to clear accumulated state
- After deploying new code or configuration

**When NOT to call `/reset`:**
- Between turns of the same conversation (you'll lose deduplication benefits)
- When processing concurrent conversations (you'll break other sessions)

**Response:**
```json
{
    "status": "success",
    "message": "Index reset to initial state",
    "conversation_tracker": "reset"
}
```

### Stats (Stateful)

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

### Get Requests (Stateful)

```
GET /requests
```

Get all tracked request IDs in the index.

**Response:**
```json
{
    "request_ids": [
        "contextpilot_abc123",
        "contextpilot_def456",
        "contextpilot_ghi789"
    ],
    "count": 3
}
```

### Search Context (Stateful)

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | List[int] | Required | Context to search for |
| `update_access` | bool | `true` | Update access time for LRU tracking |

**Response:**
```json
{
    "status": "success",
    "search_path": [0, 1, 5, 12],
    "node_id": 12,
    "prefix_length": 3,
    "message": "Context found with prefix length 3"
}
```

### Insert Context (Stateful)

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | List[int] | Required | Context to insert |
| `search_path` | List[int] | Required | Path in tree where to insert |
| `total_tokens` | int | `0` | Initial token count for this context |

**Response:**
```json
{
    "status": "success",
    "node_id": 42,
    "search_path": [0, 1, 5, 42],
    "request_id": "contextpilot_xyz789",
    "message": "Context inserted successfully"
}
```

**Note:** The `request_id` is auto-generated and can be used for token tracking with the inference engine.

---

## ContextPilotIndexClient

Python client for the HTTP server.

```python
from contextpilot.server.http_client import ContextPilotIndexClient

client = ContextPilotIndexClient("http://localhost:8765", timeout=1.0)

# Primary API — works in both stateless and stateful modes
reordered, order = client.reorder(contexts)  # simple tuple return
result = client.reorder_raw(contexts)        # full server response dict

# Stateful-only options
result = client.reorder_raw(
    contexts, deduplicate=True, parent_request_ids=[None]
)
client.deduplicate(contexts, parent_request_ids, hint_template=None)
client.evict(request_ids)  # Evict specific requests
client.reset()  # Clear index and conversation tracker

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
| `reorder(contexts, ...)` | Reorder contexts, returns `(reordered_contexts, original_indices)` tuple |
| `reorder_raw(contexts, ...)` | Reorder contexts, returns full server response dict |
| `build(...)` | **Deprecated** — alias for `reorder_raw()` |
| `schedule(...)` | **Deprecated** — alias for `reorder_raw()` |
| `deduplicate(contexts, parent_request_ids, hint_template)` | Deduplicate contexts (Turn 2+, lightweight) |
| `evict(request_ids)` | Remove requests from index |
| `reset()` | Reset index and conversation tracker |
| `get_stats()` | Get index statistics |
| `get_requests()` | Get all tracked request IDs |
| `health()` | Health check |
| `is_ready()` | Check if server is ready |
| `close()` | Close connection |

### Deduplication Example

```python
from contextpilot.server.http_client import ContextPilotIndexClient

client = ContextPilotIndexClient("http://localhost:8765")

# Turn 1: Build index via /reorder
turn1 = client.reorder_raw(
    contexts=[[4, 3, 1]],
    deduplicate=True,
    parent_request_ids=[None]
)
turn1_id = turn1["request_ids"][0]

# Turn 2+: Lightweight deduplication
turn2 = client.deduplicate(
    contexts=[[4, 3, 2]],
    parent_request_ids=[turn1_id]
)

result = turn2["results"][0]
print(f"Overlapping: {result['overlapping_docs']}")  # [4, 3]
print(f"New docs: {result['new_docs']}")             # [2]
print(f"Hints: {result['reference_hints']}")

client.close()
```
