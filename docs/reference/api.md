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

ContextPilot provides an HTTP server for live index management with SGLang integration.

### Health Check

```
GET /health
```

**Response:**
```json
{"status": "ok", "mode": "stateless"}
```

### Schedule (Stateless)

```
POST /schedule
```

Compute optimal scheduling without maintaining state.

**Request:**
```json
{
    "contexts": [[1, 2, 3], [2, 3, 4]],
    "alpha": 0.005,
    "use_gpu": false,
    "linkage_method": "average"
}
```

**Response:**
```json
{
    "original_indices": [0, 1],
    "num_groups": 1,
    "scheduling_time": 0.05
}
```

### Build Index (Stateful)

```
POST /build
```

Build a new index or incrementally update an existing one. Supports multi-turn deduplication.

**Request:**
```json
{
    "contexts": [[1, 2, 3], [2, 3, 4]],
    "initial_tokens_per_context": 0,
    "alpha": 0.005,
    "use_gpu": false,
    "linkage_method": "average",
    "incremental": false,
    "deduplicate": false,
    "parent_request_ids": [null, null],
    "hint_template": "Refer to Doc {doc_id} from Turn {turn_number}"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contexts` | List[List[int]] | Required | List of contexts (doc ID lists) |
| `initial_tokens_per_context` | int | `0` | Initial token count per context |
| `alpha` | float | `0.005` | Distance computation parameter |
| `use_gpu` | bool | `false` | Use GPU for distance computation |
| `linkage_method` | str | `"average"` | Clustering method |
| `incremental` | bool | `false` | Use incremental build (search/reorder/merge) |
| `deduplicate` | bool | `false` | Enable multi-turn deduplication |
| `parent_request_ids` | List[str\|null] | `null` | Parent request IDs for deduplication |
| `hint_template` | str | `null` | Custom template for reference hints |

**Response (without deduplication):**
```json
{
    "status": "success",
    "message": "Index built successfully",
    "mode": "initial",
    "num_contexts": 2,
    "request_ids": ["contextpilot_abc123", "contextpilot_def456"],
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
1. **Turn 1**: Call `/build` (builds index, registers request in tracker)
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

### Update Tokens (Stateful)

```
POST /update_tokens
```

**Request:**
```json
{
    "request_id": "req-abc",
    "num_tokens": 1500
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

**Response:**
```json
{
    "total_tokens": 50000,
    "num_contexts": 100,
    "cache_utilization": 0.75
}
```

---

## ContextPilotIndexClient

Python client for the HTTP server.

```python
from contextpilot.server.http_client import ContextPilotIndexClient

client = ContextPilotIndexClient("http://localhost:8765", timeout=1.0)

# Stateless
result = client.schedule(contexts, alpha=0.005, use_gpu=False)

# Stateful
client.build(contexts, alpha=0.005, use_gpu=False, deduplicate=True)
client.deduplicate(contexts, parent_request_ids, hint_template=None)
client.evict(request_ids)
client.update_tokens(request_id, num_tokens)
client.health()

client.close()
```

### Methods

| Method | Description |
|--------|-------------|
| `schedule(contexts, alpha, use_gpu)` | Get optimal scheduling (stateless) |
| `build(contexts, alpha, use_gpu, deduplicate, parent_request_ids)` | Build live index |
| `deduplicate(contexts, parent_request_ids, hint_template)` | Deduplicate contexts (Turn 2+) |
| `evict(request_ids)` | Remove requests from index |
| `update_tokens(request_id, num_tokens)` | Update token count |
| `touch(request_id)` | Update LRU access time |
| `reset()` | Reset index and conversation tracker |
| `health()` | Health check |
| `is_ready()` | Check if server is ready |
| `close()` | Close connection |

### Deduplication Example

```python
from contextpilot.server.http_client import ContextPilotIndexClient

client = ContextPilotIndexClient("http://localhost:8765")

# Turn 1: Build index
turn1 = client.build(
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
