# Multi-Turn Conversations

ContextPilot supports efficient multi-turn conversations with **automatic context deduplication**.

## How It Works

1. **Turn 1**: Retrieve documents, reorder, and register in conversation history
2. **Turn 2+**: Identify overlapping documents, replace with location hints
3. **Result**: 30-60% reduction in redundant document processing

```
Turn 1: "What is ML?"
  └─ Retrieves: [doc_1, doc_2, doc_3, doc_4, doc_5]
  └─ engine.reorder(docs, conversation_id="user_42")
  └─ Full context sent to LLM

Turn 2: "How does it differ from DL?"
  └─ Retrieves: [doc_2, doc_3, doc_6, doc_7, doc_8]
  └─ engine.deduplicate(docs, conversation_id="user_42")
  └─ Overlapping: [doc_2, doc_3] → replaced with location hints
  └─ Only [doc_6, doc_7, doc_8] sent as full text
```

---

## Safety: `conversation_id` is Required

`.deduplicate()` **requires** an explicit `conversation_id`. This prevents accidental cross-contamination between concurrent users sharing the same `ContextPilot` instance.

```python
# ✅ Correct — each user has their own conversation_id
engine.deduplicate(docs, conversation_id="user_42")

# ❌ Error — missing conversation_id
engine.deduplicate(docs)  # TypeError: missing required argument

# ❌ Error — no prior .reorder() for this conversation
engine.deduplicate(docs, conversation_id="unknown")  # ValueError
```

---

## Python API (Recommended)

### Basic Multi-Turn Flow

```python
import contextpilot as cp

engine = cp.ContextPilot(use_gpu=False)

# ═══════════════════════════════════════════════════════════
# Turn 1: Reorder and register docs under a conversation
# ═══════════════════════════════════════════════════════════
turn1_docs = [[4, 3, 1]]
reordered, indices = engine.reorder(turn1_docs, conversation_id="user_42")

# Send reordered docs to LLM...

# ═══════════════════════════════════════════════════════════
# Turn 2: Deduplicate — lightweight, no index operations
# ═══════════════════════════════════════════════════════════
turn2_docs = [[4, 3, 2]]
results = engine.deduplicate(turn2_docs, conversation_id="user_42")

r = results[0]
print(f"Overlapping: {r['overlapping_docs']}")  # [4, 3] — already sent
print(f"New docs:    {r['new_docs']}")          # [2]    — send this
print(f"Hints:       {r['reference_hints']}")   # hints for the LLM

# Build prompt with only new docs + reference hints for overlapping ones

# ═══════════════════════════════════════════════════════════
# Turn 3+: Continue the chain (history accumulates)
# ═══════════════════════════════════════════════════════════
turn3_docs = [[4, 2, 5]]
results3 = engine.deduplicate(turn3_docs, conversation_id="user_42")
r3 = results3[0]
print(f"Overlap: {r3['overlapping_docs']}")  # [4, 2] — from turns 1 & 2
print(f"New:     {r3['new_docs']}")          # [5]
```

### Multiple Concurrent Users

Each user's history is isolated by `conversation_id`:

```python
engine = cp.ContextPilot(use_gpu=False)

# User A Turn 1
engine.reorder([[1, 2, 3]], conversation_id="user_a")

# User B Turn 1 (completely independent)
engine.reorder([[10, 20, 30]], conversation_id="user_b")

# User A Turn 2 — only sees user A's history
results_a = engine.deduplicate([[1, 5]], conversation_id="user_a")
assert results_a[0]["overlapping_docs"] == [1]   # from user A's turn 1
assert results_a[0]["new_docs"] == [5]

# User B Turn 2 — only sees user B's history
results_b = engine.deduplicate([[10, 40]], conversation_id="user_b")
assert results_b[0]["overlapping_docs"] == [10]  # from user B's turn 1
assert results_b[0]["new_docs"] == [40]
```

### Batch Deduplication

Deduplicate multiple contexts in a single call:

```python
engine.reorder([[1, 2, 3, 4]], conversation_id="user_42")

# Multiple contexts in one deduplicate call
results = engine.deduplicate(
    [[1, 5], [3, 6]],
    conversation_id="user_42",
)

# results[0]: overlapping=[1], new=[5]
# results[1]: overlapping=[3], new=[6]
```

### Custom Hint Templates

Customize reference hints to match your prompt format:

```python
results = engine.deduplicate(
    [[1, 5]],
    conversation_id="user_42",
    hint_template="[See Doc {doc_id} from previous context]",
)
# hints: ["[See Doc 1 from previous context]"]
```

### String Contexts

Works with string documents too:

```python
engine.reorder(
    [["Photosynthesis converts sunlight...", "Chlorophyll absorbs light..."]],
    conversation_id="session_1",
)

results = engine.deduplicate(
    [["Photosynthesis converts sunlight...", "Mitosis is cell division..."]],
    conversation_id="session_1",
)
r = results[0]
print(r["overlapping_docs"])  # ["Photosynthesis converts sunlight..."]
print(r["new_docs"])          # ["Mitosis is cell division..."]
```

---

## Pipeline API

For applications using `RAGPipeline`, deduplication is built in:

```python
from contextpilot.pipeline import RAGPipeline, InferenceConfig

pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_contextpilot=True,
    inference=InferenceConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:30000"
    )
)

conversation_id = "user_session_123"

# Turn 1
result1 = pipeline.process_conversation_turn(
    conversation_id=conversation_id,
    query="What is machine learning?",
    top_k=10,
    enable_deduplication=True,
    generate_response=True
)

# Turn 2 — overlapping docs are deduplicated
result2 = pipeline.process_conversation_turn(
    conversation_id=conversation_id,
    query="How does it differ from deep learning?",
    top_k=10,
    enable_deduplication=True,
    generate_response=True
)

# Reset conversation when done
pipeline.reset_conversation(conversation_id)
```

---

## HTTP Server API

For production systems using the HTTP server, the `/deduplicate` endpoint provides the same functionality with request-ID-based conversation tracking.

### Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Multi-Turn Flow                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Turn 1:  POST /reorder                                             │
│           ├─ Build index                                            │
│           ├─ Register in conversation tracker                       │
│           └─ Return request_id (for linking turns)                  │
│                                                                     │
│  Turn 2+: POST /deduplicate  ← Lightweight! No index ops           │
│           ├─ Look up conversation history by parent_request_id      │
│           ├─ Find overlapping documents                             │
│           ├─ Generate reference hints                               │
│           └─ Return deduplicated context                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Example

```python
import requests

INDEX_SERVER = "http://localhost:8765"

# Turn 1: POST /reorder
turn1_response = requests.post(
    f"{INDEX_SERVER}/reorder",
    json={"contexts": [[4, 3, 1]]},
).json()

turn1_request_id = turn1_response["request_ids"][0]

# Turn 2: POST /deduplicate (lightweight)
turn2_response = requests.post(
    f"{INDEX_SERVER}/deduplicate",
    json={
        "contexts": [[4, 3, 2]],
        "parent_request_ids": [turn1_request_id],
    }
).json()

result = turn2_response["results"][0]
print(f"Overlapping: {result['overlapping_docs']}")  # [4, 3]
print(f"New docs:    {result['new_docs']}")           # [2]
```

### Why Use `/deduplicate` for Turn 2+?

| Operation | `/reorder` | `/deduplicate` |
|-----------|----------|----------------|
| Index build | ✓ | ✗ |
| Clustering | ✓ | ✗ |
| Search | ✓ | ✗ |
| Deduplication | ✓ | ✓ |
| **Latency** | ~50-200ms | ~1-5ms |

For multi-turn conversations, Turn 2+ typically doesn't need index operations — just deduplication against conversation history. The `/deduplicate` endpoint is **10-100x faster**.

---

## Managing Conversations

### Python API

Conversation state is tracked internally per `conversation_id`. No explicit reset needed — just use different IDs for different sessions.

### Pipeline API

```python
# Reset a specific conversation (frees memory)
pipeline.reset_conversation(conversation_id)

# Or reset all conversations
pipeline.reset_all_conversations()
```

### HTTP Server API

```python
# Reset all conversations and index
requests.post(f"{INDEX_SERVER}/reset")
```

---

## Next Steps

- [API Reference](../reference/api.md) - Full API documentation including HTTP endpoints
