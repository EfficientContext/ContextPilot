# Multi-Turn Conversations

ContextPilot supports efficient multi-turn conversations with **automatic context deduplication**.

## How It Works

1. **Turn 1**: Retrieve documents, store in conversation history
2. **Turn 2+**: Identify overlapping documents, replace with location hints
3. **Result**: 30-60% reduction in redundant document processing

```
Turn 1: "What is ML?"
  └─ Retrieves: [doc_1, doc_2, doc_3, doc_4, doc_5]
  └─ Full context sent to LLM

Turn 2: "How does it differ from DL?"
  └─ Retrieves: [doc_2, doc_3, doc_6, doc_7, doc_8]
  └─ Overlapping: [doc_2, doc_3] → replaced with location hints
  └─ Only [doc_6, doc_7, doc_8] sent as full text
```

---

## Two Approaches

ContextPilot provides two approaches for multi-turn deduplication:

| Approach | Best For | Pros | Cons |
|----------|----------|------|------|
| **Pipeline API** | Simple applications | Easy to use, all-in-one | Less control |
| **HTTP Server API** | Production systems | Fine-grained control, lightweight Turn 2+ | Requires server |

---

## Approach 1: Pipeline API

Best for simple applications where you want everything handled automatically.

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
print(f"Turn 1: {result1['response'][:100]}...")

# Turn 2 - overlapping docs are deduplicated
result2 = pipeline.process_conversation_turn(
    conversation_id=conversation_id,
    query="How does it differ from deep learning?",
    top_k=10,
    enable_deduplication=True,
    generate_response=True
)
print(f"Turn 2: {result2['response'][:100]}...")
print(f"Deduplicated: {result2['deduplication_stats']['num_deduplicated']} docs")
print(f"Savings: {result2['deduplication_stats']['deduplication_rate']:.0%}")

# Reset conversation when done
pipeline.reset_conversation(conversation_id)
```

---

## Approach 2: HTTP Server API (Recommended for Production)

Best for production systems where you need fine-grained control and maximum efficiency.

### Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Multi-Turn Flow                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Turn 1:  POST /reorder                                               │
│           ├─ Build index                                            │
│           ├─ Register in conversation tracker                       │
│           └─ Return request_id (for linking turns)                  │
│                                                                     │
│  Turn 2+: POST /deduplicate  ← Lightweight! No index ops            │
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

# ═══════════════════════════════════════════════════════════════════
# Turn 1: Use /reorder to create index and register conversation
# ═════════════════════════════════════════════════════════════════
turn1_docs = [4, 3, 1]  # Retrieved documents for turn 1

turn1_response = requests.post(
    f"{INDEX_SERVER}/reorder",
    json={
        "contexts": [turn1_docs],
        "deduplicate": True,
        "parent_request_ids": [None],  # No parent for turn 1
    }
).json()

turn1_request_id = turn1_response["request_ids"][0]
print(f"Turn 1 registered: {turn1_request_id}")

# ═══════════════════════════════════════════════════════════════════
# Turn 2: Use /deduplicate (lightweight - no index operations!)
# ═══════════════════════════════════════════════════════════════════
turn2_docs = [4, 3, 2]  # Retrieved documents for turn 2

turn2_response = requests.post(
    f"{INDEX_SERVER}/deduplicate",
    json={
        "contexts": [turn2_docs],
        "parent_request_ids": [turn1_request_id],  # Link to turn 1
    }
).json()

result = turn2_response["results"][0]
print(f"Turn 2 deduplication:")
print(f"  Original docs:     {result['original_docs']}")
print(f"  Overlapping docs:  {result['overlapping_docs']}")  # [4, 3]
print(f"  New docs:          {result['new_docs']}")          # [2]
print(f"  Deduplicated docs: {result['deduplicated_docs']}")  # [2]
print(f"  Reference hints:   {result['reference_hints']}")

# Use result['deduplicated_docs'] for your prompt!
# Include result['reference_hints'] to tell the LLM about previous docs

turn2_request_id = result["request_id"]

# ═══════════════════════════════════════════════════════════════════
# Turn 3+: Continue the chain
# ═══════════════════════════════════════════════════════════════════
turn3_docs = [4, 2, 5]

turn3_response = requests.post(
    f"{INDEX_SERVER}/deduplicate",
    json={
        "contexts": [turn3_docs],
        "parent_request_ids": [turn2_request_id],  # Link to turn 2
    }
).json()

# [4] from turn 1 and [2] from turn 2 are deduplicated
result3 = turn3_response["results"][0]
print(f"Turn 3: {result3['overlapping_docs']} overlap, {result3['new_docs']} new")
```

### Why Use `/deduplicate` for Turn 2+?

| Operation | `/reorder` | `/deduplicate` |
|-----------|----------|----------------|
| Index build | ✓ | ✗ |
| Clustering | ✓ | ✗ |
| Search | ✓ | ✗ |
| Deduplication | ✓ | ✓ |
| **Latency** | ~50-200ms | ~1-5ms |

For multi-turn conversations, Turn 2+ typically doesn't need index operations - just deduplication against conversation history. The `/deduplicate` endpoint is **10-100x faster**.

### Batch Deduplication

Process multiple conversations in a single request:

```python
# Multiple parallel conversations
response = requests.post(
    f"{INDEX_SERVER}/deduplicate",
    json={
        "contexts": [
            [1, 2, 4],    # Conv A Turn 2
            [10, 20, 40], # Conv B Turn 2
            [100, 200],   # Conv C Turn 1 (new)
        ],
        "parent_request_ids": [
            "conv_a_turn1_id",  # Continue Conv A
            "conv_b_turn1_id",  # Continue Conv B
            None,               # New conversation
        ],
    }
).json()

# Summary statistics
print(f"New conversations: {response['summary']['new_conversations']}")
print(f"Continued: {response['summary']['continued_conversations']}")
print(f"Total deduplicated: {response['summary']['total_docs_deduplicated']}")
```

### Custom Hint Templates

Customize reference hints for your prompt format:

```python
response = requests.post(
    f"{INDEX_SERVER}/deduplicate",
    json={
        "contexts": [[4, 3, 2]],
        "parent_request_ids": ["turn1_id"],
        "hint_template": "[Previous context includes Doc {doc_id} from message {turn_number}]"
    }
).json()

# Hints will be:
# "[Previous context includes Doc 4 from message 1]"
# "[Previous context includes Doc 3 from message 1]"
```

---

## Deduplication Statistics

Each turn returns deduplication stats:

```python
result['deduplication_stats'] = {
    'num_deduplicated': 3,       # Documents replaced with hints
    'num_new': 7,                # New documents sent in full
    'deduplication_rate': 0.30,  # 30% savings
    'original_tokens': 5000,     # Tokens without deduplication
    'actual_tokens': 3500        # Tokens with deduplication
}
```

---

## Managing Conversations

### Pipeline API

```python
# Start a new conversation
conversation_id = "session_abc"

# Process multiple turns
for query in user_queries:
    result = pipeline.process_conversation_turn(
        conversation_id=conversation_id,
        query=query,
        enable_deduplication=True,
        generate_response=True
    )

# Reset when done (frees memory)
pipeline.reset_conversation(conversation_id)

# Or reset all conversations
pipeline.reset_all_conversations()
```

### HTTP Server API

```python
# Reset all conversations and index
requests.post(f"{INDEX_SERVER}/reset")
```

### When to Call Reset

| ✅ Call `/reset` when... | ❌ Do NOT reset when... |
|--------------------------|-------------------------|
| Starting a new batch session | Between turns of the same conversation |
| User session is fully complete | Processing concurrent conversations |
| Scheduled maintenance | You need deduplication continuity |
| Deploying new configuration | Memory is not a concern |

**Typical production flow:**
```
Server starts
├─► Session A: /reorder → /deduplicate → /deduplicate → ...
├─► Session B: /reorder → /deduplicate → /deduplicate → ...
├─► Sessions complete naturally
└─► /reset (optional, for maintenance)
```

For production with concurrent users, you typically **don't call reset** - conversations complete naturally and eviction handles memory management.

---

## Disabling Deduplication

For specific turns where you want full context:

### Pipeline API

```python
result = pipeline.process_conversation_turn(
    conversation_id=conversation_id,
    query="Summarize everything we discussed",
    enable_deduplication=False,  # Send all docs in full
    generate_response=True
)
```

### HTTP Server API

Simply don't use the `/deduplicate` endpoint - process documents directly.

---

## Next Steps

- [API Reference](../reference/api.md) - Full API documentation including HTTP endpoints
