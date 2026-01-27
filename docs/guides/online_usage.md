# Online Usage

Online mode uses the **ContextPilot Index Server** for live scheduling. Two modes are available:

| Mode | Use Case | Cache Tracking |
|------|----------|----------------|
| **Stateless** | Per-batch scheduling, no state management | ❌ |
| **Stateful** | Live cache sync with the inference engine, eviction tracking | ✅ |

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

# Reorder your data according to the schedule
original_prompts = ["Prompt 0", "Prompt 1", "Prompt 2", "Prompt 3"]
scheduled_order = result['original_indices']
reordered_prompts = [original_prompts[i] for i in scheduled_order]

# Send to SGLang in this order for maximum prefix sharing
for prompt in reordered_prompts:
    # response = sglang_client.generate(prompt)
    pass
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

Stateful mode maintains a **live index** that tracks tokens and synchronizes with SGLang's cache.

**Best for:** Long-running services, cache-aware scheduling, SGLang integration.

### Architecture

```
┌─────────────┐         ┌─────────────────────┐         ┌─────────────┐
│   Client    │ ──────► │  ContextPilot Index     │ ──────► │   SGLang    │
│             │         │  Server (8765)      │         │   (30000)   │
└─────────────┘         └─────────────────────┘         └─────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  LiveContextIndex│
                        │  - Token tracking │
                        │  - LRU eviction   │
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

The server proxies requests to SGLang and tracks tokens automatically:

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

### Step 3: Manual Token Updates (Alternative)

If not using the proxy:

```python
response = requests.post("http://localhost:8765/update_tokens", json={
    "request_id": request_ids[0],
    "num_tokens": 1500
})
```

### Step 4: Eviction Sync (Automatic)

When using the patched SGLang (see [SGLang Integration](#sglang-integration) below), eviction sync is **automatic**. SGLang's radix cache calls the `/evict` endpoint with evicted `request_ids` via a callback.

If you need manual eviction (e.g., for testing), use:

```python
requests.post("http://localhost:8765/evict", json={
    "request_ids": ["request_id_1", "request_id_2"]
})
```

---

## SGLang Integration

ContextPilot integrates with SGLang via the `RAGBOOST_INDEX_URL` environment variable. When set, SGLang automatically syncs evictions with ContextPilot.

### Quick Start

**Option A: Use Patch Script (Recommended)**

Run the patch script to automatically install the patched files:

```bash
# From ContextPilot root directory
bash patches/sglang/apply_patch.sh
```

This script will:
- Find your SGLang installation automatically
- Backup original files
- Copy patched files to the correct location

**Option B: Manual Copy**

```bash
# Copy patched files to your SGLang installation
SGLANG_PATH=$(python -c "import sglang; print(sglang.__path__[0])")

cp patches/sglang/cache_init_params.py $SGLANG_PATH/srt/mem_cache/
cp patches/sglang/common.py $SGLANG_PATH/srt/mem_cache/
cp patches/sglang/radix_cache.py $SGLANG_PATH/srt/mem_cache/

echo "SGLang patched successfully!"
```

### Start SGLang with ContextPilot

```bash
# Start ContextPilot server first
python -m contextpilot.server.http_server \
    --port 8765 \
    --infer-api-url http://localhost:30000

# Start SGLang with ContextPilot integration enabled
RAGBOOST_INDEX_URL=http://localhost:8765 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B \
    --port 30000
```

### How It Works

When `RAGBOOST_INDEX_URL` is set, SGLang integrates with ContextPilot at eviction time:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Eviction Sync Flow                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SGLang Cache Full → RadixCache.evict() → Callback invoked         │
│                                      │                              │
│                                      ▼                              │
│                     POST /evict {"request_ids": ["rid1", "rid2"]}   │
│                                      │                              │
│                                      ▼                              │
│                     ContextPilot removes evicted requests from index    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **Request ID Tracking**: Each request has a unique `request_id` (e.g., `contextpilot_abc123`)
2. **Eviction Callback**: When SGLang evicts cache entries, it notifies ContextPilot
3. **Index Sync**: ContextPilot removes evicted requests from its live index

---

## Patched Files Reference

Three SGLang files are modified for ContextPilot integration:

### 1. `cache_init_params.py`

Adds eviction callback support:

```python
from typing import Callable, Optional

# Type alias for eviction callback
EvictionCallback = Optional[Callable[[set], None]]

@dataclasses.dataclass
class CacheInitParams:
    # ... existing fields ...
    
    # NEW: Callback invoked when requests are evicted
    eviction_callback: EvictionCallback = None

    def __post_init__(self):
        """Auto-create eviction callback from RAGBOOST_INDEX_URL."""
        if self.eviction_callback is None:
            from sglang.srt.mem_cache.common import create_contextpilot_eviction_callback
            self.eviction_callback = create_contextpilot_eviction_callback()
```

### 2. `common.py`

Adds the callback factory function:

```python
import os
import requests

RAGBOOST_INDEX_URL = os.environ.get("RAGBOOST_INDEX_URL")
_contextpilot_enabled = RAGBOOST_INDEX_URL is not None

def create_contextpilot_eviction_callback():
    """Create eviction callback for ContextPilot sync."""
    if not _contextpilot_enabled:
        return None
    
    def eviction_callback(evicted_request_ids: set):
        """Send evicted request IDs to ContextPilot index."""
        if not evicted_request_ids:
            return
        
        # Filter out internal requests
        filtered_ids = {
            rid for rid in evicted_request_ids 
            if not rid.startswith("HEALTH_CHECK")
        }
        
        if not filtered_ids:
            return
        
        try:
            requests.post(
                f"{RAGBOOST_INDEX_URL}/evict",
                json={"request_ids": list(filtered_ids)},
                timeout=1.0
            )
        except Exception as e:
            logger.warning(f"ContextPilot eviction sync failed: {e}")
    
    return eviction_callback
```

### 3. `radix_cache.py`

Adds request tracking and callback invocation:

```python
class TreeNode:
    def __init__(self, ...):
        # ... existing code ...
        self.request_ids: set = set()  # NEW: Track request IDs at this node

class RadixCache:
    def __init__(self, params):
        # ... existing code ...
        self.eviction_callback = params.eviction_callback  # NEW
        self._request_to_node: dict = {}  # NEW: request_id -> node mapping

    def insert(self, key, value, priority=None, request_id=None):
        # ... existing code ...
        # NEW: Track request_id at leaf node
        if request_id:
            leaf_node.request_ids.add(request_id)
            self._request_to_node[request_id] = leaf_node

    def evict(self, num_tokens, ...):
        fully_evicted_requests = set()
        
        while num_tokens > 0:
            # ... existing eviction logic ...
            
            # NEW: Collect request_ids from evicted nodes
            if node.request_ids:
                fully_evicted_requests.update(node.request_ids)
                for rid in node.request_ids:
                    self._request_to_node.pop(rid, None)
        
        # NEW: Invoke callback with evicted request IDs
        if fully_evicted_requests and self.eviction_callback:
            self.eviction_callback(fully_evicted_requests)
```

---

## Manual Integration (Advanced)

If you prefer to apply changes manually instead of using patched files, see the code snippets in the "Patched Files Reference" section above. The key changes are:

1. Add `EvictionCallback` type and `eviction_callback` field to `CacheInitParams`
2. Add `create_contextpilot_eviction_callback()` function in `common.py`
3. Add `request_ids` tracking to `TreeNode` and callback invocation in `RadixCache.evict()`

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

## Next Steps

- [Multi-Turn](multi_turn.md) - Conversation handling with deduplication
- [API Reference](../reference/api.md) - Full API documentation
- [Troubleshooting](../troubleshooting.md) - Common issues
