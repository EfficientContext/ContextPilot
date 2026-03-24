# Cache Synchronization

ContextPilot maintains a Context Index that tracks what content is currently cached in the inference backend. This index drives reordering and dedup decisions. Keeping it in sync with the backend's actual cache state is critical.

The sync strategy depends on whether you control the inference engine:

|  | Self-hosted (SGLang, vLLM, llama.cpp) | Cloud provider (OpenAI, Anthropic, etc.) |
|--|---------------------------------------|----------------------------------------|
| You deploy the engine | Yes | No |
| API protocol | OpenAI-compatible | OpenAI-compatible |
| Can patch the engine | Yes → eviction callbacks | No → TTL estimation |
| Sync accuracy | Exact | Approximate |

Both use the same OpenAI-compatible API. The difference is whether ContextPilot can install a hook into the engine's cache eviction path.

## Self-hosted: Eviction Callbacks

When you deploy SGLang, vLLM, or llama.cpp yourself, ContextPilot patches the engine's KV cache at runtime to report evictions:

```
SGLang Radix Cache evicts an entry
        │
        ▼
_sglang_hook.py intercepts RadixCache.evict()
        │  collects evicted request_ids
        │
        ▼
POST /evict {"request_ids": ["req-1", "req-2"]}
        │
        ▼
ContextPilot removes entries from Context Index
        │
        ▼
Next reorder uses updated index (no stale entries)
```

This is exact — ContextPilot knows precisely what is and isn't cached. No guessing.

The hook is installed automatically at import time via `contextpilot_hook.pth`. No engine modification needed.

## Cloud Provider: TTL Estimation

When using a cloud provider's API (OpenAI, Anthropic, MiniMax), you can't patch the engine. These providers cache prompts with a TTL window but provide no eviction callback. ContextPilot models the cache state locally:

```
Request sent to cloud API
        │
        ▼
Response received
        │
        ├─ cache_read_tokens > 0    → cache hit confirmed
        │   ├─ TTL timer refreshed
        │   └─ Context Index node access time updated
        │
        └─ cache_creation_tokens > 0 → new cache entry
            └─ TTL timer started
        
        ...time passes...

        │
        ▼
TTL expires (~5 min Anthropic, ~5-10 min OpenAI)
        │
        ▼
Entry removed from TTL tracker
        │
        ▼
Next request: ContextPilot no longer marks this content for caching
```

The worst case is a missed cache hint (ContextPilot thinks content expired when it's still cached). This means one request won't get the `cache_control` marker — but the cloud may still cache-hit on its own. It never causes incorrect behavior.

### Per-Request TTL

Each request gets its own TTL entry, even if multiple requests share the same prefix. This is important:

```
req-001: prefix [A, B, C]  → TTL entry for req-001
req-002: prefix [A, B, D]  → TTL entry for req-002

req-001 expires → only req-001 removed from TTL
req-002 still active → prefix [A, B] still considered cached
```

If req-001 and req-002 share prefix [A, B], evicting req-001 doesn't affect req-002's TTL. The shared prefix stays in the cache model as long as any request using it is alive.
