# How It Works

ContextPilot optimizes LLM inference through two mechanisms: **Reorder** and **Deduplication**. Both operate on the request before it reaches the inference engine.

## Reorder

LLM engines (SGLang, vLLM) use a prefix cache (Radix Cache) — if two requests share the same token prefix, the second request reuses the cached KV computation. But when context blocks are assembled in different orders across requests, the prefix changes and the cache misses.

Reorder solves this by sorting context blocks into a canonical order that maximizes prefix sharing:

```
Without reorder:                    With reorder:
  Request 1: [A, B, C] → cached      Request 1: [A, B, C] → cached
  Request 2: [B, C, D] → cache miss  Request 2: [B, C, D] → [B, C] prefix hit!
                                        (B, C sorted to front since they're cached)
```

ContextPilot builds a Context Index (hierarchical clustering tree) that groups similar documents. When a new request arrives, it reorders the documents so that:
1. Documents already in the cache come first (maximizing prefix reuse)
2. Similar documents are adjacent (maximizing future prefix sharing)

See [cache_sync.md](cache_sync.md) for how the Context Index stays in sync with the engine's cache.

## Deduplication

When an agent reads multiple documents that share content, the conversation history accumulates redundant text. ContextPilot removes this redundancy through two layers:

### ContextBlock-level deduplication

If a tool result is byte-identical to an earlier one in the same conversation, replace it with a reference. This is handled by the intercept pipeline's `single_doc_hashes` for cross-turn deduplication, and the conversation tracker's `deduplicate()` for the `/reorder` API.

### Content-level deduplication

Like file system deduplication — when two documents share content blocks (e.g., contracts from the same template), only the first occurrence is kept. Subsequent identical blocks are replaced with pointers.

**How it works:**

1. Split each tool result into blocks using content-defined boundaries (line hash mod M)
2. Hash each block (SHA-256)
3. If a block matches one from a different tool result, replace it with a pointer
4. Never deduplicate within the same tool result

```
Contract A (kept intact):           Contract B (after deduplication):
┌────────────────────────┐          ┌────────────────────────┐
│ Art. 1 — Definitions   │          │ Art. 1 — (unique part) │
│ Art. 2 — Scope         │          │ [... "Art. 2 — Scope"  │
│ Art. 3 — Term          │          │   — see earlier result] │
│ ...                    │          │ ...                     │
│ Art. 16 — General      │          │ [... "Art. 16"          │
│ Art. 17 — Cloud Terms  │          │   — see earlier result] │
└────────────────────────┘          │ Art. 17 — AI Terms     │
  45 KB                             └────────────────────────┘
                                      15 KB
```

Each pointer quotes the first line of the replaced block so the LLM knows what content it refers to. The LLM resolves pointers via attention to the original content above.

**Why content-defined chunking?** Fixed-size blocks have an alignment problem — if content shifts by a few lines, all block boundaries change and hashes stop matching. Content-defined boundaries (determined by `hash(line) % M`) adapt to the content, so the same text produces the same blocks regardless of its position in the document. This is the same principle used in file system deduplication (Rabin fingerprint).

### Tuning block size

The `--chunk-modulus M` flag controls the average block size (default: 13 lines per block).

```bash
python -m contextpilot.server.http_server --chunk-modulus 13   # default
```

| M | Avg block size | Best for |
|---|---------------|----------|
| 7-10 | ~7-10 lines | Documents with scattered differences (e.g., config files, code with inline changes) |
| 11-15 | ~11-15 lines | Template documents with concentrated differences (contracts, proposals) — **default** |
| 20-30 | ~20-30 lines | Documents that are nearly identical (only a few lines differ) |

Smaller M = more blocks = more fine-grained deduplication, but each pointer has ~80 chars of overhead. Larger M = fewer blocks = less overhead, but may miss partial overlaps if differences are scattered.

### API

```python
from contextpilot.dedup import dedup_chat_completions, DedupResult

result = dedup_chat_completions(body, chunk_modulus=13)
# result.blocks_deduped — number of blocks replaced with pointers
# result.blocks_total — total blocks processed  
# result.chars_saved — characters removed
```

The deduplication module (`contextpilot/dedup/`) is independent of the server — it operates on message content with no dependency on the Context Index or cache state.
