# Mac + llama.cpp (Apple Silicon)

ContextPilot runs fully on **Apple Silicon** with **llama.cpp** as the inference backend — no CUDA, no cloud, no external services required.

---

## How It Works

llama.cpp's `--cache-reuse N` flag enables prefix caching: if a new request shares `N` or more tokens with the content already held in a KV-cache slot, those tokens are not re-evaluated. ContextPilot maximises the length of those shared prefixes by clustering retrieved documents hierarchically and reordering them so overlapping content aligns at the front of each prompt.

On Apple Silicon, the entire quantised model (Q4_K_M ≈ 4.5 bits/weight) fits in unified DRAM shared by the CPU and GPU, so Metal offload (`-ngl 99`) gives near-GPU throughput without a discrete GPU.

### Three-Process Architecture

```
llama-server  :8889   (Metal, --cache-reuse 256)
      ↑  raw /completion
eviction proxy  :8890   ← measures cache hits + Apple GPU metrics
      ↑  OpenAI /v1/*
ContextPilot server  :8765   ← reorders contexts for max prefix reuse
      ↑
your application
```

The **eviction proxy** (`contextpilot_edge/proxy_server.py`) is the new Mac-specific layer. It:

- Translates OpenAI-compatible `/v1/chat/completions` requests into llama.cpp's native `/completion` format
- Reads `cache_n` / `prompt_n` from llama.cpp's timing response to measure exact KV-cache reuse per request
- Concurrently samples **Apple GPU metrics** (utilization %, clock frequency, power in watts) via `powermetrics` while inference runs — with no added latency
- Logs every request to `query_log.jsonl` and exposes a `/stats` endpoint with latency percentiles and average cache-reuse ratio

---

## Setup

### Prerequisites

- Apple Silicon Mac (M1 or later)
- llama.cpp built with Metal support
- A GGUF model (e.g. `Qwen3-8B-Q4_K_M.gguf`)

### Install dependencies

```bash
pip install -r requirements-mac.txt && pip install -e . --no-deps
```

### Start the three processes

**Terminal 1 — llama-server with Metal and prefix caching:**

```bash
llama-server -m models/Qwen3-8B-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8889 \
    -ngl 99 --cache-reuse 256 --parallel 4 -c 32768
```

| Flag | Purpose |
|------|---------|
| `-ngl 99` | Offload all layers to Metal GPU |
| `--cache-reuse 256` | Reuse KV cache when prefix overlap ≥ 256 tokens |
| `--parallel 4` | Allocate 4 independent KV-cache slots for concurrent requests |
| `-c 32768` | Context window size |

**Terminal 2 — eviction proxy:**

```bash
CONTEXTPILOT_INDEX_URL=http://localhost:8765 \
    python contextpilot_edge/proxy_server.py
```

The proxy listens on `:8890`, forwards completions to llama-server on `:8889`, and logs cache and GPU metrics for every request.

**Terminal 3 — ContextPilot HTTP server:**

```bash
python -m contextpilot.server.http_server --port 8765 \
    --infer-api-url http://localhost:8890
```

ContextPilot points at the proxy (not directly at llama-server) so that all traffic is metered.

---

## Usage

Your application connects to the ContextPilot server. The two-line integration is identical to other backends:

```python
from openai import OpenAI
import contextpilot as cp

client = OpenAI(base_url="http://localhost:8765/v1", api_key="EMPTY")
cp_instance = cp.ContextPilot(use_gpu=False)   # CPU clustering on Mac

for query in queries:
    contexts = get_contexts(query)              # your RAG retriever
    messages = cp_instance.optimize(contexts, query)
    response = client.chat.completions.create(
        model="qwen3-8b",
        messages=messages,
    )
```

`cp_instance.optimize()` reorders the retrieved documents before each request. Documents shared across queries form a common prefix that llama.cpp can cache and reuse.

### Multi-turn conversations

Pass a stable `conversation_id` across turns so ContextPilot can deduplicate documents already seen in earlier turns:

```python
import uuid
conversation_id = f"conv-{uuid.uuid4().hex[:8]}"

for query in conversation_turns:
    contexts = get_contexts(query)
    messages = cp_instance.optimize(contexts, query, conversation_id=conversation_id)
    response = client.chat.completions.create(model="qwen3-8b", messages=messages, max_tokens=200)
```

### Batch processing

`optimize_batch` schedules an entire batch in the globally optimal execution order — queries sharing documents are sent consecutively to maximise prefix reuse:

```python
all_docs = [get_contexts(q) for q in all_queries]
messages_batch, original_indices = cp_instance.optimize_batch(all_docs, all_queries)

print(f"Scheduled order: {original_indices}")

answers = [""] * len(all_queries)
for messages, orig_idx in zip(messages_batch, original_indices):
    response = client.chat.completions.create(model="qwen3-8b", messages=messages, max_tokens=200)
    answers[orig_idx] = response.choices[0].message.content
```

A complete working example covering all three patterns is at `examples/mac_llama_cpp_example.py`.

---

## Monitoring

### Cache reuse stats

```bash
curl http://localhost:8890/stats
```

```json
{
  "total_queries": 12,
  "latency_ms": { "avg": 843.2, "p50": 790.1, "p95": 1420.5, "min": 312.4, "max": 1887.3 },
  "cache_reuse": {
    "queries_with_data": 12,
    "avg_reuse_ratio_pct": 71.4
  }
}
```

The `reuse_ratio_pct` field shows what fraction of prompt tokens were served from the KV cache rather than re-evaluated. Higher is better.

### Per-request log

Every request is appended to `query_log.jsonl`:

```json
{
  "query_id": "a3f8c1d2",
  "latency_ms": 843.2,
  "cache_reuse": {
    "total_prompt_tokens": 512,
    "reused_tokens": 368,
    "newly_computed": 144,
    "reuse_ratio_pct": 71.9,
    "prompt_eval_ms": 87.4,
    "gen_ms": 755.8,
    "prompt_tps": 1646.1,
    "gen_tps": 38.2
  },
  "gpu": {
    "gpus": [{ "utilization_pct": 84.3, "peak_util_pct": 96.0, "freq_mhz": 1296.0, "power_w": 8.41 }]
  }
}
```

### GPU metrics

GPU samples are collected concurrently with inference via `powermetrics`. Because `powermetrics` requires sudo without a password, run:

```bash
sudo visudo
# Add: your_username ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

If `powermetrics` is unavailable, the proxy still works — GPU fields will show `"note": "no live samples captured"`.

---

## Tuning Tips

| Goal | Adjustment |
|------|-----------|
| Higher cache reuse | Lower `--cache-reuse` (e.g. 64) to match shorter shared prefixes |
| More concurrent requests | Increase `--parallel` (each slot uses ~0.5 MB/token of DRAM) |
| Fit larger model | Use a smaller quantisation (Q3_K_M) or reduce `-c` |
| Reduce power draw | Lower `-ngl` to keep some layers on CPU |

---

## Next Steps

- [Online Usage](online_usage.md) — Stateful/stateless server modes
- [Multi-Turn](multi_turn.md) — Cross-turn deduplication in detail
- [API Reference](../reference/api.md) — Full API documentation
