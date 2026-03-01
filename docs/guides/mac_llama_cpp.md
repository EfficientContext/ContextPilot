# Mac + llama.cpp (Apple Silicon)

ContextPilot runs fully on **Apple Silicon** with **llama.cpp** as the inference backend — no CUDA, no cloud, no external services required.

---

## How It Works

llama.cpp's `--cache-reuse N` flag enables prefix caching: if a new request shares `N` or more tokens with the content already held in a KV-cache slot, those tokens are not re-evaluated. ContextPilot maximises the length of those shared prefixes by clustering retrieved documents hierarchically and reordering them so overlapping content aligns at the front of each prompt.

On Apple Silicon, the entire quantised model (Q4_K_M ≈ 4.5 bits/weight) fits in unified DRAM shared by the CPU and GPU, so Metal offload (`-ngl 99`) gives near-GPU throughput without a discrete GPU.

### Architecture

llama-server ships its own OpenAI-compatible `/v1/*` API (including `/v1/chat/completions` with built-in chat-template handling), so no separate API server is needed.

```
llama-server  :8889   (Metal, --cache-reuse 256, native /v1/* API)
      ↑  OpenAI /v1/chat/completions
ContextPilot server  :8765   ← reorders contexts for max prefix reuse
      ↑
your application
```

---

## Setup

### Prerequisites

- Apple Silicon Mac (M1 or later)
- llama-server: `brew install llama.cpp`
- A GGUF model (e.g. `Qwen3-8B-Q4_K_M.gguf`)

### Install dependencies

```bash
pip install -e .
```

### Start

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

**Terminal 2 — ContextPilot HTTP server:**

```bash
python -m contextpilot.server.http_server --port 8765 \
    --infer-api-url http://localhost:8889
```

ContextPilot points directly at llama-server's built-in OpenAI API.

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

## Benchmarking

`tests/test_mac_contextpilot.sh` runs the full MultihopRAG benchmark end-to-end, comparing ContextPilot (reordered docs) against a baseline (original doc order).

**Additional prerequisites**

- Docker (for Elasticsearch / BM25 retrieval)

**Run**

```bash
# Full automated run — 100 queries
bash tests/test_mac_contextpilot.sh

# Fewer queries for a quick test
bash tests/test_mac_contextpilot.sh --num-queries 100

# Custom-built llama-server (not in PATH)
bash tests/test_mac_contextpilot.sh --llama-server /path/to/llama.cpp/build/bin/llama-server

# Different model
bash tests/test_mac_contextpilot.sh --model models/Llama-3.2-3B-Q4_K_M.gguf

# Skip dataset download if already done
bash tests/test_mac_contextpilot.sh --skip-data-prep
```

The script handles everything automatically:

1. Starts Elasticsearch and downloads the MultiHopRAG dataset
2. Builds BM25 retrieval data and reorders with ContextPilot (offline batch)
3. Starts llama-server and the ContextPilot HTTP server
4. Runs the ContextPilot benchmark, then restarts with a clean KV cache and runs the baseline
5. Prints a side-by-side comparison of F1 scores and latency

Results are saved to `results_contextpilot.jsonl` and `results_baseline.jsonl`.

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
