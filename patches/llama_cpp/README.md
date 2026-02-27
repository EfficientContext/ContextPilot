# llama.cpp Patch for ContextPilot Integration

This directory contains a proxy-based integration for llama.cpp.

## Why a proxy instead of a source patch?

vLLM and SGLang are Python packages, so ContextPilot can patch their in-process
KV cache managers directly.  llama.cpp is a compiled C++ binary, so the
integration is implemented as a thin **proxy server** (`eviction_proxy.py`)
that sits between ContextPilot and llama.cpp.  It intercepts every completion
request, tracks which llama.cpp **slot** each ContextPilot request lands on,
and fires an eviction notification whenever a slot's cached content is replaced.

The protocol is identical to the vLLM and SGLang patches:
```
POST <CONTEXTPILOT_INDEX_URL>/evict
{"request_ids": ["req-abc", "req-def", ...]}
```

## Architecture

```
OpenAI client / ContextPilot pipeline
        |
        ▼
ContextPilot HTTP server  :8765       (python -m contextpilot.server.http_server)
        |  adds `rid` field to /v1/completions requests
        ▼
llama.cpp Eviction Proxy  :8890       (eviction_proxy.py)   ← THIS PATCH
        |  tracks slot_id → request_id, fires POST /evict on slot transitions
        ▼
llama.cpp server          :8889       (llama-server --cache-reuse N --parallel K)
```

The proxy also exposes `/stats` and `/reset` endpoints for monitoring and
resetting slot state when the ContextPilot index is cleared.

## Installation

### Option 1: Automated (Recommended)

```bash
# From the repository root:
CONTEXTPILOT_INDEX_URL=http://localhost:8765 bash patches/llama_cpp/apply_patch.sh
```

### Option 2: Manual

```bash
pip install fastapi uvicorn httpx

# Set environment variables and run:
export LLAMA_SERVER_URL=http://localhost:8889
export CONTEXTPILOT_INDEX_URL=http://localhost:8765
python patches/llama_cpp/eviction_proxy.py
```

## Full Setup

```bash
# 1. Start llama.cpp with prefix caching and parallel slots:
llama-server -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8889 \
    --cache-reuse 256 --parallel 4 \
    -c 32768 -ngl 99

# 2. Start ContextPilot index server (pointing at the proxy, not llama.cpp):
python -m contextpilot.server.http_server \
    --port 8765 \
    --infer-api-url http://localhost:8890

# 3. Start the eviction proxy:
CONTEXTPILOT_INDEX_URL=http://localhost:8765 \
    python patches/llama_cpp/eviction_proxy.py

# 4. Your OpenAI client connects to http://localhost:8765 (ContextPilot server).
```

## What's Happening Under the Hood

### Eviction detection via slot tracking

llama.cpp's `--parallel N` flag allocates N independent KV-cache slots.  Each
slot holds the KV state of the last sequence processed on it.  When a new
request arrives on slot S:

- `cache_n` tokens are **reused** from slot S's current cached sequence
  (tokens that match the new request's prefix)
- `prompt_n` tokens are **re-evaluated** (the non-matching tail is replaced)

After the new request completes, slot S's cache belongs to the new request.
The previous request's unique tokens are gone.

The proxy tracks `slot_id → request_id` in a per-slot state dict.  After every
`/completion` response it compares the new `rid` to the previous slot occupant:

```
prev_rid = slot_state[slot_id]        # e.g. "req-abc"
new_rid  = body["rid"]                # e.g. "req-xyz"  (set by ContextPilot)

if prev_rid != new_rid:
    POST /evict  {"request_ids": ["req-abc"]}
    slot_state[slot_id] = new_rid
```

### Request ID (`rid`) flow

ContextPilot's HTTP server (`http_server.py`) adds `rid` to the body of every
`/v1/completions` request it forwards.  The proxy extracts this field, uses it
for slot tracking, and strips it from the payload before forwarding to
llama.cpp (which ignores unknown fields anyway).

When using the ContextPilot Python library directly (without the HTTP server),
you can pass `rid` manually:

```python
import contextpilot as cp
cp_instance = cp.ContextPilot(use_gpu=False)

for query in queries:
    contexts = get_contexts(query)
    messages = cp_instance.optimize(contexts, query)

    # Pass rid in extra_body so the proxy can track it
    response = client.chat.completions.create(
        model="qwen3-8b",
        messages=messages,
        extra_body={"rid": "req-my-unique-id"},
    )
```

### `/v1/chat/completions` translation

llama.cpp's `/v1/chat/completions` endpoint does not reliably expose `slot_id`
and `cache_n` in its response.  To guarantee these fields, the proxy translates
chat completion requests to the native `/completion` API (which always returns
full timing data).

By default the proxy uses **Llama-3** format.  Set `CHAT_TEMPLATE_FORMAT=chatml`
for Qwen, Mistral, and other ChatML-based models:

```bash
CHAT_TEMPLATE_FORMAT=chatml python patches/llama_cpp/eviction_proxy.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLAMA_SERVER_URL` | `http://localhost:8889` | llama.cpp backend URL |
| `CONTEXTPILOT_INDEX_URL` | *(unset)* | ContextPilot index server (enables eviction sync) |
| `PROXY_HOST` | `0.0.0.0` | Proxy bind host |
| `PROXY_PORT` | `8890` | Proxy bind port |
| `CHAT_TEMPLATE_FORMAT` | `llama3` | `llama3` or `chatml` |
| `LOG_FILE` | `query_log.jsonl` | JSONL log of all requests |

When `CONTEXTPILOT_INDEX_URL` is not set, the proxy still runs as a metrics
logging proxy (zero eviction overhead).

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat completions (with slot tracking) |
| `POST /v1/completions` | OpenAI text completions (with slot tracking, primary ContextPilot path) |
| `POST /completion` | llama.cpp native API (with slot tracking) |
| `GET /stats` | Latency, cache reuse stats, current slot occupancy |
| `POST /reset` | Reset slot state and fire evictions for all tracked requests |
| `GET /v1/*` | Generic proxy to llama.cpp |

## Monitoring

```bash
# View current slot state and cache stats:
curl http://localhost:8890/stats | python -m json.tool

# Reset slot state (e.g. after restarting llama.cpp):
curl -X POST http://localhost:8890/reset
```

## Compatibility

Tested with llama.cpp **b5262+** (any version supporting `--cache-reuse` and
`--parallel`).  The `slot_id` and `cache_n` fields in the `/completion` response
have been present since llama.cpp b2630.

## Differences from vLLM / SGLang patches

| | vLLM / SGLang | llama.cpp |
|---|---|---|
| Patch type | In-process Python module patch | External proxy server |
| Eviction granularity | Per KV-cache block / radix-tree node | Per slot (coarser) |
| Multiple requests share blocks | Yes (prefix sharing across requests) | No (each slot holds one sequence) |
| Zero overhead when disabled | Yes (`CONTEXTPILOT_INDEX_URL` unset) | Yes (eviction logic skipped) |
| GPU metrics | No | Yes (Apple Silicon powermetrics) |
