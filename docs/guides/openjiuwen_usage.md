# Using ContextPilot with openjiuwen WorkflowAgent

This guide shows how to integrate ContextPilot's document reordering into an openjiuwen WorkflowAgent, with llama.cpp as the local inference backend on macOS.

---

## Architecture

```
your app
  │
  ├─ cp.ContextPilot().optimize(docs, query)   ← reorder retrieved docs (Python library, local)
  │
  ▼
openjiuwen WorkflowAgent
  │  LLMExecutable.invoke()  →  stream=False
  ▼
API_BASE  ──► http://localhost:8765/v1  (ContextPilot HTTP proxy)
                  │  forwards non-streaming requests, tracks KV-cache state
                  ▼
              llama-server :8889   ← --cache-reuse, Metal GPU
```

ContextPilot participates at two levels:
1. **Python library** (`cp.ContextPilot().optimize()`) — reorders documents before injection into the prompt so that repeated prefixes are maximally shared across turns.
2. **HTTP proxy** (port 8765) — forwards requests to llama-server and, via the native `DYLD_INSERT_LIBRARIES` hook, keeps its internal index in sync with the KV-cache state when slots are evicted.

---

## Why Both Layers Matter

| Layer | What it does | Result in llama-server logs |
|-------|-------------|----------------------------|
| `optimize()` reordering | Makes context identical (or maximally shared) across calls | `sim_best=1.000`, `batch.n_tokens=1` |
| KV-cache hook (`/evict`) | Tells ContextPilot when a slot was discarded so it can re-rank next turn | Accurate reordering for request N+1 |

Without the reordering: `sim_best` is low, prompt tokens re-evaluated in full each call.
Without the hook: reordering quality degrades over a long multi-turn session.

---

## Prerequisites

```bash
brew install llama.cpp
pip install contextpilot openjiuwen
xcode-select --install   # needed once to compile the C++ hook
```

---

## Start the Servers

**Terminal 1 — llama-server with eviction hook:**

```bash
CONTEXTPILOT_INDEX_URL=http://localhost:8765 \
contextpilot-llama-server \
    -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8889 \
    -ngl 99 --cache-reuse 256 --parallel 4 -c 8192
```

If `contextpilot-llama-server` is not on `$PATH` after `pip install -e .`, use the module directly:

```bash
CONTEXTPILOT_INDEX_URL=http://localhost:8765 \
python -m contextpilot._llamacpp_hook \
    /opt/homebrew/bin/llama-server \
    -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8889 \
    -ngl 99 --cache-reuse 256 --parallel 4 -c 8192
```

**Terminal 2 — ContextPilot HTTP server:**

```bash
python -m contextpilot.server.http_server --port 8765 \
    --infer-api-url http://localhost:8889
```

---

## Discover the Model Name

Before writing code, confirm the model ID that llama-server registered:

```bash
curl http://localhost:8889/v1/models | python3 -m json.tool
```

Use the `id` value exactly — it's the filename without the path, e.g. `Llama-3.2-1B-Instruct-Q4_K_M.gguf`.

---

## With vs Without ContextPilot

The only difference between the two approaches is the context-building step before calling the agent.

**Without ContextPilot** — docs passed in original retrieval order:

```python
context = "\n\n".join(retrieved_docs)
```

**With ContextPilot** — docs reordered for maximum prefix reuse:

```python
cp_instance = cp.ContextPilot(use_gpu=False)

def get_reordered_context(query: str, retrieved_docs: list[str]) -> str:
    messages = cp_instance.optimize(retrieved_docs, query)
    for msg in messages:
        if msg["role"] == "system":
            return msg["content"]
    return "\n\n".join(retrieved_docs)

context = get_reordered_context(query, retrieved_docs)
```

Everything else — workflow definition, agent setup, `Runner.run_agent()` — stays identical.

See `tests/test_jiuwen_mac.py` for a runnable example.

---

## How to Verify KV-Cache Reuse Is Working

Run the same query twice and inspect the llama-server terminal. You should see:

```
# First call — partial reuse
slot 0: cache miss, recalculating 312 tokens (sim_best=0.842, batch.n_tokens=25)

# Second call — full reuse
slot 0: cache hit, skipping 311 tokens (sim_best=1.000, batch.n_tokens=1)
```

`sim_best=1.000` and `batch.n_tokens=1` confirm that the prefix is being fully reused (10× faster prompt evaluation).

---

## Choosing `API_BASE`: proxy vs direct

| `API_BASE` | When to use |
|-----------|-------------|
| `http://localhost:8765/v1` | Default. Use when `LLMComponent.invoke()` is called (non-streaming). Enables eviction tracking so ContextPilot stays in sync with the KV-cache. |
| `http://localhost:8889/v1` | Use if you call `LLMComponent.stream()` directly, or if you suspect the proxy is causing issues. You still get reordering from `optimize()`, but eviction tracking is disabled. |

> **Why this works:** openjiuwen's `LLMExecutable.invoke()` builds its OpenAI API request with `stream=False`
> (`openai_model_client.py:164`). The ContextPilot proxy handles regular JSON responses without issues.
> Only the SSE stream format (`stream=True`) is unsupported by the proxy.

---

## openjiuwen `response_format` Reference

| Value | Behavior | Use when |
|-------|----------|----------|
| `{"type": "text"}` | No prompt injection. Model output returned as-is. | Small models, free-form answers |
| `{"type": "json"}` | Injects a JSON schema instruction into the user prompt. Parses and validates output. | Large models, structured output needed |
| `{"type": "markdown"}` | Injects a markdown formatting instruction. | Rich text output |

> **Note:** openjiuwen never passes `response_format` to the OpenAI API. The `"json"` type is handled
> entirely via prompt injection — it appends a schema instruction to the last user message and then
> `json.loads()` the model output. Do not confuse with the OpenAI API's `"json_object"` mode.

---

## Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `WorkflowTimeoutError` (60 s) | Wrong `MODEL_NAME` — llama-server rejects request silently, openai client retries until timeout | Verify with `curl http://localhost:8889/v1/models` |
| `Json parse error` | Model truncated JSON mid-output (`max_tokens` too low) | Increase `max_tokens` or switch to `response_format={"type": "text"}` |
| `response_format type must be one of...` | Used `"json_object"` (OpenAI API format) instead of `"json"` (openjiuwen format) | Use `{"type": "json"}` |
| Proxy timeout when using 8765 | Wrong model name (see first row) — not a streaming issue | Fix the model name |
