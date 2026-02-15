# Quick Start

Get ContextPilot running in 5 minutes.

## Prerequisites

- ContextPilot installed ([Installation Guide](installation.md))
- An OpenAI-compatible inference engine (e.g. [SGLang](https://github.com/sgl-project/sglang))
- For SGLang eviction sync: apply the ContextPilot patch (`bash patches/sglang/apply_patch.sh`)

## Step 1: Start the Inference Engine

This quickstart uses SGLang as an example, but ContextPilot works with any OpenAI-compatible inference engine.

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B \
    --port 30000 \
    --schedule-policy lpm
```

Wait for "The server is fired up and ready to roll!".

> **Tip:** For eviction sync, prefix with `CONTEXTPILOT_INDEX_URL=http://localhost:8765`. This lets the inference engine notify ContextPilot when KV cache entries are evicted.

## Step 2: Start ContextPilot

```bash
python -m contextpilot.server.http_server \
    --port 8765 \
    --infer-api-url http://localhost:30000
```

## Step 3: Build & Infer

```python
import requests

CP = "http://localhost:8765"

# Your retriever returns doc IDs per query and a shared text mapping
documents = {
    0: "Photosynthesis converts sunlight into chemical energy in plants.",
    1: "Chlorophyll absorbs light primarily in blue and red wavelengths.",
    2: "The Calvin cycle fixes CO2 into glucose using ATP and NADPH.",
    3: "Mitochondria generate ATP through cellular respiration.",
    4: "Plant cells contain both chloroplasts and mitochondria.",
    5: "Stomata regulate gas exchange and water loss in leaves.",
}

# Each context is the list of doc IDs retrieved for one query.
# Overlapping IDs (e.g. doc 0, 1) form shared prefixes that the inference engine can cache.
contexts = [
    [0, 1, 2],  # query about photosynthesis
    [0, 1, 5],  # query about leaf structure
    [3, 4, 0],  # query about cell energy
]

# --- Step A: Build the index ---
build = requests.post(f"{CP}/build", json={"contexts": contexts}).json()
print(build["mode"])         # "initial" (first build) or "incremental" (subsequent)
print(build["request_ids"])  # one ID per context

# ContextPilot reorders contexts so shared docs come first → cache reuse
# NOTE: Field name differs between modes!
if build["mode"] == "initial":
    reordered = build["scheduled_reordered"]  # Initial build
else:
    reordered = build["reordered_contexts"]   # Incremental update
# e.g. [[0, 1, 2], [0, 1, 5], [0, 4, 3]]

# --- Step B: Construct prompts using reordered doc order ---
# The original order reflects retrieval relevance (rank 0 = most relevant).
# ContextPilot reorders for cache efficiency, so we add an importance hint
# telling the LLM which docs matter most.
def make_prompt(query, reordered_ids, original_ids):
    docs = "\n".join(f"[Doc {d}] {documents[d]}" for d in reordered_ids)
    ranking = " > ".join(f"[Doc {d}]" for d in original_ids)
    return (
        f"Documents:\n{docs}\n\n"
        f"Importance ranking: {ranking}\n\n"
        f"Prioritize higher-ranked documents. Question: {query}"
    )

# contexts[0] was [0, 1, 2] (original retrieval order)
# reordered[0] might be [0, 1, 2] (same) or reshuffled for prefix sharing
prompt = make_prompt("How does photosynthesis work?",
                     reordered_ids=reordered[0],
                     original_ids=contexts[0])

# --- Step C: Run inference ---
resp = requests.post(f"{CP}/v1/completions", json={
    "prompt": prompt,
    "request_id": build["request_ids"][0],
    "max_tokens": 128,
}).json()

print(resp["choices"][0]["text"])
```

Key idea: ContextPilot reorders the doc IDs so that shared documents (doc 0, 1) appear at the beginning of multiple contexts. The shared prefix is computed once and reused from the inference engine's KV cache.

## Step 4: Incremental Update

Just call `/build` again — since the index already exists, ContextPilot automatically searches it and reorders new contexts to reuse cached prefixes:

```python
build2 = requests.post(f"{CP}/build", json={
    "contexts": [[0, 1, 3]],  # new query reuses docs 0, 1
}).json()

print(build2["mode"])           # "incremental" (index already exists)
print(build2["matched_count"])  # docs already in the index

# Incremental mode uses "reordered_contexts" field
reordered2 = build2["reordered_contexts"]
prompt2 = make_prompt("How do plants produce and consume energy?",
                      reordered_ids=reordered2[0],
                      original_ids=[0, 1, 3])

resp2 = requests.post(f"{CP}/v1/completions", json={
    "prompt": prompt2,
    "request_id": build2["request_ids"][0],
    "max_tokens": 128,
}).json()
```

Docs 0 and 1 are already cached, reusing their KV entries instead of recomputing.

## Step 5: Deduplication (Optional)

In multi-turn, the retriever often returns the same docs again. Add `deduplicate: true` to strip docs already sent in a previous turn:

```python
# Turn 3: retriever returns docs 0, 1, 5 — but 0 and 1 were already in turn 2
build3 = requests.post(f"{CP}/build", json={
    "contexts": [[0, 1, 5]],
    "deduplicate": True,
    "parent_request_ids": [build2["request_ids"][0]],
}).json()

# Check what was deduplicated
dedup = build3["deduplication"]["results"][0]
print(dedup["overlapping_docs"])   # [0, 1] — already sent
print(dedup["new_docs"])           # [5]    — only this is new
print(dedup["deduplicated_docs"])  # [5]    — prompt should only include doc 5
print(dedup["reference_hints"])    # hints like "See Doc 0 from previous turn"
```

Without deduplication, the prompt would repeat docs 0 and 1 — wasting tokens. With it, only doc 5 is included, plus short reference hints for the repeated docs.

## Step 6: String Contexts (Alternative)

Instead of integer doc IDs, you can send **document text directly** as strings. ContextPilot automatically maps strings to internal IDs:

```python
# Option 1: Integer doc IDs (shown in previous examples)
contexts_int = [[0, 1, 2], [0, 1, 5], [3, 4, 0]]

# Option 2: String documents (ContextPilot handles ID mapping internally)
contexts_str = [
    [
        "Photosynthesis converts sunlight into chemical energy.",
        "Chlorophyll absorbs light in blue and red wavelengths.",
        "The Calvin cycle fixes CO2 into glucose."
    ],
    [
        "Photosynthesis converts sunlight into chemical energy.",  # same text = reused
        "Chlorophyll absorbs light in blue and red wavelengths.",
        "Stomata regulate gas exchange in leaves."
    ],
    [
        "Mitochondria generate ATP through respiration.",
        "Plant cells contain both chloroplasts and mitochondria.",
        "Photosynthesis converts sunlight into chemical energy."  # shared doc
    ]
]

# Build with string contexts (works exactly the same!)
build_str = requests.post(f"{CP}/build", json={"contexts": contexts_str}).json()
print(build_str["input_type"])  # "string" (auto-detected)

# Server automatically:
# 1. Maps identical strings to the same internal ID
# 2. Reorders for prefix sharing (just like with integers)
# 3. Returns request_ids for inference tracking

# Use the reordered contexts for prompts (same workflow as integers)
if build_str["mode"] == "initial":
    reordered_str = build_str["scheduled_reordered"]
else:
    reordered_str = build_str["reordered_contexts"]
```

**When to use strings vs integers:**
- **Integers**: When you have a pre-indexed corpus with doc IDs
- **Strings**: When processing dynamic content or documents not in a fixed corpus
- **Both work identically** — choose based on your data source

## Step 7: Stats & Reset

```python
stats = requests.get(f"{CP}/stats").json()
print(stats["index_stats"])

requests.post(f"{CP}/reset", json={})  # clear index
```

## Architecture

```
┌─────────┐                     ┌──────────────┐                ┌─────────────┐
│  Your   │  1. POST /build     │              │                │  Inference  │
│  App    │────────────────────→│ ContextPilot │                │  Engine     │
│         │  ←── request_ids ── │ :8765        │                │  :30000     │
│         │                     │              │                │             │
│         │  2. /v1/completions │              │   /v1/compl.   │             │
│         │────────────────────→│              │──────────────→ │             │
│         │  ←── response ──────│              │ ←── response ──│             │
└─────────┘                     └──────────────┘                └─────────────┘
```

1. **Build** — send contexts, get back reordered `request_ids`
2. **Infer** — send prompt + `request_id`, ContextPilot proxies to the inference engine

## Next Steps

- [Online Usage Guide](../guides/online_usage.md) — Stateless vs live mode, inference engine integration
- [mem0 Integration](../guides/mem0.md) — Use with long-term memory
- [Examples](../../examples/) — Python examples and benchmarks
