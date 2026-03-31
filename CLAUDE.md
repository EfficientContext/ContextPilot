# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation
```bash
pip install -e ".[dev]"
```

### Testing
```bash
# Run standard tests (excludes slow, GPU, and integration tests)
pytest tests/ -v --tb=short -m "not slow and not gpu and not integration"

# Run a single test file
pytest tests/test_live_index.py -v

# Run a single test
pytest tests/test_live_index.py::test_name -v

# Run GPU tests (requires CUDA)
pytest tests/ -v -m "gpu"

# Run all tests including slow/integration
pytest tests/ -v
```

### Linting / Formatting
```bash
black contextpilot/ --line-length 100
isort contextpilot/
```

### HTTP Server
```bash
# Live mode (tracks KV cache state, requires eviction callbacks from inference engine)
python -m contextpilot.server.http_server --port 8765 --infer-api-url http://localhost:30000

# Stateless mode (one-shot batch reordering, no state kept between calls)
python -m contextpilot.server.http_server --port 8765 --stateless --infer-api-url http://localhost:30000
```

## Architecture

ContextPilot sits between context assembly and LLM inference. For each request (or batch), it reorders context blocks so shared content aligns into a common token prefix, maximizing KV-cache hits on vLLM/SGLang backends.

### Core Pipeline

```
Retrieved docs → ContextIndex (hierarchical clustering) → IntraContextOrderer (reorder within each context)
                                                         → InterContextScheduler (schedule execution order across batch)
                                                         → Optimized prompts → Inference engine
```

**`contextpilot/context_index/`** — Builds the hierarchical cluster tree over a batch of contexts using Jaccard-based distances. CPU (`compute_distance_cpu.py`) and GPU (`compute_distance_gpu.py`) backends. Outputs `IndexResult` with the tree, reordered contexts, and search paths.

**`contextpilot/context_ordering/`** — Two-stage ordering:
- `IntraContextOrderer` — reorders documents *within* each context by traversing the cluster tree top-down so each context starts with its shared-prefix ancestor's docs.
- `InterContextScheduler` — groups contexts by their root-child search path, then sorts by path length descending. This ensures requests sharing a long prefix execute consecutively.

**`contextpilot/server/live_index.py`** — The `ContextPilot` class. Wraps `ContextIndex` for live/incremental use. Exposes `reorder()` (online multi-turn), `optimize_batch()` (offline batch), `build_and_schedule()`, `build_incremental()`, and `search()`/`insert()`/`remove_requests()` for fine-grained control.

**`contextpilot/server/http_server.py`** — FastAPI server with two modes:
- **Stateful/live mode** — maintains a persistent index; `POST /reorder` incrementally updates it; `POST /evict` is called by the inference engine's eviction callback (see `patches/`) to keep the index in sync.
- **Stateless mode** (`--stateless`) — each `POST /reorder` is independent; no cache tracking.
- Also proxies `/v1/completions` and `/v1/*` to the downstream inference engine so clients only need one URL.

**`contextpilot/pipeline/rag_pipeline.py`** — High-level `RAGPipeline` class wiring retriever + optimizer + inference into a single `.run(queries)` call. Supports BM25, FAISS, Mem0, and PageIndex retrievers.

**`contextpilot/retriever/`** — Pluggable retrievers: `BM25Retriever`, `FAISSRetriever`, `Mem0Retriever`, `PageIndexRetriever`.

**`contextpilot/server/conversation_tracker.py`** — Tracks per-conversation document history for cross-turn deduplication (replaces repeated docs with reference hints).

**`contextpilot/api.py`** — Module-level convenience functions `optimize()` and `optimize_batch()` using a lazy singleton `ContextPilot` instance. The main entry point for the two-line integration pattern.

### Engine Patches

`patches/sglang/` and `patches/vllm/` contain modified files for the respective inference engines that fire an HTTP callback to `POST /evict` when KV-cache entries are evicted. These are applied with the `apply_patch.sh` scripts in each directory.

### Key Design Points

- **String vs integer inputs**: All internal logic uses integer doc IDs. String inputs are auto-converted via a `_str_to_id` map at the API boundary and converted back on output.
- **Stateful incremental updates**: After initial `build_and_schedule()`, subsequent calls use `build_incremental()` which matches new contexts against the existing tree and only rebuilds unmatched ones.
- **Test markers**: `slow`, `gpu`, `integration`, `unit`. CI only runs tests without `slow`, `gpu`, or `integration`.
- **Formatting**: `black` with `--line-length 100` (CI uses 100, `pyproject.toml` sets 120 — use 100 to match CI).
