# Changelog

All notable changes to ContextPilot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.4] - 2026-02-21

### Added
- **vLLM APC (Automatic Prefix Caching) eviction sync patch** for vLLM — synchronizes KV cache evictions back to ContextPilot, analogous to the existing SGLang radix cache patch
  - Patched `block_pool.py` with bidirectional block-to-request tracking and HTTP eviction callback
  - `create_contextpilot_eviction_callback()` factory reads `CONTEXTPILOT_INDEX_URL` and POSTs to `/evict`
  - Request ID normalization strips vLLM-specific prefixes (`cmpl-`, `chatcmpl-`, `batch-`) and suffixes to match ContextPilot canonical `req-*` IDs
  - Health-check and internal IDs automatically filtered from tracking
  - Zero overhead when `CONTEXTPILOT_INDEX_URL` is not set
- Automated patch installer script (`patches/vllm/apply_patch.sh`) with timestamped backup and auto-detection of vLLM install path
- `GET /requests` endpoint — returns all tracked request IDs for observability and e2e verification
- `ContextPilot.get_all_request_ids()` and `ContextPilot.reset()` methods on the live index
- End-to-end vLLM patch verifier (`examples/vllm_patch_e2e_check.py`) with fast PR-validation and stress profiles
- Comprehensive vLLM patch test suite (14 unit tests, fully mocked — no vLLM dependency required)
- vLLM patch documentation (`patches/vllm/README.md`) covering automated, manual, and symlink installation methods

### Changed
- `POST /evict` endpoint is now engine-agnostic — documents both SGLang and vLLM as supported engines using the same `{"request_ids": [...]}` protocol
- `POST /reset` now also clears the global string-to-ID mapping for clean restarts
- Server-side request ID normalization added to strip vLLM-specific prefixes and suffixes
- Examples README updated with vLLM patch e2e check instructions and dual-engine (SGLang + vLLM) documentation

## [0.3.3.post2] - 2026-02-17

### Added
- `.deduplicate()` method on `ContextPilot` — multi-turn deduplication with **required `conversation_id`** for data isolation between concurrent users
- `.reorder()` now accepts a single `List` (auto-wrapped to `List[List]`) for convenience
- Cross-contamination warning when `.reorder()` is called without `conversation_id` after explicit IDs have been used
- Safety checks: `deduplicate()` raises `ValueError` if `conversation_id` is empty or has no prior `.reorder()` history

### Changed
- Default `alpha` changed from `0.005` → `0.001` across all source, docs, examples, and tests
- Updated quickstart, multi-turn guide, and API reference docs to use Python `ContextPilot` API for deduplication examples
- README examples simplified: removed tuple unpacking in favor of explicit `reordered, indices =` assignment; single-list `.reorder(mems)` usage

## [0.3.3] - 2026-02-17

### Added
- `cp.ContextPilot` — renamed from `LiveContextIndex`, now the single user-facing class for both stateful and stateless reordering
- `.reorder()` method on `ContextPilot` — unified one-call API returning `(reordered_contexts, original_indices)` tuple
- Unified `POST /reorder` HTTP endpoint that auto-dispatches between stateless and stateful modes
- `client.reorder()` and `client.reorder_raw()` methods in `ContextPilotIndexClient`

### Changed
- **Renamed `LiveContextIndex` → `ContextPilot`** across all source, tests, server, docs, and examples
- **Unified response keys**: All endpoints now return consistent `reordered_contexts` and `original_indices` — removed `scheduled_reordered`, `final_mapping`, `scheduled_order` aliases
- Removed `build_context_index` and `InterContextScheduler` from public API (`__all__`) — they remain available via submodule imports for advanced use
- Updated all examples to use `cp.ContextPilot` class and `POST /reorder` endpoint
- Batch example (`stateless_batch_example.py`) now uses `asyncio.gather` for concurrent generation
- `pageindex_e2e_example.py` and `offline/prepare_batch.py` rewritten to use public `cp.ContextPilot` API
- Fixed stale `RAGBOOST_INDEX_URL` env var → `CONTEXTPILOT_INDEX_URL` in mem0 guide
- Fixed dead doc links in `examples/offline/README.md`
- Fixed SGLang version compatibility: `0.4.x` → `0.5.x` in patches README

### Deprecated
- `POST /build` and `POST /schedule` endpoints — use `POST /reorder` instead
- `client.build()` and `client.schedule()` — use `client.reorder()` / `client.reorder_raw()` instead

## [0.3.2] - 2026-02-16

### Added
- `LiveContextIndex` exported from top-level package for convenient imports
- Automatic string-to-integer context conversion — `List[List[str]]` inputs are now accepted natively
- Leaf-node splitting in incremental build for finer-grained prefix sharing
- Sibling insertion mode when new contexts overlap but do not share a prefix with the matched node
- Mem0 LoCoMo benchmark example (`mem0_locomo_example.py`)

### Changed
- Eviction API now accepts `request_ids: List[str]` instead of `num_tokens: int` for precise request-level eviction sync
- `_handle_single_prompt` always creates an empty root above the leaf, preventing root-exclusion guard from skipping valid matches during incremental builds
- Improved diagnostic logging in `build_incremental` (prefix/sibling mode, overlap ratios)

### Fixed
- `build_incremental` incorrectly matching against the global root node; root ID is now excluded
- `reordered_contexts` output now correctly preserves the ordering used for cache prefix sharing (previously lost by `ClusterNode.__init__` sorting)
- `doc_ids` ordering restored after `_clone_subtree` to maintain reordered context order
- Batch size = 1 edge case in metric calculation
- Various live-index insertion and search-path bugs

## [0.3.1] - 2026-02-15

### Added
- Lexicographic tiebreaker in `InterContextScheduler` for deterministic ordering among equal-length paths
- SGLang patch installation instructions in online usage guide
- Dedicated PageIndex integration guide (`docs/guides/pageindex.md`)
- PyPI badge in README
- `PageIndexRetriever`-based end-to-end example (`pageindex_e2e_example.py`)

### Changed
- `/schedule` endpoint now accepts `List[List[str]]` contexts with automatic string-to-ID mapping
- Removed `incremental` field from `/build` — auto-detected based on existing index state
- Made server and documentation engine-agnostic (no longer SGLang-specific)
- Center-aligned benchmark image and table in README

### Fixed
- `/deduplicate` endpoint tests aligned with server schema (`parent_request_ids` plural)
- Documentation cleanup: fixed broken links, removed stale endpoint references

## [0.3.0] - 2026-01-30

### Added
- **PageIndex Integration**: Full support for PageIndex reasoning-based RAG retrieval
  - `PageIndexRetriever` class for tree-structured document navigation
  - LLM-based tree search for intelligent node retrieval
  - Shared context index optimization for batch queries
- End-to-end example for PageIndex + ContextPilot workflow
- Comprehensive test suite for PageIndex integration (13 new tests)
- Benchmark script for PageIndex performance evaluation
- GitHub Actions CI/CD pipeline for automated testing and releases

### Changed
- Improved context overlap detection and deduplication
- Enhanced documentation with PageIndex usage examples

### Fixed
- JSON parsing for LLM responses with markdown code blocks
- Tree structure handling for both dict and list formats

## [0.2.0] - 2025-12-15

### Added
- RAGPipeline unified API for retrieval-augmented generation
- Multiple retriever support (BM25, FAISS, Elasticsearch)
- GPU-accelerated distance computation with CuPy
- InterContextScheduler for optimal context ordering
- Mem0 integration for conversation memory

### Changed
- Refactored context index building for better performance
- Improved clustering algorithms with configurable linkage methods

## [0.1.0] - 2025-10-01

### Added
- Initial release
- Context index construction with hierarchical clustering
- Basic context ordering and optimization
- CPU-based distance computation
- Core ContextPilot pipeline
