# Changelog

All notable changes to ContextPilot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2026-02-16

### Added
- `LiveContextIndex` exported from top-level package for convenient imports
- Automatic string-to-integer context conversion in `ContextIndex` and `LiveContextIndex` — `List[List[str]]` inputs are now accepted natively
- Leaf-node splitting in `LiveContextIndex.build_incremental()` for finer-grained prefix sharing
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
