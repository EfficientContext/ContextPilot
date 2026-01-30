# Changelog

All notable changes to ContextPilot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
