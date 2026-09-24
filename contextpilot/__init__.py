"""
ContextPilot - Efficient Retrieval-Augmented Generation with Context Reuse

ContextPilot is a high-performance optimization system for RAG workloads that
maximizes KV cache efficiency through intelligent context reordering and
prefix sharing.

Quick Start:
    >>> from contextpilot.pipeline import RAGPipeline
    >>>
    >>> pipeline = RAGPipeline(
    ...     retriever="bm25",
    ...     corpus_path="corpus.jsonl",
    ...     model="Qwen/Qwen2.5-7B-Instruct"
    ... )
    >>>
    >>> results = pipeline.run(queries=["What is AI?"])

See docs/reference/api.md for detailed documentation.

Imports are lazy (PEP 562): the heavy RAG stack (``pipeline`` -> ``context_index``
-> ``scipy``) is only pulled in when one of its names is first accessed. This
keeps lightweight, dependency-free consumers -- such as the standalone token
monitor / provenance profiler in :mod:`contextpilot.hermes_opportunities` --
importable inside minimal environments where SciPy and friends are absent.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.4.1"

# Map each public name to the submodule that defines it. Submodules are imported
# on first attribute access, so importing ``contextpilot`` (or any lightweight
# subpackage like ``hermes_opportunities``) never eagerly drags in SciPy/NumPy.
_LAZY_EXPORTS = {
    # High-level pipeline API
    "RAGPipeline": ".pipeline",
    "RetrieverConfig": ".pipeline",
    "OptimizerConfig": ".pipeline",
    "InferenceConfig": ".pipeline",
    "PipelineConfig": ".pipeline",
    # Core components
    "ContextIndex": ".context_index",
    "IndexResult": ".context_index",
    "IntraContextOrderer": ".context_ordering",
    "ContextPilot": ".server.live_index",
    # Deduplication
    "dedup_chat_completions": ".dedup",
    "dedup_responses_api": ".dedup",
    "DedupResult": ".dedup",
    # Convenience functions
    "optimize": ".api",
    "optimize_batch": ".api",
    # Retrievers
    "BM25Retriever": ".retriever",
    "FAISSRetriever": ".retriever",
    "FAISS_AVAILABLE": ".retriever",
    "Mem0Retriever": ".retriever",
    "create_mem0_corpus_map": ".retriever",
    "MEM0_AVAILABLE": ".retriever",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Lazily resolve a public name to its (heavy) submodule on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent lookups skip the import machinery
    return value


def __dir__():
    return sorted(list(globals()) + __all__)


if TYPE_CHECKING:  # pragma: no cover - import-time hints for type checkers only
    from .api import optimize, optimize_batch
    from .context_index import ContextIndex, IndexResult
    from .context_ordering import IntraContextOrderer
    from .dedup import DedupResult, dedup_chat_completions, dedup_responses_api
    from .pipeline import (
        InferenceConfig,
        OptimizerConfig,
        PipelineConfig,
        RAGPipeline,
        RetrieverConfig,
    )
    from .retriever import (
        FAISS_AVAILABLE,
        MEM0_AVAILABLE,
        BM25Retriever,
        FAISSRetriever,
        Mem0Retriever,
        create_mem0_corpus_map,
    )
    from .server.live_index import ContextPilot
