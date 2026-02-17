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
"""

from .pipeline import (
    RAGPipeline,
    RetrieverConfig,
    OptimizerConfig,
    InferenceConfig,
    PipelineConfig,
)

from .context_index import (
    ContextIndex,
    IndexResult,
)

from .context_ordering import (
    IntraContextOrderer,
)

from .server.live_index import ContextPilot

from .retriever import (
    BM25Retriever,
    FAISSRetriever,
    FAISS_AVAILABLE,
    Mem0Retriever,
    create_mem0_corpus_map,
    MEM0_AVAILABLE,
)

__version__ = "0.3.3"

__all__ = [
    # High-level pipeline API
    'RAGPipeline',
    'RetrieverConfig',
    'OptimizerConfig',
    'InferenceConfig',
    'PipelineConfig',
    
    # Core components
    'ContextIndex',
    'IndexResult',
    'IntraContextOrderer',
    'ContextPilot',
    
    # Retrievers
    'BM25Retriever',
    'FAISSRetriever',
    'FAISS_AVAILABLE',
    'Mem0Retriever',
    'create_mem0_corpus_map',
    'MEM0_AVAILABLE',
]
