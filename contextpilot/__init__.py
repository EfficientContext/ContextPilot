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

See docs/PIPELINE_API.md for detailed documentation.
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
    build_context_index,
)

from .context_ordering import (
    IntraContextOrderer,
    InterContextScheduler,
)

from .retriever import (
    BM25Retriever,
    FAISSRetriever,
    FAISS_AVAILABLE,
    Mem0Retriever,
    create_mem0_corpus_map,
    MEM0_AVAILABLE,
)

__version__ = "0.2.0"

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
    'build_context_index',
    'IntraContextOrderer',
    'InterContextScheduler',
    
    # Retrievers
    'BM25Retriever',
    'FAISSRetriever',
    'FAISS_AVAILABLE',
    'Mem0Retriever',
    'create_mem0_corpus_map',
    'MEM0_AVAILABLE',
]
