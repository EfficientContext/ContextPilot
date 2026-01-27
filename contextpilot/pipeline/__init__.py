"""
ContextPilot Pipeline Module

Provides high-level abstractions for building end-to-end RAG pipelines
with or without ContextPilot optimization.

Quick Start:
    
    # With ContextPilot optimization
    from contextpilot.pipeline import RAGPipeline
    
    pipeline = RAGPipeline(
        retriever="bm25",
        corpus_path="corpus.jsonl",
        use_contextpilot=True
    )
    
    results = pipeline.run(
        queries=["What is AI?", "What is ML?"],
        model="Qwen/Qwen2.5-7B-Instruct"
    )
    
    # Without ContextPilot (standard RAG)
    pipeline = RAGPipeline(
        retriever="faiss",
        corpus_path="corpus.jsonl",
        use_contextpilot=False
    )
    
    results = pipeline.run(queries=["What is AI?"])
"""

from .rag_pipeline import RAGPipeline
from .multi_turn import MultiTurnManager, ConversationState
from .components import (
    RetrieverConfig,
    OptimizerConfig,
    InferenceConfig,
    PipelineConfig
)

__all__ = [
    'RAGPipeline',
    'MultiTurnManager',
    'ConversationState',
    'RetrieverConfig',
    'OptimizerConfig',
    'InferenceConfig',
    'PipelineConfig',
]
