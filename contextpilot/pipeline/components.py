"""
Configuration classes for RAG pipeline components.

These configuration classes provide a clean way to specify settings
for different parts of the RAG pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class RetrieverType(str, Enum):
    """Supported retriever types."""
    BM25 = "bm25"
    FAISS = "faiss"
    PAGEINDEX = "pageindex"
    CUSTOM = "custom"


@dataclass
class RetrieverConfig:
    """
    Configuration for document retrieval.
    
    Args:
        retriever_type: Type of retriever ("bm25", "faiss", or "custom")
        top_k: Number of documents to retrieve per query
        corpus_path: Path to corpus JSONL file or list of documents
        index_path: Path to save/load retrieval index (for FAISS)
        es_host: Elasticsearch host URL (for BM25)
        es_index_name: Elasticsearch index name (for BM25)
        embedding_model: Model for embeddings (for FAISS)
        embedding_base_url: URL for embedding service (for FAISS)
        custom_retriever: Custom retriever instance (for custom type)
        **kwargs: Additional retriever-specific parameters
    """
    retriever_type: str = "bm25"
    top_k: int = 20
    corpus_path: Optional[str] = None
    corpus_data: Optional[List[Dict[str, Any]]] = None
    
    # BM25-specific
    es_host: str = "http://localhost:9200"
    es_index_name: str = "bm25_index"
    
    # FAISS-specific
    index_path: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_base_url: Optional[str] = None
    hnsw_m: int = 64
    ef_construction: int = 400
    ef_search: int = 200
    
    # mem0-specific
    mem0_config: Optional[Dict[str, Any]] = None  # mem0 configuration dict
    mem0_memory: Optional[Any] = None  # Existing mem0 Memory instance
    mem0_api_key: Optional[str] = None  # API key for mem0 cloud
    mem0_use_client: bool = False  # Use mem0 MemoryClient instead of local Memory
    mem0_user_id: Optional[str] = None  # Default user_id for mem0 operations
    mem0_agent_id: Optional[str] = None  # Default agent_id for mem0 operations
    mem0_run_id: Optional[str] = None  # Default run_id for mem0 operations
    mem0_threshold: Optional[float] = None  # Score threshold for mem0 search
    mem0_rerank: bool = True  # Whether to use mem0's reranker
    
    # PageIndex-specific (reasoning-based tree search retrieval)
    pageindex_model: str = "gpt-4o"  # LLM model for tree generation and search
    pageindex_openai_api_key: Optional[str] = None  # OpenAI API key
    pageindex_tree_cache_dir: Optional[str] = None  # Directory to cache tree structures
    pageindex_document_paths: Optional[List[str]] = None  # List of PDF document paths
    pageindex_tree_paths: Optional[List[str]] = None  # List of pre-built tree structure paths
    
    # Custom retriever
    custom_retriever: Optional[Any] = None
    
    # Additional kwargs
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.retriever_type not in ["bm25", "faiss", "mem0", "pageindex", "custom"]:
            raise ValueError(f"Unsupported retriever type: {self.retriever_type}")
        
        if self.retriever_type == "custom" and self.custom_retriever is None:
            raise ValueError("custom_retriever must be provided when retriever_type='custom'")
        
        if self.retriever_type == "pageindex":
            # PageIndex requires either document_paths or tree_paths
            if not self.pageindex_document_paths and not self.pageindex_tree_paths:
                if not self.corpus_path and not self.corpus_data:
                    raise ValueError(
                        "PageIndex retriever requires either 'pageindex_document_paths', "
                        "'pageindex_tree_paths', 'corpus_path', or 'corpus_data'"
                    )
        
        if self.retriever_type == "mem0":
            # Validate mem0 configuration
            if self.mem0_use_client and self.mem0_api_key is None:
                raise ValueError("mem0_api_key is required when mem0_use_client=True")
            if not any([self.mem0_user_id, self.mem0_agent_id, self.mem0_run_id]):
                raise ValueError(
                    "At least one of 'mem0_user_id', 'mem0_agent_id', or 'mem0_run_id' "
                    "must be provided when using mem0 retriever"
                )


@dataclass
class OptimizerConfig:
    """
    Configuration for ContextPilot optimization.
    
    Args:
        enabled: Whether to use ContextPilot optimization
        use_gpu: Whether to use GPU for distance computation
        linkage_method: Clustering method ("average", "complete", "single")
        alpha: Weight for position differences in distance calculation (default: 0.001, recommended: 0.01-0.0001)
        reorder_contexts: Whether to reorder documents within contexts
        schedule_contexts: Whether to schedule context execution order
        extra_params: Additional optimizer parameters
    """
    enabled: bool = True
    use_gpu: bool = True
    linkage_method: str = "average"
    alpha: float = 0.001
    reorder_contexts: bool = True
    schedule_contexts: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.linkage_method not in ["average", "complete", "single"]:
            raise ValueError(f"Unsupported linkage method: {self.linkage_method}")


@dataclass
class InferenceConfig:
    """
    Configuration for LLM inference.
    
    Args:
        model_name: Name or path of the LLM model
        backend: Inference backend ("sglang", "vllm", "openai", "custom")
        base_url: Base URL for inference API
        port: Port for inference server
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
        prompt_template: Custom prompt template
        apply_chat_template: Whether to apply chat template to prompts
        system_prompt: System prompt to use with chat template
        extra_params: Additional inference parameters
    """
    model_name: str
    backend: str = "sglang"
    base_url: Optional[str] = None
    port: int = 30000
    temperature: float = 0.0
    max_tokens: int = 512
    batch_size: int = 1
    prompt_template: Optional[str] = None
    apply_chat_template: bool = True
    system_prompt: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration and set defaults."""
        if self.backend not in ["sglang", "vllm", "openai", "custom"]:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        if self.base_url is None:
            self.base_url = f"http://localhost:{self.port}"


@dataclass
class PipelineConfig:
    """
    Overall configuration for the RAG pipeline.
    
    Args:
        retriever: Retriever configuration
        optimizer: Optimizer configuration
        inference: Inference configuration
        cache_retrieval: Whether to cache retrieval results
        save_intermediate: Whether to save intermediate results
        output_path: Path to save final results
        verbose: Whether to print detailed logs
    """
    retriever: RetrieverConfig
    optimizer: OptimizerConfig
    inference: InferenceConfig
    cache_retrieval: bool = True
    save_intermediate: bool = False
    output_path: Optional[str] = None
    verbose: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary."""
        return cls(
            retriever=RetrieverConfig(**config_dict.get("retriever", {})),
            optimizer=OptimizerConfig(**config_dict.get("optimizer", {})),
            inference=InferenceConfig(**config_dict.get("inference", {})),
            cache_retrieval=config_dict.get("cache_retrieval", True),
            save_intermediate=config_dict.get("save_intermediate", False),
            output_path=config_dict.get("output_path"),
            verbose=config_dict.get("verbose", True),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "retriever": self.retriever.__dict__,
            "optimizer": self.optimizer.__dict__,
            "inference": self.inference.__dict__,
            "cache_retrieval": self.cache_retrieval,
            "save_intermediate": self.save_intermediate,
            "output_path": self.output_path,
            "verbose": self.verbose,
        }
