from .bm25 import BM25Retriever

# FAISS is optional - only import if available
try:
    from .faiss_embedding import FAISSRetriever
    FAISS_AVAILABLE = True
except ImportError:
    FAISSRetriever = None
    FAISS_AVAILABLE = False

# mem0 is optional - only import if available
try:
    from .mem0_retriever import Mem0Retriever, create_mem0_corpus_map, MEM0_AVAILABLE
except ImportError:
    Mem0Retriever = None
    create_mem0_corpus_map = None
    MEM0_AVAILABLE = False

# PageIndex is optional - only import if available
try:
    from .pageindex_retriever import PageIndexRetriever
    PAGEINDEX_AVAILABLE = True
except ImportError:
    PageIndexRetriever = None
    PAGEINDEX_AVAILABLE = False

__all__ = [
    "BM25Retriever",
    "FAISSRetriever",
    "FAISS_AVAILABLE",
    "Mem0Retriever",
    "create_mem0_corpus_map",
    "MEM0_AVAILABLE",
    "PageIndexRetriever",
    "PAGEINDEX_AVAILABLE",
]