from .bm25 import BM25Retriever

# FAISS is optional - only import if available
try:
    from .faiss_embedding import FAISSRetriever
    FAISS_AVAILABLE = True
except ImportError:
    FAISSRetriever = None
    FAISS_AVAILABLE = False

__all__ = ["BM25Retriever", "FAISSRetriever", "FAISS_AVAILABLE"]