"""
Test GPU distance computation performance across different dataset sizes.

This script benchmarks the performance of GPU-accelerated distance computation
for hierarchical clustering with varying numbers of contexts.
"""

import pytest
import numpy as np

# Skip entire module if CUDA is not available
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


def generate_synthetic_data(num_contexts, num_docs_per_context=20, total_docs=1000):
    """
    Generate synthetic retrieval data for testing.
    
    Args:
        num_contexts: Number of contexts/queries to generate
        num_docs_per_context: Number of retrieved documents per context
        total_docs: Total number of unique documents in the corpus
    
    Returns:
        List of document ID lists (top-k retrieval results)
    """
    topk_doc_ids = []
    for _ in range(num_contexts):
        doc_ids = np.random.choice(total_docs, size=num_docs_per_context, replace=False).tolist()
        topk_doc_ids.append(doc_ids)
    return topk_doc_ids


@pytest.fixture
def build_context_index():
    """Import the build_context_index function."""
    from contextpilot.context_index import build_context_index
    return build_context_index


class TestGPUDistancePerformance:
    """Benchmark GPU distance computation performance."""
    
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.parametrize("num_contexts", [64, 128, 512])
    def test_gpu_vs_cpu_small(self, build_context_index, num_contexts):
        """Test GPU vs CPU performance for small datasets."""
        import time
        
        topk_doc_ids = generate_synthetic_data(num_contexts)
        
        # GPU timing
        start = time.perf_counter()
        result_gpu = build_context_index(
            topk_doc_ids,
            linkage_method='average',
            use_gpu=True,
            alpha=0.005
        )
        gpu_time = time.perf_counter() - start
        
        # CPU timing
        start = time.perf_counter()
        result_cpu = build_context_index(
            topk_doc_ids,
            linkage_method='average',
            use_gpu=False,
            alpha=0.005
        )
        cpu_time = time.perf_counter() - start
        
        # Both should complete successfully
        assert result_gpu is not None
        assert result_cpu is not None
        
        # Log performance (for informational purposes)
        print(f"\n{num_contexts} contexts: GPU={gpu_time:.4f}s, CPU={cpu_time:.4f}s, speedup={cpu_time/gpu_time:.2f}x")
    
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.parametrize("num_contexts", [4096, 8192])
    def test_gpu_large_datasets(self, build_context_index, num_contexts):
        """Test GPU performance for larger datasets (CPU too slow)."""
        import time
        
        topk_doc_ids = generate_synthetic_data(num_contexts)
        
        start = time.perf_counter()
        result = build_context_index(
            topk_doc_ids,
            linkage_method='average',
            use_gpu=True,
            alpha=0.005
        )
        gpu_time = time.perf_counter() - start
        
        assert result is not None
        print(f"\n{num_contexts} contexts: GPU={gpu_time:.4f}s")
    
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_gpu_very_large_dataset(self, build_context_index):
        """Test GPU with 100k contexts."""
        import time
        
        num_contexts = 100000
        topk_doc_ids = generate_synthetic_data(num_contexts)
        
        start = time.perf_counter()
        result = build_context_index(
            topk_doc_ids,
            linkage_method='average',
            use_gpu=True,
            alpha=0.005
        )
        gpu_time = time.perf_counter() - start
        
        assert result is not None
        print(f"\n{num_contexts} contexts: GPU={gpu_time:.4f}s")
