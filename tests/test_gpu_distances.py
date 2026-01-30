#!/usr/bin/env python3
"""
Testing and verification suite for GPU distance computation.
"""

import pytest
import numpy as np

from tests.test_utils import generate_contexts


# Skip entire module if CUDA is not available
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
]


def compute_distance_cpu(context_i, context_j, alpha):
    """CPU reference implementation for distance calculation."""
    if len(context_i) == 0 or len(context_j) == 0:
        return 1.0
    
    pos_i = {chunk_id: idx for idx, chunk_id in enumerate(context_i)}
    pos_j = {chunk_id: idx for idx, chunk_id in enumerate(context_j)}
    
    set_i = set(context_i)
    set_j = set(context_j)
    intersection = set_i & set_j
    
    max_size = max(len(context_i), len(context_j))
    overlap_term = 1.0 - (len(intersection) / max_size)
    
    if len(intersection) == 0:
        position_term = 0.0
    else:
        position_diff_sum = sum(abs(pos_i[k] - pos_j[k]) for k in intersection)
        position_term = alpha * (position_diff_sum / len(intersection))
    
    return overlap_term + position_term


@pytest.fixture
def gpu_modules():
    """Import GPU modules."""
    from contextpilot.context_index.compute_distance_gpu import (
        get_gpu_info, 
        distance_kernel,
        prepare_contexts_for_gpu,
        compute_distance_matrix_gpu
    )
    return {
        'get_gpu_info': get_gpu_info,
        'distance_kernel': distance_kernel,
        'prepare_contexts_for_gpu': prepare_contexts_for_gpu,
        'compute_distance_matrix_gpu': compute_distance_matrix_gpu,
    }


class TestGPUDistanceVerification:
    """Verify GPU matches CPU implementation."""
    
    @pytest.fixture
    def contexts(self):
        """Generate test contexts."""
        return generate_contexts(100, avg_chunks=20, seed=42)
    
    @pytest.mark.parametrize("alpha", [0.001, 0.002, 0.005])
    @pytest.mark.parametrize("num_test", [20, 50])
    def test_gpu_matches_cpu(self, contexts, gpu_modules, alpha, num_test):
        """Verify GPU matches CPU for various configurations."""
        test_contexts = contexts[:num_test]
        n_test = len(test_contexts)
        
        # Prepare GPU data
        chunk_ids, original_positions, lengths, offsets = gpu_modules['prepare_contexts_for_gpu'](test_contexts)
        
        d_chunk_ids = cp.asarray(chunk_ids)
        d_original_positions = cp.asarray(original_positions)
        d_lengths = cp.asarray(lengths)
        d_offsets = cp.asarray(offsets)
        
        # CPU results
        cpu_results = []
        for i in range(n_test):
            for j in range(i + 1, n_test):
                cpu_results.append(compute_distance_cpu(test_contexts[i], test_contexts[j], alpha))
        
        # GPU results
        test_pairs = n_test * (n_test - 1) // 2
        d_test_output = cp.zeros(test_pairs, dtype=cp.float32)
        
        threads_per_block = 256
        blocks = (test_pairs + threads_per_block - 1) // threads_per_block
        
        gpu_modules['distance_kernel'](
            (blocks,), (threads_per_block,),
            (d_chunk_ids, d_original_positions, d_lengths, d_offsets,
             d_test_output, n_test, 0, n_test, np.float32(alpha))
        )
        
        cp.cuda.Stream.null.synchronize()
        gpu_results = cp.asnumpy(d_test_output)
        
        # Verify no error sentinels
        assert not np.any(gpu_results < 0), "GPU kernel returned error sentinels"
        
        # Compare
        cpu_results = np.array(cpu_results)
        diffs = np.abs(cpu_results - gpu_results)
        
        tolerance = 1e-4
        within_tol = np.sum(diffs < tolerance)
        pct = 100.0 * within_tol / len(diffs)
        
        assert pct >= 99.9, f"Only {pct:.2f}% within tolerance"


class TestGPUDistanceMatrix:
    """Test full GPU distance matrix computation."""
    
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_full_distance_matrix(self, gpu_modules):
        """Test full distance matrix computation."""
        num_contexts = 1000
        alpha = 0.001
        batch_rows = 500
        
        contexts = generate_contexts(num_contexts, avg_chunks=20, seed=42)
        
        # Compute
        condensed_matrix = gpu_modules['compute_distance_matrix_gpu'](contexts, alpha, batch_rows)
        
        assert condensed_matrix is not None, "Computation failed"
        
        expected_length = num_contexts * (num_contexts - 1) // 2
        assert len(condensed_matrix) == expected_length
        assert condensed_matrix.min() >= 0, "Distances should be non-negative"
        assert not np.any(np.isnan(condensed_matrix)), "Found NaN values"
    
    @pytest.mark.gpu
    def test_gpu_info(self, gpu_modules):
        """Test GPU info retrieval."""
        gpu_info = gpu_modules['get_gpu_info'](0)
        assert 'name' in gpu_info
        assert gpu_info['name'] is not None
