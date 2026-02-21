#!/usr/bin/env python3
"""
Testing and benchmarking suite for optimized CPU distance computation.
"""

import pytest
import numpy as np
from contextpilot.context_index.compute_distance_cpu import (
    prepare_contexts_for_cpu,
    compute_distance_optimized,
    compute_distance_matrix_cpu_optimized,
)

from tests.test_utils import generate_contexts


def compute_distance_naive(context_i, context_j, alpha):
    """
    Naive implementation using dict/set (for comparison).
    This is the old/slow method.
    """
    if len(context_i) == 0 or len(context_j) == 0:
        return 1.0
    
    # Build position maps
    pos_i = {chunk_id: idx for idx, chunk_id in enumerate(context_i)}
    pos_j = {chunk_id: idx for idx, chunk_id in enumerate(context_j)}
    
    # Find intersection
    set_i = set(context_i)
    set_j = set(context_j)
    intersection = set_i & set_j
    
    # Overlap term
    max_size = max(len(context_i), len(context_j))
    overlap_term = 1.0 - (len(intersection) / max_size)
    
    # Position term
    if len(intersection) == 0:
        position_term = 0.0
    else:
        position_diff_sum = sum(abs(pos_i[k] - pos_j[k]) for k in intersection)
        position_term = alpha * (position_diff_sum / len(intersection))
    
    return overlap_term + position_term


class TestCPUDistanceCorrectness:
    """Verify optimized implementation matches naive implementation."""
    
    @pytest.fixture
    def contexts(self):
        """Generate test contexts."""
        return generate_contexts(100, avg_chunks=20, seed=42)
    
    def test_optimized_matches_naive(self, contexts):
        """Verify optimized implementation matches naive implementation."""
        alpha = 0.001
        num_test = 50
        test_contexts = contexts[:num_test]
        n = len(test_contexts)
        
        # Prepare data for optimized version
        chunk_ids, original_positions, lengths, offsets = prepare_contexts_for_cpu(test_contexts)
        
        # Compute with both methods
        naive_results = []
        optimized_results = []
        
        for i in range(n):
            for j in range(i + 1, n):
                naive_dist = compute_distance_naive(test_contexts[i], test_contexts[j], alpha)
                optimized_dist = compute_distance_optimized(
                    chunk_ids, original_positions, lengths, offsets, i, j, alpha
                )
                
                naive_results.append(naive_dist)
                optimized_results.append(optimized_dist)
        
        # Compare
        naive_results = np.array(naive_results)
        optimized_results = np.array(optimized_results)
        
        diffs = np.abs(naive_results - optimized_results)
        max_diff = diffs.max()
        
        assert max_diff < 1e-6, f"Max difference {max_diff} exceeds tolerance"
    
    @pytest.mark.parametrize("alpha", [0.001, 0.001, 0.01])
    def test_various_alpha_values(self, contexts, alpha):
        """Test with different alpha values."""
        test_contexts = contexts[:20]
        n = len(test_contexts)
        
        chunk_ids, original_positions, lengths, offsets = prepare_contexts_for_cpu(test_contexts)
        
        for i in range(n):
            for j in range(i + 1, n):
                naive_dist = compute_distance_naive(test_contexts[i], test_contexts[j], alpha)
                optimized_dist = compute_distance_optimized(
                    chunk_ids, original_positions, lengths, offsets, i, j, alpha
                )
                
                assert abs(naive_dist - optimized_dist) < 1e-6


class TestFullPipeline:
    """Test the full multi-threaded pipeline."""
    
    @pytest.mark.slow
    def test_distance_matrix_computation(self):
        """Test the full multi-threaded pipeline."""
        num_contexts = 1000
        contexts = generate_contexts(num_contexts, avg_chunks=20, seed=42)
        
        # Compute matrix
        condensed = compute_distance_matrix_cpu_optimized(
            contexts, 
            alpha=0.001, 
            num_workers=None,
            batch_size=1000
        )
        
        # Verify result
        expected_length = num_contexts * (num_contexts - 1) // 2
        assert len(condensed) == expected_length, f"Expected {expected_length}, got {len(condensed)}"
        assert condensed.min() >= 0, "Distances should be non-negative"
        assert condensed.max() <= 3.0, f"Max distance {condensed.max()} seems too high"
        assert not np.any(np.isnan(condensed)), "Found NaN values in distance matrix"
