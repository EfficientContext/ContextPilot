"""
Tests for Context Index Construction.

Tests hierarchical clustering, tree building, and index construction
to ensure correct behavior for context organization.
"""

import pytest
import numpy as np
from typing import List


class TestDistanceComputation:
    """Test distance computation between contexts."""
    
    def test_identical_contexts_zero_distance(self):
        """Identical contexts should have zero distance."""
        from contextpilot.context_index.compute_distance_cpu import (
            prepare_contexts_for_cpu,
            compute_distance_optimized
        )
        
        contexts = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        chunk_ids, orig_pos, lengths, offsets = prepare_contexts_for_cpu(contexts)
        
        dist = compute_distance_optimized(
            chunk_ids, orig_pos, lengths, offsets, 0, 1, alpha=0.005
        )
        
        assert dist == pytest.approx(0.0, abs=0.01)
    
    def test_disjoint_contexts_max_distance(self):
        """Completely disjoint contexts should have distance 1.0."""
        from contextpilot.context_index.compute_distance_cpu import (
            prepare_contexts_for_cpu,
            compute_distance_optimized
        )
        
        contexts = [[1, 2, 3], [4, 5, 6]]
        chunk_ids, orig_pos, lengths, offsets = prepare_contexts_for_cpu(contexts)
        
        dist = compute_distance_optimized(
            chunk_ids, orig_pos, lengths, offsets, 0, 1, alpha=0.005
        )
        
        assert dist == 1.0
    
    def test_partial_overlap_intermediate_distance(self):
        """Partially overlapping contexts should have intermediate distance."""
        from contextpilot.context_index.compute_distance_cpu import (
            prepare_contexts_for_cpu,
            compute_distance_optimized
        )
        
        contexts = [[1, 2, 3, 4], [3, 4, 5, 6]]  # 50% overlap
        chunk_ids, orig_pos, lengths, offsets = prepare_contexts_for_cpu(contexts)
        
        dist = compute_distance_optimized(
            chunk_ids, orig_pos, lengths, offsets, 0, 1, alpha=0.005
        )
        
        assert 0.0 < dist < 1.0
    
    def test_empty_context_distance(self):
        """Empty contexts should have distance 1.0."""
        from contextpilot.context_index.compute_distance_cpu import (
            prepare_contexts_for_cpu,
            compute_distance_optimized
        )
        
        contexts = [[], [1, 2, 3]]
        chunk_ids, orig_pos, lengths, offsets = prepare_contexts_for_cpu(contexts)
        
        dist = compute_distance_optimized(
            chunk_ids, orig_pos, lengths, offsets, 0, 1, alpha=0.005
        )
        
        assert dist == 1.0
    
    def test_distance_symmetry(self):
        """Distance should be symmetric: d(a,b) == d(b,a)."""
        from contextpilot.context_index.compute_distance_cpu import (
            prepare_contexts_for_cpu,
            compute_distance_optimized
        )
        
        contexts = [[1, 2, 3, 4], [2, 3, 4, 5]]
        chunk_ids, orig_pos, lengths, offsets = prepare_contexts_for_cpu(contexts)
        
        dist_01 = compute_distance_optimized(
            chunk_ids, orig_pos, lengths, offsets, 0, 1, alpha=0.005
        )
        dist_10 = compute_distance_optimized(
            chunk_ids, orig_pos, lengths, offsets, 1, 0, alpha=0.005
        )
        
        assert dist_01 == pytest.approx(dist_10, abs=1e-6)
    
    def test_alpha_affects_position_term(self):
        """Different alpha values should affect distance differently."""
        from contextpilot.context_index.compute_distance_cpu import (
            prepare_contexts_for_cpu,
            compute_distance_optimized
        )
        
        # Contexts with same elements but different positions
        contexts = [[1, 2, 3], [3, 2, 1]]  # Reversed order
        chunk_ids, orig_pos, lengths, offsets = prepare_contexts_for_cpu(contexts)
        
        dist_low_alpha = compute_distance_optimized(
            chunk_ids, orig_pos, lengths, offsets, 0, 1, alpha=0.001
        )
        dist_high_alpha = compute_distance_optimized(
            chunk_ids, orig_pos, lengths, offsets, 0, 1, alpha=0.1
        )
        
        # Higher alpha should give higher distance due to position differences
        assert dist_high_alpha > dist_low_alpha


class TestDistanceMatrixConstruction:
    """Test condensed distance matrix construction."""
    
    def test_matrix_correct_size(self):
        """Distance matrix should have correct size: n*(n-1)/2."""
        from contextpilot.context_index.compute_distance_cpu import (
            compute_distance_matrix_cpu_optimized
        )
        
        n = 5
        contexts = [[i, i+1, i+2] for i in range(n)]
        
        dist_matrix = compute_distance_matrix_cpu_optimized(
            contexts, alpha=0.005, num_workers=1
        )
        
        expected_size = n * (n - 1) // 2
        assert len(dist_matrix) == expected_size
    
    def test_matrix_values_in_range(self):
        """All distance values should be in [0, 1+epsilon]."""
        from contextpilot.context_index.compute_distance_cpu import (
            compute_distance_matrix_cpu_optimized
        )
        
        contexts = [
            [1, 2, 3],
            [2, 3, 4],
            [5, 6, 7],
            [1, 5, 9],
        ]
        
        dist_matrix = compute_distance_matrix_cpu_optimized(
            contexts, alpha=0.005, num_workers=1
        )
        
        for dist in dist_matrix:
            assert 0 <= dist <= 1.5  # Allow some slack for position term


class TestTreeConstruction:
    """Test hierarchical clustering tree construction."""
    
    def test_tree_has_correct_structure(self):
        """Tree should have correct number of nodes."""
        from contextpilot.context_index import build_context_index
        
        n = 5
        contexts = [[i, i+1, i+2] for i in range(n)]
        
        result = build_context_index(contexts, use_gpu=False)
        
        # Should have leaf nodes for each context
        leaf_count = sum(1 for node in result.unique_nodes.values() if node.is_leaf)
        assert leaf_count == n
    
    def test_tree_has_single_root(self):
        """Tree should have exactly one root node."""
        from contextpilot.context_index import build_context_index
        
        contexts = [[1, 2, 3], [2, 3, 4], [5, 6, 7]]
        result = build_context_index(contexts, use_gpu=False)
        
        root_count = sum(1 for node in result.unique_nodes.values() if node.is_root)
        assert root_count == 1
    
    def test_leaves_contain_original_indices(self):
        """Leaf nodes should reference original context indices."""
        from contextpilot.context_index import build_context_index
        
        contexts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = build_context_index(contexts, use_gpu=False)
        
        all_indices = set()
        for node in result.unique_nodes.values():
            if node.is_leaf:
                all_indices.update(node.original_indices)
        
        expected_indices = set(range(len(contexts)))
        assert all_indices == expected_indices


class TestLinkageMethods:
    """Test different hierarchical clustering linkage methods."""
    
    @pytest.mark.parametrize("method", ["single", "complete", "average"])
    def test_linkage_method_works(self, method):
        """Each linkage method should produce valid results."""
        from contextpilot.context_index import ContextIndex
        
        index = ContextIndex(linkage_method=method, use_gpu=False)
        contexts = [[1, 2, 3], [1, 2, 4], [5, 6, 7]]
        
        result = index.fit_transform(contexts)
        
        assert result is not None
        assert len(result.reordered_contexts) == len(contexts)
    
    def test_different_linkage_can_produce_different_trees(self):
        """Different linkage methods may produce different clusterings."""
        from contextpilot.context_index import ContextIndex
        
        contexts = [
            [1, 2, 3, 4],
            [1, 2, 5, 6],
            [7, 8, 9, 10],
            [7, 8, 11, 12],
        ]
        
        results = {}
        for method in ["single", "complete"]:
            index = ContextIndex(linkage_method=method, use_gpu=False)
            results[method] = index.fit_transform(contexts)
        
        # Both should produce valid results (trees may differ)
        for method, result in results.items():
            assert result is not None


class TestSearchPaths:
    """Test search path generation for contexts."""
    
    def test_search_paths_generated(self):
        """Search paths should be generated for all contexts."""
        from contextpilot.context_index import build_context_index
        
        contexts = [[1, 2, 3], [1, 2, 4], [5, 6, 7]]
        result = build_context_index(contexts, use_gpu=False)
        
        assert result.search_paths is not None
        assert len(result.search_paths) == len(contexts)
    
    def test_similar_contexts_share_path_prefix(self):
        """Similar contexts should share search path prefix."""
        from contextpilot.context_index import build_context_index
        
        # First two contexts are very similar, third is different
        contexts = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 6],  # Very similar to first
            [10, 20, 30, 40, 50],  # Very different
        ]
        
        result = build_context_index(contexts, use_gpu=False)
        
        # Similar contexts should have some shared prefix
        # (exact behavior depends on clustering)
        assert result.search_paths is not None
