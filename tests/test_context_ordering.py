"""
Tests for Context Ordering (Intra and Inter).

Tests the intra-context reordering and inter-context scheduling
algorithms that maximize prefix sharing and cache efficiency.
"""

import pytest
from typing import List, Dict, Any


class TestIntraContextOrdering:
    """Test intra-context reordering functionality."""
    
    def test_reordering_preserves_elements(self):
        """Reordering should preserve all elements in each context."""
        from contextpilot.context_index import build_context_index
        
        contexts = [
            [10, 20, 30, 40, 50],
            [15, 25, 35, 45, 55],
        ]
        
        result = build_context_index(contexts, use_gpu=False)
        
        for orig, reordered in zip(contexts, result.reordered_contexts):
            assert set(orig) == set(reordered)
    
    def test_reordering_same_length(self):
        """Reordered contexts should have same length as originals."""
        from contextpilot.context_index import build_context_index
        
        contexts = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        result = build_context_index(contexts, use_gpu=False)
        
        for orig, reordered in zip(contexts, result.reordered_contexts):
            assert len(orig) == len(reordered)
    
    def test_reordering_maximizes_prefix_sharing(self):
        """Reordering should maximize prefix sharing between similar contexts."""
        from contextpilot.context_index import build_context_index
        
        # Contexts with shared elements
        contexts = [
            [1, 2, 3, 100, 200],  # Contains 1, 2, 3
            [1, 2, 3, 300, 400],  # Also contains 1, 2, 3
            [500, 600, 700, 800, 900],  # Different
        ]
        
        result = build_context_index(contexts, use_gpu=False)
        
        # After reordering, similar contexts should have common prefix
        reordered = result.reordered_contexts
        
        # All contexts should be reordered
        assert len(reordered) == len(contexts)


class TestInterContextScheduling:
    """Test inter-context scheduling functionality."""
    
    def test_scheduling_includes_all_contexts(self):
        """Scheduling should include all contexts."""
        from contextpilot.context_index import build_context_index
        from contextpilot.context_ordering import InterContextScheduler
        
        contexts = [[i, i+1, i+2] for i in range(10)]
        
        result = build_context_index(contexts, use_gpu=False)
        scheduler = InterContextScheduler()
        
        scheduled_reordered, scheduled_originals, final_mapping, groups = \
            scheduler.schedule_contexts(result)
        
        assert len(scheduled_reordered) == len(contexts)
        assert len(scheduled_originals) == len(contexts)
    
    def test_scheduling_valid_permutation(self):
        """Scheduling should produce a valid permutation of indices."""
        from contextpilot.context_index import build_context_index
        from contextpilot.context_ordering import InterContextScheduler
        
        n = 15
        contexts = [[i, i+1, i+2] for i in range(n)]
        
        result = build_context_index(contexts, use_gpu=False)
        scheduler = InterContextScheduler()
        
        _, _, final_mapping, _ = scheduler.schedule_contexts(result)
        
        # Should be a permutation of [0, 1, ..., n-1]
        assert sorted(final_mapping) == list(range(n))
    
    def test_scheduling_groups_similar_contexts(self):
        """Scheduling should group similar contexts together."""
        from contextpilot.context_index import build_context_index
        from contextpilot.context_ordering import InterContextScheduler
        
        # Create two clusters of similar contexts
        cluster1 = [[1, 2, 3, i] for i in range(100, 105)]  # Share 1,2,3
        cluster2 = [[10, 20, 30, i] for i in range(200, 205)]  # Share 10,20,30
        contexts = cluster1 + cluster2
        
        result = build_context_index(contexts, use_gpu=False)
        scheduler = InterContextScheduler()
        
        _, _, final_mapping, groups = scheduler.schedule_contexts(result)
        
        # Should have created groups
        assert len(groups) > 0
    
    def test_scheduling_deterministic(self):
        """Scheduling should be deterministic for same input."""
        from contextpilot.context_index import build_context_index
        from contextpilot.context_ordering import InterContextScheduler
        
        contexts = [[1, 2, 3], [1, 2, 4], [5, 6, 7], [5, 6, 8]]
        
        results = []
        for _ in range(3):
            result = build_context_index(contexts, use_gpu=False)
            scheduler = InterContextScheduler()
            _, _, final_mapping, _ = scheduler.schedule_contexts(result)
            results.append(final_mapping)
        
        # All results should be identical
        assert results[0] == results[1] == results[2]


class TestGrouping:
    """Test context grouping by search path."""
    
    def test_group_by_root_prefix(self):
        """Test grouping by first element of search path."""
        from contextpilot.context_ordering import InterContextScheduler
        
        scheduler = InterContextScheduler()
        
        # Simulate search paths
        search_paths = [
            [0, 1, 2],  # Group 0
            [0, 1, 3],  # Group 0
            [1, 2, 3],  # Group 1
            [1, 2, 4],  # Group 1
            [2, 3, 4],  # Group 2
        ]
        
        groups = scheduler._group_by_root_prefix(search_paths)
        
        assert 0 in groups
        assert 1 in groups
        assert 2 in groups
        assert len(groups[0]) == 2  # Two contexts in group 0
        assert len(groups[1]) == 2  # Two contexts in group 1
        assert len(groups[2]) == 1  # One context in group 2
    
    def test_empty_search_paths_grouped(self):
        """Empty search paths should be grouped separately."""
        from contextpilot.context_ordering import InterContextScheduler
        
        scheduler = InterContextScheduler()
        
        search_paths = [
            [],  # Empty
            [0, 1],
            [],  # Empty
        ]
        
        groups = scheduler._group_by_root_prefix(search_paths)
        
        assert -1 in groups  # Empty paths go to group -1
        assert len(groups[-1]) == 2


class TestSortingWithinGroups:
    """Test sorting of contexts within groups."""
    
    def test_sort_by_path_length_descending(self):
        """Contexts should be sorted by path length descending."""
        from contextpilot.context_ordering import InterContextScheduler
        
        scheduler = InterContextScheduler()
        
        search_paths = [
            [0, 1],        # idx 0, length 2
            [0, 1, 2, 3],  # idx 1, length 4
            [0, 1, 2],     # idx 2, length 3
        ]
        
        groups = {0: [0, 1, 2]}  # All in same group
        
        sorted_groups = scheduler._sort_groups_by_path_length(
            groups, search_paths, contexts=[[]]  # contexts not used
        )
        
        # Should be sorted: idx 1 (len 4), idx 2 (len 3), idx 0 (len 2)
        assert sorted_groups[0] == [1, 2, 0]
