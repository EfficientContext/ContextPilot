"""
Tests for Live Index and Server Components.

Tests dynamic index operations including search, insertion,
eviction, and request tracking for online serving.
"""

import pytest
from typing import List, Dict, Optional


class TestLiveIndexInitialization:
    """Test live index initialization."""
    
    def test_live_index_creation(self):
        """Test basic live index creation."""
        from contextpilot.server.live_index import LiveContextIndex
        
        index = LiveContextIndex(
            alpha=0.005,
            use_gpu=False,
        )
        
        assert index is not None
        assert index.is_live is False
    
    def test_live_index_with_different_configs(self):
        """Test live index with various configurations."""
        from contextpilot.server.live_index import LiveContextIndex
        
        configs = [
            {"alpha": 0.001},
            {"alpha": 0.01},
            {"alpha": 0.005},
        ]
        
        for config in configs:
            index = LiveContextIndex(use_gpu=False, **config)
            assert index is not None


class TestBuildAndSchedule:
    """Test build and schedule functionality."""
    
    def test_build_and_schedule(self):
        """Test building and scheduling contexts."""
        from contextpilot.server.live_index import LiveContextIndex
        
        index = LiveContextIndex(use_gpu=False)
        
        contexts = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 6, 7],
            [8, 9, 10, 11, 12],
        ]
        
        result = index.build_and_schedule(contexts)
        
        assert result is not None
        assert index.initial_result is not None
        assert index.scheduled_result is not None
    
    def test_index_becomes_live_after_build(self):
        """Test that index becomes live after build_and_schedule."""
        from contextpilot.server.live_index import LiveContextIndex
        
        index = LiveContextIndex(use_gpu=False)
        
        contexts = [[1, 2, 3], [4, 5, 6]]
        index.build_and_schedule(contexts)
        
        # build_and_schedule automatically sets is_live = True
        assert index.is_live is True
    
    def test_schedule_only_stateless(self):
        """Test schedule_only for stateless mode."""
        from contextpilot.server.live_index import LiveContextIndex
        
        index = LiveContextIndex(use_gpu=False)
        
        contexts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = index.schedule_only(contexts)
        
        assert result is not None
        assert 'scheduled_reordered' in result
        assert 'scheduled_originals' in result
        assert 'final_mapping' in result
        # In stateless mode, is_live should remain False
        assert index.is_live is False


class TestEvictionHeap:
    """Test eviction heap functionality."""
    
    def test_eviction_heap_initialization(self):
        """Test eviction heap initializes correctly."""
        from contextpilot.server.eviction_heap import EvictionHeap
        
        heap = EvictionHeap(max_tokens=10000)
        
        assert heap is not None
        assert heap.max_tokens == 10000
    
    def test_eviction_heap_push(self):
        """Test pushing metadata to eviction heap."""
        from contextpilot.server.eviction_heap import EvictionHeap
        from contextpilot.server.metadata import NodeMetadata
        
        heap = EvictionHeap(max_tokens=10000)
        
        metadata = NodeMetadata(node_id=1, total_tokens=100, extra_tokens=50)
        heap.push(metadata)
        
        assert len(heap) == 1
    
    def test_eviction_heap_pop(self):
        """Test popping from eviction heap."""
        from contextpilot.server.eviction_heap import EvictionHeap
        from contextpilot.server.metadata import NodeMetadata
        import time
        
        heap = EvictionHeap(max_tokens=10000)
        
        # Add items with different timestamps
        m1 = NodeMetadata(node_id=1, total_tokens=100, extra_tokens=50)
        m1.last_access_time = time.time() - 100  # Oldest
        
        m2 = NodeMetadata(node_id=2, total_tokens=100, extra_tokens=50)
        m2.last_access_time = time.time()  # Newest
        
        heap.push(m1)
        heap.push(m2)
        
        # Pop should return oldest (LRU)
        popped = heap.pop()
        assert popped.node_id == 1


class TestNodeMetadata:
    """Test node metadata handling."""
    
    def test_metadata_creation(self):
        """Test creating node metadata."""
        from contextpilot.server.metadata import NodeMetadata
        
        metadata = NodeMetadata(
            node_id=1,
            total_tokens=100,
            extra_tokens=50
        )
        
        assert metadata.node_id == 1
        assert metadata.total_tokens == 100
        assert metadata.extra_tokens == 50
    
    def test_metadata_access_time_update(self):
        """Test updating access time."""
        from contextpilot.server.metadata import NodeMetadata
        import time
        
        metadata = NodeMetadata(node_id=1)
        old_time = metadata.last_access_time
        
        time.sleep(0.01)
        metadata.update_access_time()
        
        assert metadata.last_access_time > old_time
    
    def test_metadata_add_tokens(self):
        """Test adding tokens to metadata."""
        from contextpilot.server.metadata import NodeMetadata
        
        metadata = NodeMetadata(node_id=1, total_tokens=100, extra_tokens=50)
        
        metadata.add_tokens(25)
        
        assert metadata.total_tokens == 125
        assert metadata.extra_tokens == 75
    
    def test_metadata_remove_tokens(self):
        """Test removing tokens from metadata."""
        from contextpilot.server.metadata import NodeMetadata
        
        metadata = NodeMetadata(node_id=1, total_tokens=100, extra_tokens=50)
        
        removed = metadata.remove_tokens(30)
        
        assert removed == 30
        assert metadata.extra_tokens == 20
        assert metadata.total_tokens == 70


class TestComputePrefixLength:
    """Test prefix length computation utility."""
    
    def test_identical_lists(self):
        """Identical lists should have full prefix length."""
        from contextpilot.server.live_index import compute_prefix_length
        
        list1 = [1, 2, 3, 4, 5]
        list2 = [1, 2, 3, 4, 5]
        
        assert compute_prefix_length(list1, list2) == 5
    
    def test_partial_prefix(self):
        """Lists with partial prefix should return correct length."""
        from contextpilot.server.live_index import compute_prefix_length
        
        list1 = [1, 2, 3, 100, 200]
        list2 = [1, 2, 3, 300, 400]
        
        assert compute_prefix_length(list1, list2) == 3
    
    def test_no_common_prefix(self):
        """Lists with no common prefix should return 0."""
        from contextpilot.server.live_index import compute_prefix_length
        
        list1 = [1, 2, 3]
        list2 = [4, 5, 6]
        
        assert compute_prefix_length(list1, list2) == 0
    
    def test_empty_list(self):
        """Empty list should have 0 prefix length."""
        from contextpilot.server.live_index import compute_prefix_length
        
        assert compute_prefix_length([], [1, 2, 3]) == 0
        assert compute_prefix_length([1, 2, 3], []) == 0
        assert compute_prefix_length([], []) == 0
    
    def test_different_length_lists(self):
        """Lists of different lengths should work correctly."""
        from contextpilot.server.live_index import compute_prefix_length
        
        list1 = [1, 2, 3]
        list2 = [1, 2, 3, 4, 5]
        
        assert compute_prefix_length(list1, list2) == 3


class TestLiveIndexRequestTracking:
    """Test request tracking in live index."""
    
    def test_request_id_auto_generated(self):
        """Test that request IDs are auto-generated during build."""
        from contextpilot.server.live_index import LiveContextIndex
        
        index = LiveContextIndex(use_gpu=False)
        
        contexts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = index.build_and_schedule(contexts)
        
        # Should have request_id_mapping in result
        assert 'request_id_mapping' in result
        assert 'request_ids' in result
        assert len(result['request_ids']) == len(contexts)
