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
        from contextpilot import ContextPilot
        
        index = ContextPilot(
            alpha=0.001,
            use_gpu=False,
        )
        
        assert index is not None
        assert index.is_live is False
    
    def test_live_index_with_different_configs(self):
        """Test live index with various configurations."""
        from contextpilot import ContextPilot
        
        configs = [
            {"alpha": 0.001},
            {"alpha": 0.01},
            {"alpha": 0.001},
        ]
        
        for config in configs:
            index = ContextPilot(use_gpu=False, **config)
            assert index is not None


class TestBuildAndSchedule:
    """Test build and schedule functionality."""
    
    def test_build_and_schedule(self):
        """Test building and scheduling contexts."""
        from contextpilot import ContextPilot
        
        index = ContextPilot(use_gpu=False)
        
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
        from contextpilot import ContextPilot
        
        index = ContextPilot(use_gpu=False)
        
        contexts = [[1, 2, 3], [4, 5, 6]]
        index.build_and_schedule(contexts)
        
        # build_and_schedule automatically sets is_live = True
        assert index.is_live is True
    
    def test_schedule_only_stateless(self):
        """Test schedule_only for stateless mode."""
        from contextpilot import ContextPilot
        
        index = ContextPilot(use_gpu=False)
        
        contexts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = index.schedule_only(contexts)
        
        assert result is not None
        assert 'reordered_contexts' in result
        assert 'scheduled_originals' in result
        assert 'original_indices' in result
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
        from contextpilot import ContextPilot
        
        index = ContextPilot(use_gpu=False)
        
        contexts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = index.build_and_schedule(contexts)
        
        # Should have request_id_mapping in result
        assert 'request_id_mapping' in result
        assert 'request_ids' in result
        assert len(result['request_ids']) == len(contexts)

    def test_reorder_single_list(self):
        """reorder() should accept a single list and auto-wrap it."""
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)
        # Pass a flat list instead of list-of-lists
        reordered, indices = engine.reorder([1, 2, 3])

        assert len(reordered) == 1
        assert set(reordered[0]) == {1, 2, 3}
        assert indices == [0]

    def test_reorder_single_list_strings(self):
        """reorder() should accept a single list of strings."""
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)
        reordered, indices = engine.reorder(["doc_a", "doc_b", "doc_c"])

        assert len(reordered) == 1
        assert set(reordered[0]) == {"doc_a", "doc_b", "doc_c"}
        assert indices == [0]


class TestDeduplication:
    """Test ContextPilot.deduplicate() for multi-turn deduplication."""

    def test_deduplicate_first_turn_no_overlap(self):
        """First deduplicate with no prior reorder → ValueError."""
        import pytest
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)
        with pytest.raises(ValueError, match="No prior .reorder\\(\\) call found"):
            engine.deduplicate([[10, 20, 30]], conversation_id="c1")

    def test_deduplicate_after_reorder(self):
        """Turn 2 should detect overlap with Turn 1 docs."""
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)

        # Turn 1: reorder registers docs
        engine.reorder([[1, 2, 3]], conversation_id="c1")

        # Turn 2: deduplicate — docs 1, 2 overlap
        results = engine.deduplicate([[1, 2, 5]], conversation_id="c1")

        assert len(results) == 1
        r = results[0]
        assert set(r["overlapping_docs"]) == {1, 2}
        assert r["new_docs"] == [5]
        assert r["deduplicated_docs"] == [5]
        assert len(r["reference_hints"]) == 2

    def test_deduplicate_chained_turns(self):
        """Turn 3 should see docs from both Turn 1 and Turn 2."""
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)

        # Turn 1
        engine.reorder([[1, 2, 3]], conversation_id="c1")

        # Turn 2
        engine.deduplicate([[3, 4, 5]], conversation_id="c1")

        # Turn 3 should see overlap from Turn 1 (doc 2) and Turn 2 (doc 4)
        results = engine.deduplicate([[2, 4, 6]], conversation_id="c1")

        r = results[0]
        assert set(r["overlapping_docs"]) == {2, 4}
        assert r["new_docs"] == [6]

    def test_deduplicate_no_overlap(self):
        """All-new docs should have empty overlap."""
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)

        engine.reorder([[1, 2, 3]], conversation_id="c1")
        results = engine.deduplicate([[10, 20, 30]], conversation_id="c1")

        r = results[0]
        assert r["overlapping_docs"] == []
        assert set(r["new_docs"]) == {10, 20, 30}

    def test_deduplicate_full_overlap(self):
        """All docs already seen → all are overlapping."""
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)

        engine.reorder([[1, 2, 3]], conversation_id="c1")
        results = engine.deduplicate([[1, 2, 3]], conversation_id="c1")

        r = results[0]
        assert set(r["overlapping_docs"]) == {1, 2, 3}
        assert r["new_docs"] == []

    def test_deduplicate_multiple_contexts(self):
        """Deduplicate a batch of contexts at once."""
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)

        engine.reorder([[1, 2], [3, 4]], conversation_id="c1")

        results = engine.deduplicate([[1, 5], [3, 6]], conversation_id="c1")

        assert len(results) == 2
        assert results[0]["overlapping_docs"] == [1]
        assert results[0]["new_docs"] == [5]
        assert results[1]["overlapping_docs"] == [3]
        assert results[1]["new_docs"] == [6]

    def test_deduplicate_isolated_conversations(self):
        """Different conversation_ids should be isolated."""
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)

        # User A
        engine.reorder([[1, 2, 3]], conversation_id="user_a")
        # User B
        engine.reorder([[10, 20, 30]], conversation_id="user_b")

        # User A's turn 2 — should only see overlap with user A's docs
        results_a = engine.deduplicate([[1, 10]], conversation_id="user_a")
        assert results_a[0]["overlapping_docs"] == [1]
        assert results_a[0]["new_docs"] == [10]

        # User B's turn 2 — should only see overlap with user B's docs
        results_b = engine.deduplicate([[1, 10]], conversation_id="user_b")
        assert results_b[0]["overlapping_docs"] == [10]
        assert results_b[0]["new_docs"] == [1]

    def test_deduplicate_string_contexts(self):
        """String contexts should work with deduplication."""
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)

        engine.reorder([["a", "b", "c"]], conversation_id="c1")
        results = engine.deduplicate([["a", "d"]], conversation_id="c1")

        r = results[0]
        assert len(r["overlapping_docs"]) == 1
        assert len(r["new_docs"]) == 1

    def test_deduplicate_custom_hint_template(self):
        """Custom hint template should be used."""
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)

        engine.reorder([[1, 2, 3]], conversation_id="c1")
        results = engine.deduplicate(
            [[1, 5]],
            conversation_id="c1",
            hint_template="See doc {doc_id} above.",
        )

        r = results[0]
        assert any("See doc" in h for h in r["reference_hints"])

    def test_deduplicate_cross_contamination_warning(self):
        """Warn on reorder without ID after explicit IDs; error on deduplicate without ID."""
        import warnings
        import pytest
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)

        # Explicit conversation_id — sets multi-user mode flag
        engine.reorder([[1, 2, 3]], conversation_id="user_a")

        # Now call reorder without conversation_id → should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.reorder([[4, 5, 6]])
            assert any("cross-contamination" in str(m.message) for m in w)

    def test_deduplicate_requires_conversation_id(self):
        """deduplicate() requires a non-empty conversation_id."""
        import pytest
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)
        engine.reorder([[1, 2, 3]])

        # Missing argument → TypeError from Python
        with pytest.raises(TypeError):
            engine.deduplicate([[1, 2]])

        # Empty string → ValueError from our validation
        with pytest.raises(ValueError, match="conversation_id is required"):
            engine.deduplicate([[1, 2]], conversation_id="")

    def test_no_warning_single_user_reorder(self):
        """No warnings for single-user quickstart flow (reorder only)."""
        import warnings
        from contextpilot import ContextPilot

        engine = ContextPilot(use_gpu=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.reorder([[1, 2, 3]])
            cross_warns = [m for m in w if "cross-contamination" in str(m.message)]
            assert len(cross_warns) == 0
