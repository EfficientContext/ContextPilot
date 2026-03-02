"""
Tests for the SGLang monkey-patch approach (contextpilot/_sglang_hook.py).

These tests use a lightweight mock of SGLang's RadixCache to verify the
monkey-patching logic without requiring SGLang to be installed.
"""

import os
import sys
import types
import heapq
import pytest
from collections import defaultdict
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Lightweight mock of SGLang's radix cache primitives
# ---------------------------------------------------------------------------

class MockRadixKey:
    """Minimal stand-in for sglang.srt.mem_cache.radix_cache.RadixKey."""

    def __init__(self, token_ids, extra_key=None):
        self.token_ids = list(token_ids)
        self.extra_key = extra_key

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, key):
        return MockRadixKey(self.token_ids[key], self.extra_key)

    def __repr__(self):
        return f"MockRadixKey({self.token_ids})"


class MockTreeNode:
    """Minimal stand-in for sglang.srt.mem_cache.radix_cache.TreeNode."""

    counter = 0

    def __init__(self, priority=0):
        self.children = {}
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = 0
        self.priority = priority
        self.id = MockTreeNode.counter
        MockTreeNode.counter += 1

    def __lt__(self, other):
        return self.last_access_time < other.last_access_time


def _mock_key_match(a, b):
    """Return length of common prefix between two MockRadixKeys."""
    n = min(len(a), len(b))
    for i in range(n):
        if a.token_ids[i] != b.token_ids[i]:
            return i
    return n


class MockRadixCache:
    """Minimal stand-in for sglang.srt.mem_cache.radix_cache.RadixCache.

    Implements enough of the real interface (insert, evict, _delete_leaf,
    _insert_helper, cache_finished_req, cache_unfinished_req, reset) so the
    monkey-patch can be applied and exercised.
    """

    def __init__(self):
        self.disable = False
        self.evictable_size_ = 0
        self.page_size = 1
        self.get_child_key_fn = lambda key: key.token_ids[0]
        self.key_match_fn = _mock_key_match
        self.reset()

    def reset(self):
        self.root_node = MockTreeNode()
        self.root_node.key = MockRadixKey([])
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0

    def insert(self, key, value=None, chunked=False, priority=0):
        if self.disable:
            return 0
        return self._insert_helper(self.root_node, key, value, priority)

    def _insert_helper(self, node, key, value, priority=0):
        node.last_access_time += 1
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            node.last_access_time += 1
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:] if value is not None else None

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                node = new_node
            else:
                node.priority = max(node.priority, priority)

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = MockTreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = list(range(len(key)))  # placeholder values
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)

        return total_prefix_length

    def _split_node(self, key, child, split_len):
        new_node = MockTreeNode(priority=child.priority)
        suffix_key = self.get_child_key_fn(key[split_len:])
        new_node.children = {suffix_key: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len] if child.value else []
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:] if child.value else []
        parent_key = self.get_child_key_fn(key)
        new_node.parent.children[parent_key] = new_node
        return new_node

    def _delete_leaf(self, node):
        for k, v in list(node.parent.children.items()):
            if v is node:
                del node.parent.children[k]
                break
        self.evictable_size_ -= len(node.key)

    def evict(self, num_tokens):
        leaves = self._collect_leaves()
        heap = [(n.last_access_time, n) for n in leaves]
        heapq.heapify(heap)
        num_evicted = 0
        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            num_evicted += len(x.key)
            self._delete_leaf(x)
            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                heapq.heappush(heap, (x.parent.last_access_time, x.parent))
        return num_evicted

    def _collect_leaves(self):
        leaves = []
        stack = list(self.root_node.children.values())
        while stack:
            node = stack.pop()
            if len(node.children) == 0:
                if node.lock_ref == 0:
                    leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def cache_finished_req(self, req, is_insert=True):
        if is_insert:
            key = MockRadixKey(req.fill_ids, req.extra_key)
            self.insert(key, priority=getattr(req, "priority", 0) or 0)

    def cache_unfinished_req(self, req, chunked=False):
        key = MockRadixKey(req.fill_ids, req.extra_key)
        self.insert(key, priority=getattr(req, "priority", 0) or 0)

    def maybe_bigram_convert(self, key, value=None):
        return key, value


class MockReq:
    """Minimal stand-in for sglang.srt.managers.schedule_batch.Req."""

    def __init__(self, rid, fill_ids, extra_key=None, priority=0):
        self.rid = rid
        self.fill_ids = fill_ids
        self.extra_key = extra_key
        self.priority = priority


# ---------------------------------------------------------------------------
# Helper: build a fake module and apply the monkey-patch
# ---------------------------------------------------------------------------

def _build_patched_module(index_url="http://test:8765"):
    """Create a fake sglang.srt.mem_cache.radix_cache module, patch it, return it."""
    from contextpilot._sglang_hook import _apply_radix_cache_patches

    module = types.ModuleType("sglang.srt.mem_cache.radix_cache")
    module.TreeNode = MockTreeNode
    module.RadixCache = MockRadixCache

    _apply_radix_cache_patches(module, index_url)
    return module


# ===========================================================================
# Tests
# ===========================================================================

class TestTreeNodePatch:
    """Verify that TreeNode gets request_ids after patching."""

    def test_new_tree_node_has_request_ids(self):
        _build_patched_module()
        node = MockTreeNode()
        assert hasattr(node, "request_ids")
        assert isinstance(node.request_ids, set)
        assert len(node.request_ids) == 0

    def test_request_ids_are_independent(self):
        _build_patched_module()
        a = MockTreeNode()
        b = MockTreeNode()
        a.request_ids.add("r1")
        assert "r1" not in b.request_ids


class TestRadixCachePatch:
    """Verify that RadixCache gets new attributes after patching."""

    def test_init_adds_tracking_state(self):
        _build_patched_module()
        cache = MockRadixCache()
        assert hasattr(cache, "_request_to_node")
        assert hasattr(cache, "eviction_callback")
        assert hasattr(cache, "_eviction_buffer")
        assert hasattr(cache, "_current_rid")
        assert cache._current_rid is None
        assert callable(cache.eviction_callback)

    def test_reset_clears_tracking(self):
        _build_patched_module()
        cache = MockRadixCache()
        cache._request_to_node["fake"] = "node"
        cache.reset()
        assert cache._request_to_node == {}

    def test_has_convenience_methods(self):
        _build_patched_module()
        cache = MockRadixCache()
        assert callable(cache.set_eviction_callback)
        assert callable(cache.get_tracked_request_ids)
        assert callable(cache.is_request_in_cache)
        assert callable(cache.get_request_node)


class TestInsertTracking:
    """Verify that insert tags new leaf nodes with request_id."""

    def test_insert_via_cache_finished_req(self):
        _build_patched_module()
        cache = MockRadixCache()
        req = MockReq(rid="req-001", fill_ids=[1, 2, 3])
        cache.cache_finished_req(req)

        assert cache.is_request_in_cache("req-001")
        assert "req-001" in cache.get_tracked_request_ids()

        leaf = cache.get_request_node("req-001")
        assert leaf is not None
        assert "req-001" in leaf.request_ids

    def test_insert_via_cache_unfinished_req(self):
        _build_patched_module()
        cache = MockRadixCache()
        req = MockReq(rid="req-002", fill_ids=[4, 5, 6])
        cache.cache_unfinished_req(req)

        assert cache.is_request_in_cache("req-002")
        leaf = cache.get_request_node("req-002")
        assert "req-002" in leaf.request_ids

    def test_full_cache_match_not_tracked(self):
        """If a request fully matches existing cache, no new tracking needed."""
        _build_patched_module()
        cache = MockRadixCache()

        # First request creates the cache entry
        req1 = MockReq(rid="req-A", fill_ids=[1, 2, 3])
        cache.cache_finished_req(req1)

        # Second request with identical tokens — full match, no new node
        req2 = MockReq(rid="req-B", fill_ids=[1, 2, 3])
        cache.cache_finished_req(req2)

        assert cache.is_request_in_cache("req-A")
        assert not cache.is_request_in_cache("req-B")  # full match → not tracked separately

    def test_partial_match_creates_new_leaf(self):
        """Request sharing a prefix but with extra tokens gets its own tracking."""
        _build_patched_module()
        cache = MockRadixCache()

        req1 = MockReq(rid="req-X", fill_ids=[1, 2, 3])
        cache.cache_finished_req(req1)

        req2 = MockReq(rid="req-Y", fill_ids=[1, 2, 3, 4, 5])
        cache.cache_finished_req(req2)

        assert cache.is_request_in_cache("req-X")
        assert cache.is_request_in_cache("req-Y")

        # They should be on different nodes
        node_x = cache.get_request_node("req-X")
        node_y = cache.get_request_node("req-Y")
        assert node_x is not node_y

    def test_divergent_requests(self):
        """Two requests diverging after a shared prefix."""
        _build_patched_module()
        cache = MockRadixCache()

        req1 = MockReq(rid="r1", fill_ids=[1, 2, 3])
        req2 = MockReq(rid="r2", fill_ids=[1, 2, 4])
        cache.cache_finished_req(req1)
        cache.cache_finished_req(req2)

        assert cache.is_request_in_cache("r1")
        assert cache.is_request_in_cache("r2")

        node1 = cache.get_request_node("r1")
        node2 = cache.get_request_node("r2")
        assert node1 is not node2


class TestEvictionTracking:
    """Verify that eviction correctly collects and reports request_ids."""

    def test_evict_fires_callback(self):
        _build_patched_module()
        cache = MockRadixCache()
        evicted = []
        cache.set_eviction_callback(lambda ids: evicted.append(ids.copy()))

        req = MockReq(rid="req-evict", fill_ids=[10, 20, 30])
        cache.cache_finished_req(req)
        assert cache.evictable_size_ == 3

        cache.evict(3)

        assert len(evicted) == 1
        assert "req-evict" in evicted[0]
        assert not cache.is_request_in_cache("req-evict")

    def test_evict_multiple_requests(self):
        _build_patched_module()
        cache = MockRadixCache()
        evicted = []
        cache.set_eviction_callback(lambda ids: evicted.append(ids.copy()))

        for i in range(5):
            req = MockReq(rid=f"req-{i}", fill_ids=[100 + i, 200 + i, 300 + i])
            cache.cache_finished_req(req)

        # Evict enough tokens to remove all
        cache.evict(999)

        all_evicted = set()
        for batch in evicted:
            all_evicted.update(batch)

        for i in range(5):
            assert f"req-{i}" in all_evicted

    def test_partial_eviction(self):
        """Only the evicted requests should be reported, not retained ones."""
        _build_patched_module()
        cache = MockRadixCache()
        evicted = []
        cache.set_eviction_callback(lambda ids: evicted.append(ids.copy()))

        # Two disjoint requests
        req1 = MockReq(rid="keep", fill_ids=[1, 2, 3])
        req2 = MockReq(rid="evict-me", fill_ids=[4, 5, 6])
        cache.cache_finished_req(req1)
        cache.cache_finished_req(req2)

        # Evict only 3 tokens (one leaf)
        cache.evict(3)

        all_evicted = set()
        for batch in evicted:
            all_evicted.update(batch)

        # At least one should be evicted, one should remain
        assert len(all_evicted) == 1
        remaining = cache.get_tracked_request_ids()
        assert len(remaining) == 1
        # The one evicted should not be in remaining
        assert all_evicted.isdisjoint(remaining)

    def test_no_callback_if_none(self):
        """If callback is None, eviction should still work without error."""
        _build_patched_module()
        cache = MockRadixCache()
        cache.set_eviction_callback(None)

        req = MockReq(rid="r", fill_ids=[1, 2])
        cache.cache_finished_req(req)
        cache.evict(2)  # Should not raise

    def test_callback_exception_is_caught(self):
        """If callback raises, it should not crash the eviction."""
        _build_patched_module()
        cache = MockRadixCache()
        cache.set_eviction_callback(lambda ids: 1 / 0)  # ZeroDivisionError

        req = MockReq(rid="r", fill_ids=[1, 2])
        cache.cache_finished_req(req)
        cache.evict(2)  # Should not raise


class TestSplitPreservesTracking:
    """Verify that node splits don't lose request_id tracking."""

    def test_split_keeps_request_on_child(self):
        _build_patched_module()
        cache = MockRadixCache()

        # Insert [1,2,3,4] → creates one leaf
        req1 = MockReq(rid="original", fill_ids=[1, 2, 3, 4])
        cache.cache_finished_req(req1)

        node_before = cache.get_request_node("original")
        assert node_before is not None

        # Insert [1,2,5,6] → splits at position 2, creating:
        #   root -> [1,2] -> [3,4] (original's node)
        #                 -> [5,6] (new node)
        req2 = MockReq(rid="divergent", fill_ids=[1, 2, 5, 6])
        cache.cache_finished_req(req2)

        # "original" should still be tracked
        assert cache.is_request_in_cache("original")
        # Its node should have the remaining [3,4] key
        node_after = cache.get_request_node("original")
        assert "original" in node_after.request_ids

        # "divergent" should also be tracked on a different node
        assert cache.is_request_in_cache("divergent")
        assert cache.get_request_node("divergent") is not node_after


class TestSourcePatchDetection:
    """Verify that monkey-patch is skipped when source patch is detected."""

    def test_skips_if_already_patched(self):
        from contextpilot._sglang_hook import _apply_radix_cache_patches

        # Simulate source-patched TreeNode (already has request_ids)
        class PatchedTreeNode:
            counter = 0
            def __init__(self, priority=0):
                self.children = {}
                self.parent = None
                self.key = None
                self.value = None
                self.lock_ref = 0
                self.last_access_time = 0
                self.priority = priority
                self.request_ids = set()  # <-- source patch already adds this
                self.id = PatchedTreeNode.counter
                PatchedTreeNode.counter += 1

        # Completely independent RadixCache class (no inheritance from MockRadixCache)
        class FreshRadixCache:
            _NOT_patched_marker = True  # just a marker to verify identity

        module = types.ModuleType("fake_radix_cache")
        module.TreeNode = PatchedTreeNode
        module.RadixCache = FreshRadixCache

        _apply_radix_cache_patches(module, "http://test:8765")

        # _contextpilot_patched should NOT be set (detection skipped patching)
        assert not hasattr(FreshRadixCache, "_contextpilot_patched")


class TestPublicAPI:
    """Test the patch_sglang() public API."""

    def test_patch_sglang_no_url_raises(self):
        from contextpilot._sglang_hook import patch_sglang

        with patch.dict(os.environ, {}, clear=True):
            # Remove CONTEXTPILOT_INDEX_URL from module-level var
            import contextpilot._sglang_hook as hook
            original = hook.CONTEXTPILOT_INDEX_URL
            hook.CONTEXTPILOT_INDEX_URL = None
            try:
                with pytest.raises(ValueError, match="No index URL"):
                    patch_sglang()
            finally:
                hook.CONTEXTPILOT_INDEX_URL = original


class TestFindLeaf:
    """Test the _find_leaf helper directly."""

    def test_find_leaf_simple(self):
        from contextpilot._sglang_hook import _find_leaf

        cache = MockRadixCache()
        key = MockRadixKey([1, 2, 3])
        cache.insert(key)

        leaf = _find_leaf(cache, MockRadixKey([1, 2, 3]))
        assert leaf is not cache.root_node
        assert leaf.key.token_ids == [1, 2, 3]

    def test_find_leaf_after_split(self):
        from contextpilot._sglang_hook import _find_leaf

        cache = MockRadixCache()
        cache.insert(MockRadixKey([1, 2, 3, 4]))
        cache.insert(MockRadixKey([1, 2, 5, 6]))

        leaf1 = _find_leaf(cache, MockRadixKey([1, 2, 3, 4]))
        assert leaf1.key.token_ids == [3, 4]

        leaf2 = _find_leaf(cache, MockRadixKey([1, 2, 5, 6]))
        assert leaf2.key.token_ids == [5, 6]

    def test_find_leaf_empty_key(self):
        from contextpilot._sglang_hook import _find_leaf

        cache = MockRadixCache()
        result = _find_leaf(cache, MockRadixKey([]))
        assert result is cache.root_node


class TestEndToEnd:
    """Full insert → evict cycle."""

    def test_insert_evict_cycle(self):
        _build_patched_module()
        cache = MockRadixCache()

        evicted_batches = []
        cache.set_eviction_callback(lambda ids: evicted_batches.append(ids.copy()))

        # Insert 3 requests with different prefixes
        for i in range(3):
            req = MockReq(rid=f"r{i}", fill_ids=[i * 10 + 1, i * 10 + 2, i * 10 + 3])
            cache.cache_finished_req(req)

        assert len(cache.get_tracked_request_ids()) == 3
        assert cache.evictable_size_ == 9  # 3 requests × 3 tokens

        # Evict 6 tokens (should remove 2 requests)
        cache.evict(6)

        all_evicted = set()
        for batch in evicted_batches:
            all_evicted.update(batch)

        assert len(all_evicted) == 2
        assert len(cache.get_tracked_request_ids()) == 1

        # The remaining request should still be findable
        remaining = cache.get_tracked_request_ids()
        remaining_id = remaining.pop()
        assert cache.is_request_in_cache(remaining_id)
        assert cache.get_request_node(remaining_id) is not None

    def test_shared_prefix_eviction(self):
        """Requests sharing prefix: evicting the unique-token leaf triggers callback."""
        _build_patched_module()
        cache = MockRadixCache()

        evicted_ids = set()
        cache.set_eviction_callback(lambda ids: evicted_ids.update(ids))

        # req-A: [1,2,3]
        # req-B: [1,2,3,4,5]  (shares prefix with A, has extra [4,5])
        req_a = MockReq(rid="A", fill_ids=[1, 2, 3])
        req_b = MockReq(rid="B", fill_ids=[1, 2, 3, 4, 5])
        cache.cache_finished_req(req_a)
        cache.cache_finished_req(req_b)

        # Only [4,5] is a leaf (the extra tokens from B)
        # [1,2,3] is a shared prefix (non-leaf, has child)
        # Evicting 2 tokens should remove the [4,5] leaf
        cache.evict(2)

        assert "B" in evicted_ids
        assert "A" not in evicted_ids  # A's node is a parent, not evictable
