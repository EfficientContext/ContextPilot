"""
Tests for the vLLM monkey-patch approach (contextpilot/_vllm_hook.py).

These tests use a lightweight mock of vLLM's BlockPool to verify the
monkey-patching logic without requiring vLLM to be installed.
"""

import os
import sys
import types
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Lightweight mocks of vLLM's block pool primitives
# ---------------------------------------------------------------------------

class MockKVCacheBlock:
    """Minimal stand-in for vllm.v1.core.kv_cache_utils.KVCacheBlock."""

    def __init__(self, block_id):
        self.block_id = block_id
        self.block_hash = None
        self.ref_cnt = 0
        self.is_null = False

    def reset_hash(self):
        self.block_hash = None


class MockBlockHashToBlockMap:
    """Minimal stand-in for BlockHashToBlockMap."""

    def __init__(self):
        self._cache = {}

    def get_one_block(self, key):
        blocks = self._cache.get(key)
        if blocks is None:
            return None
        if isinstance(blocks, MockKVCacheBlock):
            return blocks
        if isinstance(blocks, dict):
            return next(iter(blocks.values()))
        return None

    def insert(self, key, block):
        existing = self._cache.get(key)
        if existing is None:
            self._cache[key] = block
        elif isinstance(existing, MockKVCacheBlock):
            self._cache[key] = {existing.block_id: existing, block.block_id: block}
        elif isinstance(existing, dict):
            existing[block.block_id] = block

    def pop(self, key, block_id):
        blocks = self._cache.pop(key, None)
        if blocks is None:
            return None
        if isinstance(blocks, MockKVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            block = blocks.pop(block_id, None)
            if blocks:
                self._cache[key] = blocks
            return block
        return None

    def __len__(self):
        return len(self._cache)


class MockFreeBlockQueue:
    """Minimal stand-in for FreeKVCacheBlockQueue."""

    def __init__(self, blocks):
        self._queue = list(blocks)
        self.num_free_blocks = len(self._queue)

    def popleft(self):
        block = self._queue.pop(0)
        self.num_free_blocks -= 1
        return block

    def popleft_n(self, n):
        result = self._queue[:n]
        self._queue = self._queue[n:]
        self.num_free_blocks -= n
        return result

    def append_n(self, blocks):
        self._queue.extend(blocks)
        self.num_free_blocks += len(blocks)


class MockRequest:
    """Minimal stand-in for vllm.v1.request.Request."""

    def __init__(self, request_id, block_hashes):
        self.request_id = request_id
        self.block_hashes = block_hashes
        self.all_token_ids = []
        self.lora_request = None


class MockBlockPool:
    """Minimal stand-in for vllm.v1.core.block_pool.BlockPool."""

    def __init__(self, num_gpu_blocks=100, enable_caching=True,
                 hash_block_size=16):
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        self.blocks = [MockKVCacheBlock(i) for i in range(num_gpu_blocks)]
        self.free_block_queue = MockFreeBlockQueue(list(self.blocks))
        self.cached_block_hash_to_block = MockBlockHashToBlockMap()
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
        self.enable_kv_cache_events = False
        self.kv_event_queue = []
        self.metrics_collector = None

    def cache_full_blocks(self, request, blocks, num_cached_blocks,
                          num_full_blocks, block_size, kv_cache_group_id):
        if num_cached_blocks >= num_full_blocks:
            return
        for i, blk in enumerate(blocks[num_cached_blocks:num_full_blocks]):
            if blk.is_null:
                continue
            bh = (request.block_hashes[num_cached_blocks + i], kv_cache_group_id)
            blk.block_hash = bh
            self.cached_block_hash_to_block.insert(bh, blk)

    def _maybe_evict_cached_block(self, block):
        if self.metrics_collector:
            pass
        block_hash = block.block_hash
        if block_hash is None:
            return
        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            return
        block.reset_hash()

    def get_new_blocks(self, num_blocks):
        if num_blocks > self.free_block_queue.num_free_blocks:
            raise ValueError(f"Cannot get {num_blocks} free blocks")
        ret = self.free_block_queue.popleft_n(num_blocks)
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                block.ref_cnt += 1
        else:
            for block in ret:
                block.ref_cnt += 1
        return ret

    def evict_blocks(self, block_ids):
        for block_id in block_ids:
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self):
        num_used = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used != 1:
            return False
        self.cached_block_hash_to_block = MockBlockHashToBlockMap()
        for block in self.blocks:
            block.reset_hash()
        return True

    def get_num_free_blocks(self):
        return self.free_block_queue.num_free_blocks

    def free_blocks(self, ordered_blocks):
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [b for b in blocks_list if b.ref_cnt == 0 and not b.is_null]
        )


# ---------------------------------------------------------------------------
# Helper: build a fake module and apply the monkey-patch
# ---------------------------------------------------------------------------

def _build_patched_module(index_url="http://test:8765"):
    """Create a fake vllm.v1.core.block_pool module, patch it, return it."""
    from contextpilot._vllm_hook import _apply_block_pool_patches

    module = types.ModuleType("vllm.v1.core.block_pool")
    module.BlockPool = MockBlockPool

    _apply_block_pool_patches(module, index_url)
    return module


def _make_blocks(pool, count):
    """Allocate blocks from the pool and return them."""
    return pool.get_new_blocks(count)


def _cache_request(pool, request_id, block_hashes, group_id=0):
    """Simulate caching a request's blocks."""
    n = len(block_hashes)
    blocks = _make_blocks(pool, n)
    req = MockRequest(request_id=request_id, block_hashes=block_hashes)
    pool.cache_full_blocks(req, blocks, 0, n, pool.hash_block_size, group_id)
    # Release the blocks so they're eviction candidates
    pool.free_blocks(blocks)
    return blocks


# ===========================================================================
# Tests
# ===========================================================================

class TestBlockPoolPatch:
    """Verify that BlockPool gets new attributes after patching."""

    def test_init_adds_tracking_state(self):
        _build_patched_module()
        pool = MockBlockPool()
        assert hasattr(pool, "_block_to_requests")
        assert hasattr(pool, "_request_to_blocks")
        assert hasattr(pool, "_eviction_buffer")
        assert hasattr(pool, "eviction_callback")
        assert callable(pool.eviction_callback)

    def test_has_convenience_methods(self):
        _build_patched_module()
        pool = MockBlockPool()
        assert callable(pool.get_tracked_request_ids)
        assert callable(pool.is_request_in_cache)
        assert callable(pool.set_eviction_callback)


class TestCacheTracking:
    """Verify that cache_full_blocks tracks request -> block ownership."""

    def test_cache_tracks_request(self):
        _build_patched_module()
        pool = MockBlockPool()
        _cache_request(pool, "req-001", ["hash_a", "hash_b"])

        assert pool.is_request_in_cache("req-001")
        assert "req-001" in pool.get_tracked_request_ids()

    def test_multiple_requests_tracked(self):
        _build_patched_module()
        pool = MockBlockPool()
        _cache_request(pool, "req-001", ["hash_a", "hash_b"])
        _cache_request(pool, "req-002", ["hash_c", "hash_d"])

        assert pool.is_request_in_cache("req-001")
        assert pool.is_request_in_cache("req-002")
        assert len(pool.get_tracked_request_ids()) == 2

    def test_untracked_request_id_ignored(self):
        """Request IDs that don't pass the filter are not tracked."""
        _build_patched_module()
        pool = MockBlockPool()
        # "HEALTH_CHECK-123" should be filtered out
        _cache_request(pool, "HEALTH_CHECK-123", ["hash_a"])

        assert not pool.is_request_in_cache("HEALTH_CHECK-123")


class TestEvictionViaGetNewBlocks:
    """Verify eviction tracking through get_new_blocks path."""

    def test_eviction_fires_callback(self):
        _build_patched_module()
        pool = MockBlockPool(num_gpu_blocks=10)
        evicted = []
        pool.set_eviction_callback(lambda ids: evicted.append(ids.copy()))

        # Cache a request using 3 blocks
        _cache_request(pool, "req-evict", ["h1", "h2", "h3"])
        assert pool.is_request_in_cache("req-evict")

        # Allocate enough blocks to force eviction of cached ones
        pool.get_new_blocks(pool.get_num_free_blocks())

        all_evicted = set()
        for batch in evicted:
            all_evicted.update(batch)

        assert "req-evict" in all_evicted
        assert not pool.is_request_in_cache("req-evict")

    def test_partial_eviction(self):
        """Only fully evicted requests should be reported."""
        _build_patched_module()
        pool = MockBlockPool(num_gpu_blocks=20)
        evicted = []
        pool.set_eviction_callback(lambda ids: evicted.append(ids.copy()))

        # Cache two requests with different hashes
        _cache_request(pool, "req-A", ["h_a1", "h_a2"])
        _cache_request(pool, "req-B", ["h_b1", "h_b2"])

        # Evict just enough for one request (2 blocks)
        pool.get_new_blocks(2)

        all_evicted = set()
        for batch in evicted:
            all_evicted.update(batch)

        # At most one request should be fully evicted
        assert len(all_evicted) <= 1


class TestEvictionViaEvictBlocks:
    """Verify eviction tracking through evict_blocks path."""

    def test_evict_blocks_fires_callback(self):
        _build_patched_module()
        pool = MockBlockPool(num_gpu_blocks=10)
        evicted = []
        pool.set_eviction_callback(lambda ids: evicted.append(ids.copy()))

        blocks = _cache_request(pool, "req-target", ["h1", "h2"])

        # Evict specific blocks by ID
        pool.evict_blocks({b.block_id for b in blocks})

        all_evicted = set()
        for batch in evicted:
            all_evicted.update(batch)

        assert "req-target" in all_evicted


class TestResetPrefixCache:
    """Verify eviction tracking through reset_prefix_cache path."""

    def test_reset_fires_callback_for_all(self):
        _build_patched_module()
        pool = MockBlockPool(num_gpu_blocks=10)
        evicted = []
        pool.set_eviction_callback(lambda ids: evicted.append(ids.copy()))

        _cache_request(pool, "req-1", ["h1"])
        _cache_request(pool, "req-2", ["h2"])

        result = pool.reset_prefix_cache()
        assert result is True

        all_evicted = set()
        for batch in evicted:
            all_evicted.update(batch)

        assert "req-1" in all_evicted
        assert "req-2" in all_evicted


class TestSharedBlocks:
    """Verify behavior when multiple requests share cached blocks."""

    def test_shared_block_eviction(self):
        """Shared block: both requests should be evicted when block is evicted."""
        _build_patched_module()
        pool = MockBlockPool(num_gpu_blocks=10)
        evicted = []
        pool.set_eviction_callback(lambda ids: evicted.append(ids.copy()))

        # Two requests sharing the same block hash
        _cache_request(pool, "req-A", ["shared_hash"])
        _cache_request(pool, "req-B", ["shared_hash"])

        # Both should be tracked
        assert pool.is_request_in_cache("req-A")
        assert pool.is_request_in_cache("req-B")

        # Evict all blocks
        pool.get_new_blocks(pool.get_num_free_blocks())

        all_evicted = set()
        for batch in evicted:
            all_evicted.update(batch)

        assert "req-A" in all_evicted
        assert "req-B" in all_evicted


class TestCallbackEdgeCases:
    """Test callback error handling and edge cases."""

    def test_no_callback_if_none(self):
        """If callback is None, eviction should still work."""
        _build_patched_module()
        pool = MockBlockPool(num_gpu_blocks=10)
        pool.set_eviction_callback(None)

        _cache_request(pool, "req-x", ["h1"])
        pool.get_new_blocks(pool.get_num_free_blocks())  # should not raise

    def test_callback_exception_is_caught(self):
        """If callback raises, it should not crash."""
        _build_patched_module()
        pool = MockBlockPool(num_gpu_blocks=10)
        pool.set_eviction_callback(lambda ids: 1 / 0)

        _cache_request(pool, "req-x", ["h1"])
        pool.get_new_blocks(pool.get_num_free_blocks())  # should not raise


class TestSourcePatchDetection:
    """Verify that monkey-patch is skipped when source patch is detected."""

    def test_skips_if_already_has_tracking(self):
        from contextpilot._vllm_hook import _apply_block_pool_patches

        class AlreadyPatchedBlockPool:
            def get_tracked_request_ids(self):
                return set()

        module = types.ModuleType("fake_block_pool")
        module.BlockPool = AlreadyPatchedBlockPool

        _apply_block_pool_patches(module, "http://test:8765")

        # Should NOT set our marker
        assert not hasattr(AlreadyPatchedBlockPool, "_contextpilot_patched")

    def test_skips_if_marker_set(self):
        from contextpilot._vllm_hook import _apply_block_pool_patches

        class MarkedBlockPool:
            _contextpilot_patched = True

        module = types.ModuleType("fake_block_pool")
        module.BlockPool = MarkedBlockPool

        _apply_block_pool_patches(module, "http://test:8765")

        # Should remain as-is
        assert not hasattr(MarkedBlockPool, "set_eviction_callback")


class TestPublicAPI:
    """Test the patch_vllm() public API."""

    def test_patch_vllm_no_url_raises(self):
        from contextpilot._vllm_hook import patch_vllm

        with patch.dict(os.environ, {}, clear=True):
            import contextpilot._vllm_hook as hook
            original = hook.CONTEXTPILOT_INDEX_URL
            hook.CONTEXTPILOT_INDEX_URL = None
            try:
                with pytest.raises(ValueError, match="No index URL"):
                    patch_vllm()
            finally:
                hook.CONTEXTPILOT_INDEX_URL = original


class TestNormalization:
    """Test request ID normalization."""

    def test_strip_cmpl_prefix(self):
        from contextpilot._vllm_hook import _normalize_request_id
        assert _normalize_request_id("cmpl-req-123") == "req-123"

    def test_strip_chatcmpl_prefix(self):
        from contextpilot._vllm_hook import _normalize_request_id
        assert _normalize_request_id("chatcmpl-req-abc") == "req-abc"

    def test_strip_vllm_suffix(self):
        from contextpilot._vllm_hook import _normalize_request_id
        assert _normalize_request_id("req-abc-0-deadbeef") == "req-abc"

    def test_passthrough_clean_id(self):
        from contextpilot._vllm_hook import _normalize_request_id
        assert _normalize_request_id("req-myid") == "req-myid"

    def test_should_track_filters_health_check(self):
        from contextpilot._vllm_hook import _should_track_request_id
        assert not _should_track_request_id("HEALTH_CHECK-123")

    def test_should_track_empty(self):
        from contextpilot._vllm_hook import _should_track_request_id
        assert not _should_track_request_id("")
        assert not _should_track_request_id(None)
