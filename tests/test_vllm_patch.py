"""
Tests for vLLM block_pool.py eviction sync patch.

Tests the ContextPilot tracking dicts and eviction callback logic
without requiring a vLLM installation — all vLLM internals are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Mock vLLM types so we can import/test block_pool logic without vLLM
# ---------------------------------------------------------------------------

@dataclass
class MockKVCacheBlock:
    block_id: int
    block_hash: Optional[bytes] = None
    ref_cnt: int = 0
    is_null: bool = False
    prev_free_block: Optional["MockKVCacheBlock"] = field(default=None, repr=False)
    next_free_block: Optional["MockKVCacheBlock"] = field(default=None, repr=False)

    def reset_hash(self):
        self.block_hash = None


class MockFreeKVCacheBlockQueue:
    """Simplified free block queue for testing."""

    def __init__(self, blocks):
        self._blocks = list(blocks)
        self.num_free_blocks = len(self._blocks)

    def popleft(self):
        self.num_free_blocks -= 1
        return self._blocks.pop(0)

    def popleft_n(self, n):
        result = self._blocks[:n]
        self._blocks = self._blocks[n:]
        self.num_free_blocks -= n
        return result

    def remove(self, block):
        if block in self._blocks:
            self._blocks.remove(block)
            self.num_free_blocks -= 1

    def append_n(self, blocks):
        self._blocks.extend(blocks)
        self.num_free_blocks += len(blocks)


class MockRequest:
    def __init__(self, request_id, block_hashes, all_token_ids=None):
        self.request_id = request_id
        self.block_hashes = block_hashes
        self.all_token_ids = all_token_ids or []
        self.lora_request = None


class MockBlockHashToBlockMap:
    """Mirrors the real BlockHashToBlockMap for testing."""

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
        raise AssertionError(f"Invalid cache block type: {type(blocks)}")

    def insert(self, key, block):
        blocks = self._cache.get(key)
        if blocks is None:
            self._cache[key] = block
        elif isinstance(blocks, MockKVCacheBlock):
            self._cache[key] = {
                blocks.block_id: blocks,
                block.block_id: block,
            }
        elif isinstance(blocks, dict):
            blocks[block.block_id] = block
        else:
            raise AssertionError(f"Invalid cache block type: {type(blocks)}")

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

        self._cache[key] = blocks
        return None

    def __len__(self):
        return len(self._cache)


# ---------------------------------------------------------------------------
# BlockPool under test — extracted logic (no vLLM imports needed)
# ---------------------------------------------------------------------------

class TestableBlockPool:
    """ContextPilot-patched BlockPool logic, mocked for testing."""

    def __init__(self, num_blocks=10, eviction_callback=None):
        self.blocks = [MockKVCacheBlock(i) for i in range(num_blocks)]
        self.free_block_queue = MockFreeKVCacheBlockQueue(list(self.blocks))
        self.cached_block_hash_to_block = MockBlockHashToBlockMap()
        self.enable_caching = True
        self.num_gpu_blocks = num_blocks
        self.metrics_collector = None

        # Null block
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        # ContextPilot tracking
        self._block_to_requests: dict[bytes, set[str]] = {}
        self._request_to_blocks: dict[str, set[bytes]] = {}
        self.eviction_callback = eviction_callback

    def cache_full_blocks_simple(self, request_id, block_indices, block_hashes):
        for idx, bh in zip(block_indices, block_hashes):
            blk = self.blocks[idx]
            blk.block_hash = bh
            self.cached_block_hash_to_block.insert(bh, blk)

            if self.eviction_callback is not None:
                self._block_to_requests.setdefault(bh, set()).add(request_id)
                self._request_to_blocks.setdefault(request_id, set()).add(bh)

    def _maybe_evict_cached_block(self, block) -> set:
        fully_evicted = set()
        block_hash = block.block_hash
        if block_hash is None:
            return fully_evicted

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            return fully_evicted

        if self.cached_block_hash_to_block.get_one_block(block_hash) is None:
            request_ids = self._block_to_requests.pop(block_hash, None)
            if request_ids:
                for rid in request_ids:
                    blocks_set = self._request_to_blocks.get(rid)
                    if blocks_set is not None:
                        blocks_set.discard(block_hash)
                        if not blocks_set:
                            fully_evicted.add(rid)
                            del self._request_to_blocks[rid]

        block.reset_hash()
        return fully_evicted

    def get_new_blocks(self, num_blocks):
        ret = self.free_block_queue.popleft_n(num_blocks)
        fully_evicted = set()

        if self.enable_caching:
            for block in ret:
                evicted = self._maybe_evict_cached_block(block)
                fully_evicted.update(evicted)
                block.ref_cnt += 1
        else:
            for block in ret:
                block.ref_cnt += 1

        if fully_evicted and self.eviction_callback is not None:
            try:
                self.eviction_callback(fully_evicted)
            except Exception:
                pass

        return ret

    def free_blocks(self, blocks):
        blocks_list = list(blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [b for b in blocks_list if b.ref_cnt == 0 and not b.is_null]
        )

    def touch(self, blocks):
        if not blocks:
            return
        if isinstance(blocks[0], MockKVCacheBlock):
            block_iter = blocks
        else:
            block_iter = (b for group in blocks for b in group)

        for block in block_iter:
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    def evict_blocks(self, block_ids):
        fully_evicted = set()
        for block_id in block_ids:
            block = self.blocks[block_id]
            evicted = self._maybe_evict_cached_block(block)
            fully_evicted.update(evicted)
        if fully_evicted and self.eviction_callback is not None:
            try:
                self.eviction_callback(fully_evicted)
            except Exception:
                pass

    def reset_prefix_cache(self):
        if self._request_to_blocks and self.eviction_callback is not None:
            all_requests = set(self._request_to_blocks.keys())
            try:
                self.eviction_callback(all_requests)
            except Exception:
                pass
        self._block_to_requests.clear()
        self._request_to_blocks.clear()
        self.cached_block_hash_to_block = MockBlockHashToBlockMap()
        for block in self.blocks:
            block.reset_hash()

    def get_tracked_request_ids(self):
        return set(self._request_to_blocks.keys())

    def is_request_in_cache(self, request_id):
        return request_id in self._request_to_blocks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrackingDicts:

    def test_cache_records_mapping(self):
        callback = MagicMock()
        pool = TestableBlockPool(eviction_callback=callback)

        pool.cache_full_blocks_simple(
            "req-1", [1, 2, 3], [b"h1", b"h2", b"h3"]
        )

        assert pool.is_request_in_cache("req-1")
        assert pool._request_to_blocks["req-1"] == {b"h1", b"h2", b"h3"}
        assert "req-1" in pool._block_to_requests[b"h1"]
        assert "req-1" in pool._block_to_requests[b"h2"]
        assert "req-1" in pool._block_to_requests[b"h3"]

    def test_shared_blocks_track_both_requests(self):
        callback = MagicMock()
        pool = TestableBlockPool(num_blocks=20, eviction_callback=callback)

        # Two requests share block hash h1 (different block_ids, same hash)
        pool.cache_full_blocks_simple("req-A", [1, 2], [b"h1", b"h2"])
        pool.cache_full_blocks_simple("req-B", [3, 4], [b"h1", b"h3"])

        assert pool._block_to_requests[b"h1"] == {"req-A", "req-B"}
        assert pool._request_to_blocks["req-A"] == {b"h1", b"h2"}
        assert pool._request_to_blocks["req-B"] == {b"h1", b"h3"}

    def test_no_tracking_when_callback_is_none(self):
        pool = TestableBlockPool(eviction_callback=None)

        pool.cache_full_blocks_simple("req-1", [1, 2], [b"h1", b"h2"])

        assert len(pool._block_to_requests) == 0
        assert len(pool._request_to_blocks) == 0


class TestEvictionCallback:

    def test_full_eviction_fires_callback(self):
        callback = MagicMock()
        pool = TestableBlockPool(num_blocks=10, eviction_callback=callback)

        # Cache 3 blocks for req-1 using blocks 1,2,3
        pool.cache_full_blocks_simple("req-1", [1, 2, 3], [b"h1", b"h2", b"h3"])

        # Evict all 3 blocks
        pool.evict_blocks({1, 2, 3})

        callback.assert_called_once()
        evicted_ids = callback.call_args[0][0]
        assert "req-1" in evicted_ids

    def test_partial_eviction_does_not_fire_callback(self):
        callback = MagicMock()
        pool = TestableBlockPool(num_blocks=10, eviction_callback=callback)

        pool.cache_full_blocks_simple("req-1", [1, 2, 3], [b"h1", b"h2", b"h3"])

        # Evict only 2 of 3 blocks — request still has h3
        pool.evict_blocks({1, 2})

        callback.assert_not_called()
        assert pool.is_request_in_cache("req-1")
        assert pool._request_to_blocks["req-1"] == {b"h3"}

    def test_evict_last_block_fires_callback(self):
        callback = MagicMock()
        pool = TestableBlockPool(num_blocks=10, eviction_callback=callback)

        pool.cache_full_blocks_simple("req-1", [1, 2, 3], [b"h1", b"h2", b"h3"])

        # Evict 2, then the last 1
        pool.evict_blocks({1, 2})
        callback.assert_not_called()

        pool.evict_blocks({3})
        callback.assert_called_once()
        assert "req-1" in callback.call_args[0][0]
        assert not pool.is_request_in_cache("req-1")

    def test_shared_hash_not_evicted_until_last_copy_removed(self):
        callback = MagicMock()
        pool = TestableBlockPool(num_blocks=12, eviction_callback=callback)

        # req-A: shared + unique, req-B: shared only
        pool.cache_full_blocks_simple("req-A", [1, 2], [b"h_shared", b"h_a"])
        pool.cache_full_blocks_simple("req-B", [3], [b"h_shared"])

        # Remove one shared copy + req-A unique block.
        # h_shared is still available via req-B's block.
        pool.evict_blocks({1, 2})

        callback.assert_not_called()
        assert pool.is_request_in_cache("req-A")
        assert pool.is_request_in_cache("req-B")

        # Remove final shared copy: now both requests are fully evicted.
        pool.evict_blocks({3})
        callback.assert_called_once()
        assert callback.call_args[0][0] == {"req-A", "req-B"}

    def test_multiple_requests_evicted_together(self):
        callback = MagicMock()
        pool = TestableBlockPool(num_blocks=10, eviction_callback=callback)

        pool.cache_full_blocks_simple("req-A", [1], [b"hA"])
        pool.cache_full_blocks_simple("req-B", [2], [b"hB"])

        pool.evict_blocks({1, 2})

        callback.assert_called_once()
        evicted = callback.call_args[0][0]
        assert evicted == {"req-A", "req-B"}

    def test_callback_not_called_when_none(self):
        pool = TestableBlockPool(eviction_callback=None)

        pool.cache_full_blocks_simple("req-1", [1], [b"h1"])
        # Should not raise
        pool.evict_blocks({1})

    def test_callback_exception_is_swallowed(self):
        callback = MagicMock(side_effect=Exception("network error"))
        pool = TestableBlockPool(num_blocks=10, eviction_callback=callback)

        pool.cache_full_blocks_simple("req-1", [1], [b"h1"])
        # Should not raise even though callback throws
        pool.evict_blocks({1})
        callback.assert_called_once()


class TestGetNewBlocksEviction:

    def test_allocating_cached_blocks_fires_callback(self):
        """When get_new_blocks pops cached blocks, eviction callback fires."""
        callback = MagicMock()
        pool = TestableBlockPool(num_blocks=10, eviction_callback=callback)

        # Allocate/cache/free blocks first so they become eviction candidates.
        blocks = pool.get_new_blocks(3)
        block_ids = [b.block_id for b in blocks]
        pool.cache_full_blocks_simple("req-X", block_ids, [b"h1", b"h2", b"h3"])
        pool.free_blocks(blocks)
        assert pool.is_request_in_cache("req-X")

        # Force allocation of all free blocks to guarantee cached blocks are popped.
        pool.get_new_blocks(pool.free_block_queue.num_free_blocks)

        callback.assert_called_once()
        assert "req-X" in callback.call_args[0][0]
        assert not pool.is_request_in_cache("req-X")


class TestTouchCompatibility:

    def test_touch_accepts_grouped_blocks(self):
        pool = TestableBlockPool(num_blocks=8, eviction_callback=None)
        blocks = pool.get_new_blocks(2)
        pool.free_blocks(blocks)

        # Upstream style: tuple[Sequence[KVCacheBlock], ...]
        pool.touch((blocks,))
        assert blocks[0].ref_cnt == 1
        assert blocks[1].ref_cnt == 1


class TestResetPrefixCache:

    def test_reset_fires_callback_for_all(self):
        callback = MagicMock()
        pool = TestableBlockPool(num_blocks=10, eviction_callback=callback)

        pool.cache_full_blocks_simple("req-A", [1], [b"hA"])
        pool.cache_full_blocks_simple("req-B", [2], [b"hB"])
        pool.cache_full_blocks_simple("req-C", [3], [b"hC"])

        pool.reset_prefix_cache()

        callback.assert_called_once()
        evicted = callback.call_args[0][0]
        assert evicted == {"req-A", "req-B", "req-C"}

        # Tracking should be cleared
        assert len(pool._block_to_requests) == 0
        assert len(pool._request_to_blocks) == 0

    def test_reset_with_no_tracked_requests(self):
        callback = MagicMock()
        pool = TestableBlockPool(num_blocks=10, eviction_callback=callback)

        pool.reset_prefix_cache()

        callback.assert_not_called()


class TestCallbackPrefixStripping:

    def test_strips_cmpl_prefix(self):
        import re
        prefix_re = re.compile(r"^(cmpl-|chatcmpl-|batch-)")

        ids = {"cmpl-req-123", "chatcmpl-req-456", "batch-req-789", "plain-id"}
        stripped = {prefix_re.sub("", rid) for rid in ids}

        assert stripped == {"req-123", "req-456", "req-789", "plain-id"}

    def test_no_prefix_unchanged(self):
        import re
        prefix_re = re.compile(r"^(cmpl-|chatcmpl-|batch-)")

        ids = {"my-request-1", "another-req"}
        stripped = {prefix_re.sub("", rid) for rid in ids}

        assert stripped == {"my-request-1", "another-req"}


class TestHelperMethods:

    def test_get_tracked_request_ids(self):
        callback = MagicMock()
        pool = TestableBlockPool(num_blocks=10, eviction_callback=callback)

        pool.cache_full_blocks_simple("req-A", [1], [b"hA"])
        pool.cache_full_blocks_simple("req-B", [2], [b"hB"])

        assert pool.get_tracked_request_ids() == {"req-A", "req-B"}

    def test_is_request_in_cache(self):
        callback = MagicMock()
        pool = TestableBlockPool(num_blocks=10, eviction_callback=callback)

        pool.cache_full_blocks_simple("req-A", [1], [b"hA"])

        assert pool.is_request_in_cache("req-A")
        assert not pool.is_request_in_cache("req-B")

        pool.evict_blocks({1})
        assert not pool.is_request_in_cache("req-A")
