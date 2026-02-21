# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ContextPilot patch for vLLM v0.15.1 block_pool.py
# Adds eviction callback support for KV cache synchronization.
#
# Changes from vanilla vLLM:
#   - Added _block_to_requests / _request_to_blocks tracking dicts
#   - Added eviction_callback (auto-created from CONTEXTPILOT_INDEX_URL env var)
#   - Modified cache_full_blocks() to record request_id -> block_hash mappings
#   - Modified _maybe_evict_cached_block() to return evicted request_ids
#   - Modified get_new_blocks() to batch and fire eviction callback
#   - Modified evict_blocks() and reset_prefix_cache() to fire callback

import os
import re
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Optional

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)

CONTEXTPILOT_INDEX_URL = os.environ.get("CONTEXTPILOT_INDEX_URL")
_contextpilot_enabled = CONTEXTPILOT_INDEX_URL is not None
_TRACK_ONLY_PREFIX = os.environ.get("CONTEXTPILOT_TRACK_ONLY_PREFIX", "req-")

# Strip vLLM ID prefixes (cmpl-, chatcmpl-, batch-) and suffixes (req-<base>-N-<hex>)
_VLLM_REQUEST_ID_PREFIX = re.compile(r"^(cmpl-|chatcmpl-|batch-)")
_VLLM_REQUEST_ID_SUFFIX = re.compile(r"^(req-[^-]+)-\d+-[0-9a-f]+$")

EvictionCallback = Optional[Callable[[set], None]]


def create_contextpilot_eviction_callback() -> EvictionCallback:
    """Create eviction callback if CONTEXTPILOT_INDEX_URL is set, else None."""
    if not _contextpilot_enabled:
        return None

    import requests as http_requests

    def eviction_callback(evicted_request_ids: set):
        if not evicted_request_ids:
            return

        filtered_ids = {
            rid for rid in (
                _normalize_request_id(r) for r in evicted_request_ids
            )
            if rid and not rid.startswith("HEALTH_CHECK")
        }

        if not filtered_ids:
            return

        try:
            preview_ids = sorted(filtered_ids)
            preview = ", ".join(preview_ids[:10])
            suffix = "" if len(preview_ids) <= 10 else ", ..."
            logger.info(
                "[ContextPilot] Syncing eviction: %d requests [%s%s]",
                len(filtered_ids),
                preview,
                suffix,
            )
            http_requests.post(
                f"{CONTEXTPILOT_INDEX_URL}/evict",
                json={"request_ids": list(filtered_ids)},
                timeout=1.0,
            )
        except Exception as e:
            logger.warning("ContextPilot eviction sync failed: %s", e)

    return eviction_callback


def _should_track_request_id(request_id: str) -> bool:
    if not request_id:
        return False
    stripped = _normalize_request_id(request_id)
    if not stripped or stripped.startswith("HEALTH_CHECK"):
        return False
    # CONTEXTPILOT_TRACK_ONLY_PREFIX="" tracks all request IDs.
    if _TRACK_ONLY_PREFIX:
        return stripped.startswith(_TRACK_ONLY_PREFIX)
    return True


def _normalize_request_id(request_id: str) -> str:
    """Normalize vLLM request IDs to ContextPilot canonical form."""
    rid = _VLLM_REQUEST_ID_PREFIX.sub("", request_id)
    m = _VLLM_REQUEST_ID_SUFFIX.match(rid)
    if m:
        return m.group(1)
    return rid


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

        # ContextPilot eviction tracking
        self._block_to_requests: dict[BlockHashWithGroupId, set[str]] = {}
        self._request_to_blocks: dict[str, set[BlockHashWithGroupId]] = {}
        self.eviction_callback: EvictionCallback = (
            create_contextpilot_eviction_callback()
        )

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null blocks when enabling sparse attention like
            # sliding window attention, or Mamba models with prefix-caching in
            # align mode. We skip null blocks here.
            if blk.is_null:
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)

            # ContextPilot: track request -> block ownership
            if self.eviction_callback is not None:
                req_id = request.request_id
                if _should_track_request_id(req_id):
                    req_id = _normalize_request_id(req_id)
                    self._block_to_requests.setdefault(
                        block_hash_with_group_id, set()
                    ).add(req_id)
                    self._request_to_blocks.setdefault(
                        req_id, set()
                    ).add(block_hash_with_group_id)

            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[
                        num_cached_blocks
                        * block_size : num_full_blocks
                        * block_size
                    ],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                )
            )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool"
            )

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        fully_evicted: set[str] = set()  # ContextPilot

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                evicted = self._maybe_evict_cached_block(block)
                fully_evicted.update(evicted)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)

        if fully_evicted and self.eviction_callback is not None:
            try:
                self.eviction_callback(fully_evicted)
            except Exception as e:
                logger.warning("Eviction callback failed: %s", e)

        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> set[str]:
        """Evict block from cache; return request_ids fully evicted."""
        fully_evicted: set[str] = set()

        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return fully_evicted

        if self.cached_block_hash_to_block.pop(
            block_hash, block.block_id
        ) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return fully_evicted

        # Only drop ownership when the last copy of this hash is evicted.
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

        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[
                        maybe_convert_block_hash(get_block_hash(block_hash))
                    ],
                    medium=MEDIUM_GPU,
                )
            )
        return fully_evicted

    def touch(
        self,
        blocks: Sequence[KVCacheBlock] | Sequence[Sequence[KVCacheBlock]],
    ) -> None:
        """Touch blocks to increase their reference counts.

        vLLM calls this with grouped blocks (e.g. one sequence per KV group).
        Accept a flat sequence as well for backwards compatibility.

        Args:
            blocks: Either a flat sequence of blocks or a sequence of block
                sequences.
        """
        if not blocks:
            return

        # vLLM uses grouped blocks: tuple[Sequence[KVCacheBlock], ...].
        if isinstance(blocks[0], KVCacheBlock):
            block_iter: Iterable[KVCacheBlock] = blocks  # type: ignore[assignment]
        else:
            block_iter = (b for group in blocks for b in group)  # type: ignore[misc]

        for block in block_iter:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [
                block
                for block in blocks_list
                if block.ref_cnt == 0 and not block.is_null
            ]
        )

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        fully_evicted: set[str] = set()

        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            evicted = self._maybe_evict_cached_block(block)
            fully_evicted.update(evicted)

        if fully_evicted and self.eviction_callback is not None:
            try:
                self.eviction_callback(fully_evicted)
            except Exception as e:
                logger.warning("Eviction callback failed: %s", e)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        if self._request_to_blocks and self.eviction_callback is not None:
            all_requests = set(self._request_to_blocks.keys())
            try:
                self.eviction_callback(all_requests)
            except Exception as e:
                logger.warning("Eviction callback failed: %s", e)
        self._block_to_requests.clear()
        self._request_to_blocks.clear()

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events

    # --- ContextPilot helper methods ---

    def get_tracked_request_ids(self) -> set[str]:
        """Return the set of all request IDs currently tracked in the cache."""
        return set(self._request_to_blocks.keys())

    def is_request_in_cache(self, request_id: str) -> bool:
        """Check if a request still has cached blocks."""
        return request_id in self._request_to_blocks

    def set_eviction_callback(self, callback: EvictionCallback) -> None:
        """Set or replace the eviction callback at runtime."""
        self.eviction_callback = callback
