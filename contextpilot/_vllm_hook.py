"""
vLLM Runtime Monkey-Patch for ContextPilot.

Automatically patches vLLM's BlockPool to track request-to-block
associations and report evictions to the ContextPilot index server.

Activation:
    Set the CONTEXTPILOT_INDEX_URL environment variable before starting vLLM.
    If installed via pip, a .pth file auto-imports this module on Python startup.

    CONTEXTPILOT_INDEX_URL=http://localhost:8765 vllm serve --model ...

Manual activation (if .pth is not installed):
    import contextpilot._vllm_hook
    # then start vllm as usual

How it works:
    1. On import, checks CONTEXTPILOT_INDEX_URL env var
    2. If set, registers a sys.meta_path import hook
    3. When vLLM imports block_pool, the hook intercepts and monkey-patches:
       - BlockPool.__init__: adds eviction callback + request tracking state
       - BlockPool.cache_full_blocks: records request_id -> block_hash mappings
       - BlockPool._maybe_evict_cached_block: detects fully evicted requests
       - BlockPool.get_new_blocks: batches and fires eviction callback
       - BlockPool.evict_blocks: batches and fires eviction callback
       - BlockPool.reset_prefix_cache: fires callback for all tracked requests
    4. No vLLM source files are modified

Accuracy:
    Every block eviction goes through _maybe_evict_cached_block(), so eviction
    tracking is exact — identical to the source-code patch approach.
"""

import importlib
import importlib.abc
import logging
import os
import re
import sys

logger = logging.getLogger("contextpilot.vllm_hook")

CONTEXTPILOT_INDEX_URL = os.environ.get("CONTEXTPILOT_INDEX_URL")
_TRACK_ONLY_PREFIX = os.environ.get("CONTEXTPILOT_TRACK_ONLY_PREFIX", "req-")

_VLLM_REQUEST_ID_PREFIX = re.compile(r"^(cmpl-|chatcmpl-|batch-)")
_VLLM_REQUEST_ID_SUFFIX = re.compile(r"^(req-[^-]+)-\d+-[0-9a-f]+$")


def _normalize_request_id(request_id: str) -> str:
    """Normalize vLLM request IDs to ContextPilot canonical form."""
    rid = _VLLM_REQUEST_ID_PREFIX.sub("", request_id)
    m = _VLLM_REQUEST_ID_SUFFIX.match(rid)
    if m:
        return m.group(1)
    return rid


def _should_track_request_id(request_id: str) -> bool:
    if not request_id:
        return False
    stripped = _normalize_request_id(request_id)
    if not stripped or stripped.startswith("HEALTH_CHECK"):
        return False
    if _TRACK_ONLY_PREFIX:
        return stripped.startswith(_TRACK_ONLY_PREFIX)
    return True


def _create_eviction_callback(index_url: str):
    """Create an HTTP callback that notifies ContextPilot index of cache evictions."""
    import requests as http_requests

    def eviction_callback(evicted_request_ids: set):
        if not evicted_request_ids:
            return
        filtered = {
            rid for rid in (
                _normalize_request_id(r) for r in evicted_request_ids
            )
            if rid and not rid.startswith("HEALTH_CHECK")
        }
        if not filtered:
            return
        try:
            preview = ", ".join(sorted(filtered)[:10])
            suffix = ", ..." if len(filtered) > 10 else ""
            logger.info(
                "[ContextPilot] Syncing eviction: %d requests [%s%s]",
                len(filtered), preview, suffix,
            )
            http_requests.post(
                f"{index_url}/evict",
                json={"request_ids": list(filtered)},
                timeout=1.0,
            )
        except Exception as e:
            logger.warning("[ContextPilot] Eviction sync failed: %s", e)

    return eviction_callback


def _apply_block_pool_patches(module, index_url: str):
    """Apply monkey-patches to an already-imported block_pool module."""
    BlockPool = module.BlockPool

    # ------------------------------------------------------------------
    # Source-patch detection: if BlockPool already has tracking methods,
    # the source-code patch is in place and we should not double-patch.
    # ------------------------------------------------------------------
    if (
        hasattr(BlockPool, "get_tracked_request_ids")
        or getattr(BlockPool, "_contextpilot_patched", False)
    ):
        logger.info(
            "[ContextPilot] vLLM source patch detected — skipping monkey-patch"
        )
        return

    logger.info("[ContextPilot] Applying monkey-patches to vLLM BlockPool …")

    eviction_callback = _create_eviction_callback(index_url)

    # ---- BlockPool.__init__ -------------------------------------------
    _orig_init = BlockPool.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._block_to_requests: dict = {}   # block_hash -> set[str]
        self._request_to_blocks: dict = {}   # request_id -> set[block_hash]
        self._eviction_buffer: set = set()
        self.eviction_callback = eviction_callback

    BlockPool.__init__ = _patched_init

    # ---- BlockPool.cache_full_blocks ----------------------------------
    _orig_cache_full_blocks = BlockPool.cache_full_blocks

    def _patched_cache_full_blocks(self, request, blocks, num_cached_blocks,
                                   num_full_blocks, block_size,
                                   kv_cache_group_id):
        _orig_cache_full_blocks(
            self, request, blocks, num_cached_blocks,
            num_full_blocks, block_size, kv_cache_group_id,
        )

        # Track request -> block ownership for newly cached blocks
        if self.eviction_callback is not None and num_cached_blocks < num_full_blocks:
            req_id = request.request_id
            if _should_track_request_id(req_id):
                req_id = _normalize_request_id(req_id)
                for blk in blocks[num_cached_blocks:num_full_blocks]:
                    if blk.is_null or blk.block_hash is None:
                        continue
                    bh = blk.block_hash
                    self._block_to_requests.setdefault(bh, set()).add(req_id)
                    self._request_to_blocks.setdefault(req_id, set()).add(bh)

    BlockPool.cache_full_blocks = _patched_cache_full_blocks

    # ---- BlockPool._maybe_evict_cached_block --------------------------
    # Wraps the original to detect when a request loses ALL its cached
    # blocks (fully evicted).  Uses _eviction_buffer so the caller
    # (get_new_blocks / evict_blocks) can batch-fire the callback.
    _orig_maybe_evict = BlockPool._maybe_evict_cached_block

    def _patched_maybe_evict(self, block):
        block_hash = block.block_hash          # capture before original resets it
        _orig_maybe_evict(self, block)

        # If block had no hash, or hash wasn't reset → no actual eviction
        if block_hash is None or block.block_hash is not None:
            return

        # Last copy of this hash evicted?
        if self.cached_block_hash_to_block.get_one_block(block_hash) is None:
            request_ids = self._block_to_requests.pop(block_hash, None)
            if request_ids:
                for rid in request_ids:
                    blocks_set = self._request_to_blocks.get(rid)
                    if blocks_set is not None:
                        blocks_set.discard(block_hash)
                        if not blocks_set:
                            self._eviction_buffer.add(rid)
                            del self._request_to_blocks[rid]

    BlockPool._maybe_evict_cached_block = _patched_maybe_evict

    # ---- BlockPool.get_new_blocks -------------------------------------
    _orig_get_new_blocks = BlockPool.get_new_blocks

    def _patched_get_new_blocks(self, num_blocks):
        self._eviction_buffer = set()
        result = _orig_get_new_blocks(self, num_blocks)
        if self._eviction_buffer and self.eviction_callback is not None:
            try:
                self.eviction_callback(self._eviction_buffer)
            except Exception as e:
                logger.warning("[ContextPilot] Eviction callback failed: %s", e)
        self._eviction_buffer = set()
        return result

    BlockPool.get_new_blocks = _patched_get_new_blocks

    # ---- BlockPool.evict_blocks ---------------------------------------
    _orig_evict_blocks = BlockPool.evict_blocks

    def _patched_evict_blocks(self, block_ids):
        self._eviction_buffer = set()
        _orig_evict_blocks(self, block_ids)
        if self._eviction_buffer and self.eviction_callback is not None:
            try:
                self.eviction_callback(self._eviction_buffer)
            except Exception as e:
                logger.warning("[ContextPilot] Eviction callback failed: %s", e)
        self._eviction_buffer = set()

    BlockPool.evict_blocks = _patched_evict_blocks

    # ---- BlockPool.reset_prefix_cache ---------------------------------
    _orig_reset = BlockPool.reset_prefix_cache

    def _patched_reset_prefix_cache(self):
        # Check if reset will succeed (mirror original guard)
        num_used = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used == 1:
            # Reset will succeed — fire callback for all tracked requests
            if self._request_to_blocks and self.eviction_callback is not None:
                all_requests = set(self._request_to_blocks.keys())
                try:
                    self.eviction_callback(all_requests)
                except Exception as e:
                    logger.warning("[ContextPilot] Eviction callback failed: %s", e)
            self._block_to_requests.clear()
            self._request_to_blocks.clear()
        return _orig_reset(self)

    BlockPool.reset_prefix_cache = _patched_reset_prefix_cache

    # ---- Convenience methods -----------------------------------------
    def get_tracked_request_ids(self) -> set:
        """Return the set of all request IDs currently tracked in the cache."""
        return set(self._request_to_blocks.keys())

    def is_request_in_cache(self, request_id: str) -> bool:
        """Check if a request still has cached blocks."""
        return request_id in self._request_to_blocks

    def set_eviction_callback(self, callback):
        """Set or replace the eviction callback at runtime."""
        self.eviction_callback = callback

    BlockPool.get_tracked_request_ids = get_tracked_request_ids
    BlockPool.is_request_in_cache = is_request_in_cache
    BlockPool.set_eviction_callback = set_eviction_callback

    # Mark class so we never double-patch
    BlockPool._contextpilot_patched = True
    logger.info("[ContextPilot] vLLM BlockPool monkey-patched successfully")


# ---------------------------------------------------------------------------
# Public API — manual patching (call before creating BlockPool instances)
# ---------------------------------------------------------------------------

def patch_vllm(index_url: str | None = None):
    """Manually apply ContextPilot monkey-patches to vLLM's BlockPool.

    Call this after ``import vllm`` but before any BlockPool is instantiated.

    Args:
        index_url: ContextPilot index server URL.
                   Defaults to ``CONTEXTPILOT_INDEX_URL`` environment variable.
    """
    url = index_url or CONTEXTPILOT_INDEX_URL
    if url is None:
        raise ValueError(
            "No index URL provided.  Pass index_url= or set CONTEXTPILOT_INDEX_URL."
        )
    try:
        module = importlib.import_module("vllm.v1.core.block_pool")
    except ImportError as e:
        raise ImportError("vLLM is not installed or block_pool module not found") from e

    if getattr(module.BlockPool, "_contextpilot_patched", False):
        logger.info("[ContextPilot] Already patched — skipping")
        return

    _apply_block_pool_patches(module, url)


# ---------------------------------------------------------------------------
# Automatic activation via import hook (.pth file triggers this on startup)
# ---------------------------------------------------------------------------

if CONTEXTPILOT_INDEX_URL:

    class _PatchingLoader(importlib.abc.Loader):
        """Wraps the original module loader, applies monkey-patches after exec."""

        def __init__(self, original_loader, index_url):
            self._original = original_loader
            self._index_url = index_url

        def create_module(self, spec):
            if hasattr(self._original, "create_module"):
                return self._original.create_module(spec)
            return None

        def exec_module(self, module):
            self._original.exec_module(module)
            if not getattr(module.BlockPool, "_contextpilot_patched", False):
                _apply_block_pool_patches(module, self._index_url)
            else:
                logger.info(
                    "[ContextPilot] Source patch already applied — skipping monkey-patch"
                )

    class _VLLMImportHook(importlib.abc.MetaPathFinder):
        """Intercept the import of vllm.v1.core.block_pool and patch it."""

        _target = "vllm.v1.core.block_pool"
        _done = False

        def find_spec(self, fullname, path, target=None):
            if fullname != self._target or self._done:
                return None

            self._done = True
            sys.meta_path.remove(self)
            try:
                real_spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)

            if real_spec is None:
                return None

            real_spec.loader = _PatchingLoader(real_spec.loader, CONTEXTPILOT_INDEX_URL)
            return real_spec

    sys.meta_path.insert(0, _VLLMImportHook())
    logger.debug(
        "[ContextPilot] vLLM import hook registered (index: %s)",
        CONTEXTPILOT_INDEX_URL,
    )
