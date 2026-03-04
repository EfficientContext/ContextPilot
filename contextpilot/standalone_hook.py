"""
Standalone ContextPilot Hook for SGLang, vLLM, and llama.cpp.

A single, self-contained file that monkey-patches SGLang's RadixCache and/or
vLLM's BlockPool to track request-to-cache associations and report evictions
to the ContextPilot index server.  For llama.cpp (a C++ binary), it provides
a polling-based LlamaCppSlotWatcher instead of a monkey-patch.

Zero dependency on the ``contextpilot`` package — only stdlib + ``requests``
(lazy-imported on first eviction callback).

Activation:
    Set CONTEXTPILOT_INDEX_URL before starting your inference engine.
    If installed via the companion install script, a .pth file auto-imports
    this module on Python startup.

    # SGLang / vLLM (import hook activates automatically):
    CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m sglang.launch_server ...
    CONTEXTPILOT_INDEX_URL=http://localhost:8765 vllm serve ...

    # llama.cpp (polling watcher activates automatically):
    CONTEXTPILOT_INDEX_URL=http://localhost:8765 \\
    CONTEXTPILOT_LLAMACPP_URL=http://localhost:8889 \\
    python contextpilot_hook.py

Manual activation (without .pth):
    import contextpilot_hook        # registers import hooks for SGLang/vLLM
                                    # and starts llama.cpp watcher if env vars set
"""

import importlib
import importlib.abc
import importlib.util
import logging
import os
import re
import sys
import threading
import time

logger = logging.getLogger("contextpilot_hook")

CONTEXTPILOT_INDEX_URL = os.environ.get("CONTEXTPILOT_INDEX_URL")
CONTEXTPILOT_LLAMACPP_URL = os.environ.get("CONTEXTPILOT_LLAMACPP_URL")
_TRACK_ONLY_PREFIX = os.environ.get("CONTEXTPILOT_TRACK_ONLY_PREFIX", "req-")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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
    """Create an HTTP callback that notifies the ContextPilot index of cache evictions."""
    import requests as http_requests  # lazy — only needed at eviction time

    def eviction_callback(evicted_request_ids: set):
        if not evicted_request_ids:
            return
        filtered = {
            rid for rid in (
                _normalize_request_id(str(r)) for r in evicted_request_ids
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


# ===========================================================================
# SGLang patches
# ===========================================================================

def _find_leaf(cache, key):
    """Walk the radix tree from root following *key* to the deepest matching node."""
    node = cache.root_node
    if len(key) == 0:
        return node
    child_key = cache.get_child_key_fn(key)
    while child_key in node.children:
        child = node.children[child_key]
        prefix_len = cache.key_match_fn(child.key, key)
        if prefix_len == 0:
            break
        key = key[prefix_len:]
        node = child
        if len(key) == 0:
            break
        child_key = cache.get_child_key_fn(key)
    return node


def _apply_radix_cache_patches(module, index_url: str):
    """Apply monkey-patches to an already-imported SGLang radix_cache module."""
    TreeNode = module.TreeNode
    RadixCache = module.RadixCache

    # Double-patch guard
    if getattr(RadixCache, "_contextpilot_patched", False):
        logger.info("[ContextPilot] SGLang already patched — skipping")
        return

    # Source-patch detection
    try:
        _test = TreeNode()
        if hasattr(_test, "request_ids"):
            logger.info(
                "[ContextPilot] SGLang source patch detected — skipping monkey-patch"
            )
            return
    except Exception:
        pass

    logger.info("[ContextPilot] Applying monkey-patches to SGLang RadixCache …")

    # ---- TreeNode.__init__ -------------------------------------------
    _orig_tree_init = TreeNode.__init__

    def _patched_tree_init(self, *args, **kwargs):
        _orig_tree_init(self, *args, **kwargs)
        self.request_ids: set = set()

    TreeNode.__init__ = _patched_tree_init

    # ---- RadixCache.__init__ -----------------------------------------
    _orig_cache_init = RadixCache.__init__

    def _patched_cache_init(self, *args, **kwargs):
        _orig_cache_init(self, *args, **kwargs)
        self._request_to_node: dict = {}
        self._eviction_buffer: set = set()
        self._current_rid = None
        self.eviction_callback = _create_eviction_callback(index_url)

    RadixCache.__init__ = _patched_cache_init

    # ---- RadixCache.reset --------------------------------------------
    _orig_reset = RadixCache.reset

    def _patched_reset(self, *args, **kwargs):
        result = _orig_reset(self, *args, **kwargs)
        self._request_to_node = {}
        return result

    RadixCache.reset = _patched_reset

    # ---- RadixCache._delete_leaf -------------------------------------
    _orig_delete_leaf = RadixCache._delete_leaf

    def _patched_delete_leaf(self, node):
        if hasattr(node, "request_ids") and node.request_ids:
            self._eviction_buffer.update(node.request_ids)
            for rid in node.request_ids:
                self._request_to_node.pop(rid, None)
        _orig_delete_leaf(self, node)

    RadixCache._delete_leaf = _patched_delete_leaf

    # ---- RadixCache.evict --------------------------------------------
    _orig_evict = RadixCache.evict

    def _patched_evict(self, *args, **kwargs):
        self._eviction_buffer = set()
        result = _orig_evict(self, *args, **kwargs)
        if self._eviction_buffer and self.eviction_callback is not None:
            try:
                self.eviction_callback(self._eviction_buffer)
            except Exception as e:
                logger.warning("[ContextPilot] Eviction callback failed: %s", e)
        self._eviction_buffer = set()
        return result

    RadixCache.evict = _patched_evict

    # ---- RadixCache.cache_finished_req -------------------------------
    _orig_cache_finished = RadixCache.cache_finished_req

    def _patched_cache_finished(self, req, *args, **kwargs):
        rid = getattr(req, "rid", None)
        logger.debug("[ContextPilot] cache_finished_req: rid=%s", rid)
        self._current_rid = rid
        try:
            return _orig_cache_finished(self, req, *args, **kwargs)
        finally:
            self._current_rid = None

    RadixCache.cache_finished_req = _patched_cache_finished

    # ---- RadixCache.cache_unfinished_req -----------------------------
    _orig_cache_unfinished = RadixCache.cache_unfinished_req

    def _patched_cache_unfinished(self, req, *args, **kwargs):
        rid = getattr(req, "rid", None)
        logger.debug("[ContextPilot] cache_unfinished_req: rid=%s", rid)
        self._current_rid = rid
        try:
            return _orig_cache_unfinished(self, req, *args, **kwargs)
        finally:
            self._current_rid = None

    RadixCache.cache_unfinished_req = _patched_cache_unfinished

    # ---- RadixCache._insert_helper -----------------------------------
    _orig_insert_helper = RadixCache._insert_helper

    def _patched_insert_helper(self, node, key, value, priority=0, **kwargs):
        evictable_before = self.evictable_size_
        result = _orig_insert_helper(self, node, key, value, priority, **kwargs)
        rid = self._current_rid
        if rid is not None and self.evictable_size_ > evictable_before:
            leaf = _find_leaf(self, key)
            if leaf is not None and leaf is not self.root_node:
                leaf.request_ids.add(rid)
                self._request_to_node[rid] = leaf
        return result

    RadixCache._insert_helper = _patched_insert_helper

    # ---- Convenience methods -----------------------------------------
    def set_eviction_callback(self, callback):
        self.eviction_callback = callback

    def get_tracked_request_ids(self) -> set:
        return set(self._request_to_node.keys())

    def is_request_in_cache(self, request_id: str) -> bool:
        return request_id in self._request_to_node

    def get_request_node(self, request_id: str):
        return self._request_to_node.get(request_id)

    RadixCache.set_eviction_callback = set_eviction_callback
    RadixCache.get_tracked_request_ids = get_tracked_request_ids
    RadixCache.is_request_in_cache = is_request_in_cache
    RadixCache.get_request_node = get_request_node

    RadixCache._contextpilot_patched = True
    logger.info("[ContextPilot] SGLang RadixCache monkey-patched successfully")


# ===========================================================================
# vLLM patches
# ===========================================================================

def _apply_block_pool_patches(module, index_url: str):
    """Apply monkey-patches to an already-imported vLLM block_pool module."""
    BlockPool = module.BlockPool

    # Double-patch guard
    if (
        hasattr(BlockPool, "get_tracked_request_ids")
        or getattr(BlockPool, "_contextpilot_patched", False)
    ):
        logger.info("[ContextPilot] vLLM already patched — skipping")
        return

    logger.info("[ContextPilot] Applying monkey-patches to vLLM BlockPool …")

    eviction_callback = _create_eviction_callback(index_url)

    # ---- BlockPool.__init__ -------------------------------------------
    _orig_init = BlockPool.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._block_to_requests: dict = {}
        self._request_to_blocks: dict = {}
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
    _orig_maybe_evict = BlockPool._maybe_evict_cached_block

    def _patched_maybe_evict(self, block):
        block_hash = block.block_hash
        _orig_maybe_evict(self, block)
        if block_hash is None or block.block_hash is not None:
            return
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
        num_used = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used == 1:
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
        return set(self._request_to_blocks.keys())

    def is_request_in_cache(self, request_id: str) -> bool:
        return request_id in self._request_to_blocks

    def set_eviction_callback(self, callback):
        self.eviction_callback = callback

    BlockPool.get_tracked_request_ids = get_tracked_request_ids
    BlockPool.is_request_in_cache = is_request_in_cache
    BlockPool.set_eviction_callback = set_eviction_callback

    BlockPool._contextpilot_patched = True
    logger.info("[ContextPilot] vLLM BlockPool monkey-patched successfully")


# ===========================================================================
# Public API — manual patching
# ===========================================================================

def patch_sglang(index_url: str | None = None):
    """Manually apply ContextPilot monkey-patches to SGLang's RadixCache.

    Call this after ``import sglang`` but before any RadixCache is instantiated.

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
        module = importlib.import_module("sglang.srt.mem_cache.radix_cache")
    except ImportError as e:
        raise ImportError("SGLang is not installed or radix_cache module not found") from e

    if getattr(module.RadixCache, "_contextpilot_patched", False):
        logger.info("[ContextPilot] Already patched — skipping")
        return

    _apply_radix_cache_patches(module, url)


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


# ===========================================================================
# Automatic activation via import hooks (.pth file triggers this on startup)
# ===========================================================================

if CONTEXTPILOT_INDEX_URL:

    class _PatchingLoader(importlib.abc.Loader):
        """Wraps the original module loader, applies monkey-patches after exec."""

        def __init__(self, original_loader, index_url, apply_fn):
            self._original = original_loader
            self._index_url = index_url
            self._apply_fn = apply_fn

        def create_module(self, spec):
            if hasattr(self._original, "create_module"):
                return self._original.create_module(spec)
            return None

        def exec_module(self, module):
            self._original.exec_module(module)
            self._apply_fn(module, self._index_url)

    class _SGLangImportHook(importlib.abc.MetaPathFinder):
        """Intercept the import of sglang.srt.mem_cache.radix_cache and patch it."""

        _target = "sglang.srt.mem_cache.radix_cache"
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
            real_spec.loader = _PatchingLoader(
                real_spec.loader, CONTEXTPILOT_INDEX_URL, _apply_radix_cache_patches,
            )
            return real_spec

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
            real_spec.loader = _PatchingLoader(
                real_spec.loader, CONTEXTPILOT_INDEX_URL, _apply_block_pool_patches,
            )
            return real_spec

    sys.meta_path.insert(0, _SGLangImportHook())
    sys.meta_path.insert(0, _VLLMImportHook())
    logger.debug(
        "[ContextPilot] Import hooks registered (index: %s)",
        CONTEXTPILOT_INDEX_URL,
    )


# ===========================================================================
# llama.cpp slot watcher
# ===========================================================================

_DEFAULT_POLL_INTERVAL = 0.5  # seconds


class LlamaCppSlotWatcher:
    """
    Polls llama-server's /slots endpoint and reports slot evictions to ContextPilot.

    Each llama-server KV-cache slot is analogous to a RadixCache node in SGLang
    or a BlockPool entry in vLLM: when a slot's cached tokens are discarded
    (n_past resets to 0), the associated request is evicted from the index.

    Requires llama-server to be started with ``--endpoint-slots``.

    Usage::

        watcher = LlamaCppSlotWatcher("http://localhost:8889", "http://localhost:8765")
        watcher.start()

        # Inject slot into each request before forwarding:
        slot_id = watcher.next_slot()
        body["id_slot"] = slot_id
        watcher.register_slot(slot_id, request_id)

        watcher.stop()
    """

    def __init__(
        self,
        llamacpp_url: str,
        index_url: str,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ):
        self._llamacpp_url = llamacpp_url.rstrip("/")
        self._eviction_callback = _create_eviction_callback(index_url)
        self._poll_interval = poll_interval

        # slot_id -> {"request_id": str | None, "n_past": int, "state": int}
        self._slot_state: dict = {}
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._n_slots: int = 0
        self._rr_counter: int = 0

    def register_slot(self, slot_id: int, request_id: str) -> None:
        """Associate a request_id with a slot after injecting id_slot into the request."""
        with self._lock:
            self._slot_state.setdefault(slot_id, {"n_past": 0, "state": 0})[
                "request_id"
            ] = request_id
        logger.debug("[ContextPilot] Registered slot %d -> %s", slot_id, request_id)

    def next_slot(self) -> int:
        """Return the next slot ID for round-robin allocation."""
        with self._lock:
            n = self._n_slots or 1
            slot_id = self._rr_counter % n
            self._rr_counter += 1
        return slot_id

    def get_tracked_slots(self) -> dict:
        """Return {slot_id: request_id} for all currently registered slots."""
        with self._lock:
            return {
                sid: info["request_id"]
                for sid, info in self._slot_state.items()
                if info.get("request_id")
            }

    def start(self) -> None:
        """Start the background slot-polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="contextpilot-llamacpp-watcher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[ContextPilot] llama.cpp slot watcher started "
            "(polling %s/slots every %.1fs)",
            self._llamacpp_url, self._poll_interval,
        )

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("[ContextPilot] llama.cpp slot watcher stopped")

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._check_slots()
            except Exception as e:
                logger.debug("[ContextPilot] Slot poll error: %s", e)
            self._stop_event.wait(self._poll_interval)

    def _check_slots(self) -> None:
        import requests as http_requests
        resp = http_requests.get(f"{self._llamacpp_url}/slots", timeout=2.0)
        if resp.status_code != 200:
            return

        slots = resp.json()
        evicted: set = set()

        with self._lock:
            if self._n_slots == 0 and slots:
                self._n_slots = len(slots)
                logger.debug(
                    "[ContextPilot] Discovered %d llama.cpp slots", self._n_slots
                )

            for slot in slots:
                slot_id = slot.get("id", -1)
                if slot_id < 0:
                    continue

                new_state = slot.get("state", 0)   # 0=idle, 1=processing
                new_n_past = slot.get("n_past", 0)

                prev = self._slot_state.get(slot_id, {})
                prev_n_past = prev.get("n_past", 0)
                request_id = prev.get("request_id")

                # Eviction: cached KV tokens (n_past > 0) are gone (n_past == 0)
                # while the slot is idle — its context has been discarded.
                if request_id and prev_n_past > 0 and new_n_past == 0 and new_state == 0:
                    evicted.add(request_id)
                    prev.pop("request_id", None)
                    logger.debug(
                        "[ContextPilot] Slot %d evicted request %s (n_past %d → 0)",
                        slot_id, request_id, prev_n_past,
                    )

                self._slot_state.setdefault(slot_id, {}).update(
                    state=new_state, n_past=new_n_past
                )

        if evicted:
            self._eviction_callback(evicted)


def watch_llamacpp(
    llamacpp_url: str | None = None,
    index_url: str | None = None,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> LlamaCppSlotWatcher:
    """Start a background slot watcher for a llama-server instance.

    The llama.cpp counterpart of patch_sglang() / patch_vllm().
    Because llama-server is a C++ binary, eviction tracking is done by
    polling GET /slots instead of monkey-patching Python internals.

    Args:
        llamacpp_url:  llama-server base URL. Defaults to CONTEXTPILOT_LLAMACPP_URL.
        index_url:     ContextPilot index URL. Defaults to CONTEXTPILOT_INDEX_URL.
        poll_interval: Seconds between /slots polls (default 0.5).

    Returns:
        A running LlamaCppSlotWatcher. Call .stop() to halt.

    Note:
        llama-server must be started with --endpoint-slots.
    """
    url = index_url or CONTEXTPILOT_INDEX_URL
    if url is None:
        raise ValueError(
            "No index URL provided.  Pass index_url= or set CONTEXTPILOT_INDEX_URL."
        )
    lcpp_url = llamacpp_url or CONTEXTPILOT_LLAMACPP_URL
    if lcpp_url is None:
        raise ValueError(
            "No llama.cpp URL provided.  "
            "Pass llamacpp_url= or set CONTEXTPILOT_LLAMACPP_URL."
        )
    watcher = LlamaCppSlotWatcher(
        llamacpp_url=lcpp_url,
        index_url=url,
        poll_interval=poll_interval,
    )
    watcher.start()
    return watcher


# ---------------------------------------------------------------------------
# Auto-start llama.cpp watcher when both env vars are set
# ---------------------------------------------------------------------------

_llamacpp_watcher: LlamaCppSlotWatcher | None = None

if CONTEXTPILOT_INDEX_URL and CONTEXTPILOT_LLAMACPP_URL:
    _llamacpp_watcher = watch_llamacpp()
    logger.debug(
        "[ContextPilot] llama.cpp watcher auto-started "
        "(index: %s, llama.cpp: %s)",
        CONTEXTPILOT_INDEX_URL, CONTEXTPILOT_LLAMACPP_URL,
    )
