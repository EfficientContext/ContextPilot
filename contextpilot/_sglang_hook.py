"""
SGLang Runtime Monkey-Patch for ContextPilot.

Automatically patches SGLang's RadixCache to track request-to-cache-node
associations and report evictions to the ContextPilot index server.

Activation:
    Set the CONTEXTPILOT_INDEX_URL environment variable before starting SGLang.
    If installed via pip, a .pth file auto-imports this module on Python startup.

    CONTEXTPILOT_INDEX_URL=http://localhost:8765 sglang serve --model-path ...

Manual activation (if .pth is not installed):
    import contextpilot._sglang_hook
    # then start sglang as usual

How it works:
    1. On import, checks CONTEXTPILOT_INDEX_URL env var
    2. If set, registers a sys.meta_path import hook
    3. When SGLang imports radix_cache, the hook intercepts and monkey-patches:
       - TreeNode.__init__: adds request_ids tracking set
       - RadixCache.__init__: adds eviction callback + request tracking state
       - RadixCache._delete_leaf: captures evicted request_ids
       - RadixCache.evict: collects evicted IDs and fires callback
       - RadixCache.cache_finished_req/cache_unfinished_req: sets request context
       - RadixCache._insert_helper: tags new leaf nodes with request_id
    4. No SGLang source files are modified

Accuracy:
    Every node deletion goes through _delete_leaf(), so eviction tracking is
    exact — identical to the source-code patch approach.
"""

import importlib
import importlib.abc
import logging
import os
import sys

logger = logging.getLogger("contextpilot.sglang_hook")

CONTEXTPILOT_INDEX_URL = os.environ.get("CONTEXTPILOT_INDEX_URL")


def _create_eviction_callback(index_url: str):
    """Create an HTTP callback that notifies ContextPilot index of cache evictions."""
    import requests as http_requests

    def eviction_callback(evicted_request_ids: set):
        if not evicted_request_ids:
            return
        # Filter out internal SGLang health-check requests
        filtered = {
            rid for rid in evicted_request_ids if not str(rid).startswith("HEALTH_CHECK")
        }
        if not filtered:
            return
        try:
            logger.info(f"[ContextPilot] Syncing eviction: {len(filtered)} requests")
            http_requests.post(
                f"{index_url}/evict",
                json={"request_ids": list(filtered)},
                timeout=1.0,
            )
        except Exception as e:
            logger.warning(f"[ContextPilot] Eviction sync failed: {e}")

    return eviction_callback


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
    """Apply monkey-patches to an already-imported radix_cache module."""
    TreeNode = module.TreeNode
    RadixCache = module.RadixCache

    # ------------------------------------------------------------------
    # Source-patch detection: if TreeNode already initialises request_ids,
    # the source-code patch is in place and we should not double-patch.
    # ------------------------------------------------------------------
    try:
        _test = TreeNode()
        if hasattr(_test, "request_ids"):
            logger.info(
                "[ContextPilot] SGLang source patch detected — skipping monkey-patch"
            )
            return
    except Exception:
        pass  # TreeNode() might need args in future versions; proceed anyway

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
    # This is the critical interception point.  Every evicted node passes
    # through _delete_leaf, so we capture request_ids HERE — before the
    # node is detached from the tree.
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
                logger.warning(f"[ContextPilot] Eviction callback failed: {e}")
        self._eviction_buffer = set()
        return result

    RadixCache.evict = _patched_evict

    # ---- RadixCache.cache_finished_req -------------------------------
    _orig_cache_finished = RadixCache.cache_finished_req

    def _patched_cache_finished(self, req, *args, **kwargs):
        self._current_rid = getattr(req, "rid", None)
        try:
            return _orig_cache_finished(self, req, *args, **kwargs)
        finally:
            self._current_rid = None

    RadixCache.cache_finished_req = _patched_cache_finished

    # ---- RadixCache.cache_unfinished_req -----------------------------
    _orig_cache_unfinished = RadixCache.cache_unfinished_req

    def _patched_cache_unfinished(self, req, *args, **kwargs):
        self._current_rid = getattr(req, "rid", None)
        try:
            return _orig_cache_unfinished(self, req, *args, **kwargs)
        finally:
            self._current_rid = None

    RadixCache.cache_unfinished_req = _patched_cache_unfinished

    # ---- RadixCache._insert_helper -----------------------------------
    # After the original _insert_helper creates a new leaf node, we walk
    # the tree to find that leaf and tag it with the current request_id.
    _orig_insert_helper = RadixCache._insert_helper

    def _patched_insert_helper(self, node, key, value, priority=0, **kwargs):
        evictable_before = self.evictable_size_
        result = _orig_insert_helper(self, node, key, value, priority, **kwargs)
        rid = self._current_rid
        if rid is not None and self.evictable_size_ > evictable_before:
            # New tokens were added — the tree now has a new leaf.
            # Walk from root with the (already-converted) key to find it.
            leaf = _find_leaf(self, key)
            if leaf is not None and leaf is not self.root_node:
                leaf.request_ids.add(rid)
                self._request_to_node[rid] = leaf
        return result

    RadixCache._insert_helper = _patched_insert_helper

    # ---- Convenience methods -----------------------------------------
    def set_eviction_callback(self, callback):
        """Set or update the eviction callback."""
        self.eviction_callback = callback

    def get_tracked_request_ids(self) -> set:
        """Get request IDs that currently have extra tokens in cache."""
        return set(self._request_to_node.keys())

    def is_request_in_cache(self, request_id: str) -> bool:
        """Check if a request has extra tokens in the cache."""
        return request_id in self._request_to_node

    def get_request_node(self, request_id: str):
        """Get the tree node where a request's extra tokens are stored."""
        return self._request_to_node.get(request_id)

    RadixCache.set_eviction_callback = set_eviction_callback
    RadixCache.get_tracked_request_ids = get_tracked_request_ids
    RadixCache.is_request_in_cache = is_request_in_cache
    RadixCache.get_request_node = get_request_node

    # Mark class so we never double-patch
    RadixCache._contextpilot_patched = True
    logger.info("[ContextPilot] SGLang RadixCache monkey-patched successfully")


# ---------------------------------------------------------------------------
# Public API — manual patching (call before creating RadixCache instances)
# ---------------------------------------------------------------------------

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
            # Module is now fully loaded — apply patches
            if not getattr(module.RadixCache, "_contextpilot_patched", False):
                _apply_radix_cache_patches(module, self._index_url)
            else:
                logger.info(
                    "[ContextPilot] Source patch already applied — skipping monkey-patch"
                )

    class _SGLangImportHook(importlib.abc.MetaPathFinder):
        """Intercept the import of sglang.srt.mem_cache.radix_cache and patch it."""

        _target = "sglang.srt.mem_cache.radix_cache"
        _done = False

        def find_spec(self, fullname, path, target=None):
            if fullname != self._target or self._done:
                return None

            self._done = True
            # Remove ourselves temporarily to find the REAL spec
            sys.meta_path.remove(self)
            try:
                real_spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)

            if real_spec is None:
                return None

            # Swap the loader so we can patch after exec_module
            real_spec.loader = _PatchingLoader(real_spec.loader, CONTEXTPILOT_INDEX_URL)
            return real_spec

    sys.meta_path.insert(0, _SGLangImportHook())
    logger.debug(
        f"[ContextPilot] Import hook registered (index: {CONTEXTPILOT_INDEX_URL})"
    )
