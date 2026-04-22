"""
OpenAI Client Monkey-Patch for ContextPilot.

Patches the OpenAI Python SDK's chat.completions.create() with the full
ContextPilot intercept pipeline:

  1. Restore cached prefix — replace old messages with previously modified
     copies so KV cache prefix stays identical across turns
  2. Extract documents — from system prompt and tool results
  3. Reorder — documents for maximal prefix cache sharing
  4. Cross-turn dedup — single-doc tool results (repeated file reads)
  5. Block-level dedup — content-defined chunking within tool results
  6. Cache modified messages — for next turn's prefix replay

Works with any OpenAI-SDK-based agent: Hermes, OpenHands, Aider, etc.

Activation:
    CONTEXTPILOT=1 hermes chat
    CONTEXTPILOT=1 python my_agent.py

Manual activation:
    import contextpilot._openai_hook
"""

import copy
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List

logger = logging.getLogger("contextpilot.openai_hook")

_ENABLED = os.environ.get("CONTEXTPILOT", "").lower() in (
    "1",
    "true",
    "yes",
) or os.environ.get("CONTEXTPILOT_DEDUP", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Per-session conversation state (mirrors http_server._InterceptConvState)
# ---------------------------------------------------------------------------


@dataclass
class _ConvState:
    cached_messages: list = dc_field(default_factory=list)
    first_tool_result_done: bool = False
    seen_doc_hashes: set = dc_field(default_factory=set)
    single_doc_hashes: dict = dc_field(default_factory=dict)
    system_processed: bool = False
    last_message_count: int = 0


_sessions: Dict[str, _ConvState] = {}
_MAX_SESSIONS = 64

_total_chars_saved = 0
_total_reordered = 0
_total_calls = 0

# Lazy-initialized reorder index (needs numpy)
_intercept_index = None
_has_reorder = None  # None = not checked yet


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _session_fingerprint(messages: list) -> str:
    parts = []
    for msg in messages[:5]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        if role == "system":
            parts.append(str(msg.get("content", ""))[:500])
        elif role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            parts.append(str(content)[:500])
            break
    if not parts:
        return _hash_text(json.dumps(messages[:2], sort_keys=True, default=str))
    return _hash_text("\x00".join(parts))


def _get_state(messages: list) -> _ConvState:
    key = _session_fingerprint(messages)
    msg_count = len(messages)
    state = _sessions.get(key)

    if state is None:
        state = _ConvState()
        if len(_sessions) >= _MAX_SESSIONS:
            oldest = next(iter(_sessions))
            del _sessions[oldest]
        _sessions[key] = state
    elif msg_count < state.last_message_count:
        # Message count dropped → compaction happened, reset state
        state = _ConvState()
        _sessions[key] = state

    state.last_message_count = msg_count
    return state


# ---------------------------------------------------------------------------
# Reorder support (graceful degradation if numpy unavailable)
# ---------------------------------------------------------------------------


def _check_reorder():
    global _has_reorder
    if _has_reorder is not None:
        return _has_reorder
    try:
        from contextpilot.server.live_index import ContextPilot as _CP  # noqa: F401
        from contextpilot.server.intercept_parser import get_format_handler  # noqa: F401

        _has_reorder = True
    except (ImportError, Exception):
        _has_reorder = False
        logger.info("[ContextPilot] Reorder unavailable (numpy?), dedup-only mode")
    return _has_reorder


def _reorder_docs(docs: List[str], alpha: float = 0.001) -> List[str]:
    global _intercept_index
    if len(docs) < 2:
        return docs

    from contextpilot.server.live_index import ContextPilot as CP

    contexts = [docs]

    if _intercept_index is None:
        _intercept_index = CP(alpha=alpha, use_gpu=False, linkage_method="average")
        _intercept_index.build_and_schedule(contexts=contexts)
        return docs  # First call → build only, no reorder

    result = _intercept_index.build_incremental(contexts=contexts)
    reordered = result.get("reordered_contexts", [docs])[0]

    # Map back to original doc objects
    doc_to_orig = {}
    for i, doc in enumerate(docs):
        doc_to_orig.setdefault(doc, []).append(i)
    order = []
    used = set()
    for doc in reordered:
        for idx in doc_to_orig.get(doc, []):
            if idx not in used:
                order.append(idx)
                used.add(idx)
                break

    return [docs[i] for i in order]


# ---------------------------------------------------------------------------
# Full intercept pipeline
# ---------------------------------------------------------------------------


def _optimize_messages(kwargs):
    """Full ContextPilot pipeline: prefix replay → extract → reorder → dedup."""
    global _total_chars_saved, _total_reordered, _total_calls

    messages = kwargs.get("messages")
    if not messages or not isinstance(messages, list):
        return

    _total_calls += 1
    state = _get_state(messages)
    has_reorder = _check_reorder()
    chars_saved = 0
    docs_reordered = 0

    # ── Step 1: Prefix replay (replace old msgs with cached modified copies) ──
    old_count = len(state.cached_messages)
    if old_count > 0 and len(messages) >= old_count:
        prefix_ok = True
        for i in range(old_count):
            cached_h = _hash_text(
                json.dumps(state.cached_messages[i], sort_keys=True, default=str)
            )
            current_h = _hash_text(json.dumps(messages[i], sort_keys=True, default=str))
            if cached_h != current_h:
                prefix_ok = False
                break
        if prefix_ok:
            messages[:old_count] = copy.deepcopy(state.cached_messages)
            kwargs["messages"] = messages

    # ── Step 2-4: Extract & reorder (if intercept_parser available) ──
    if has_reorder:
        try:
            from contextpilot.server.intercept_parser import (
                get_format_handler,
                InterceptConfig,
            )

            config = InterceptConfig(
                enabled=True,
                mode="auto",
                tag="document",
                separator="---",
                alpha=0.001,
                linkage_method="average",
                scope="all",
            )
            handler = get_format_handler("openai_chat")
            body = {"messages": messages}
            multi = handler.extract_all(body, config)

            # Reorder system docs (first turn only)
            if multi.system_extraction and not state.system_processed:
                extraction, sys_idx = multi.system_extraction
                if len(extraction.documents) >= 2:
                    reordered = _reorder_docs(extraction.documents)
                    if reordered != extraction.documents:
                        handler.reconstruct_system(body, extraction, reordered, sys_idx)
                        docs_reordered += len(extraction.documents)
                state.system_processed = True

            # Reorder/dedup tool results
            for extraction, location in multi.tool_extractions:
                if location.msg_index < old_count:
                    continue
                if len(extraction.documents) < 2:
                    continue

                if not state.first_tool_result_done:
                    state.first_tool_result_done = True
                    reordered = _reorder_docs(extraction.documents)
                    for doc in extraction.documents:
                        state.seen_doc_hashes.add(_hash_text(doc))
                    if reordered != extraction.documents:
                        handler.reconstruct_tool_result(
                            body, extraction, reordered, location
                        )
                        docs_reordered += len(extraction.documents)
                else:
                    # Dedup against previously seen docs
                    new_docs = []
                    deduped = 0
                    for doc in extraction.documents:
                        h = _hash_text(doc)
                        if h in state.seen_doc_hashes:
                            deduped += 1
                        else:
                            state.seen_doc_hashes.add(h)
                            new_docs.append(doc)
                    if deduped > 0:
                        if not new_docs:
                            orig_chars = len(extraction.original_content)
                            new_docs = [
                                f"[All {deduped} documents identical to a "
                                f"previous tool result ({orig_chars} chars). "
                                f"Refer to the earlier result above.]"
                            ]
                        handler.reconstruct_tool_result(
                            body, extraction, new_docs, location
                        )

            # Cross-turn single-doc dedup
            for single_doc, location in multi.single_doc_extractions:
                if location.msg_index < old_count:
                    continue
                if single_doc.content_hash in state.single_doc_hashes:
                    prev_id = state.single_doc_hashes[single_doc.content_hash]
                    if single_doc.tool_call_id != prev_id and handler.tool_call_present(
                        body, prev_id
                    ):
                        hint = (
                            f"[Duplicate content — identical to a previous "
                            f"tool result ({prev_id}). "
                            f"Refer to the earlier result above.]"
                        )
                        handler.replace_single_doc(body, location, hint)
                else:
                    state.single_doc_hashes[single_doc.content_hash] = (
                        single_doc.tool_call_id
                    )

            # Sync messages back (handler mutates body["messages"] in-place)
            kwargs["messages"] = body["messages"]
            messages = kwargs["messages"]

        except Exception as e:
            logger.debug("[ContextPilot] Extract/reorder failed: %s", e)

    # ── Step 5: Block-level dedup ──
    from contextpilot.dedup import dedup_chat_completions

    system_content = None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            sc = msg.get("content", "")
            if isinstance(sc, str):
                system_content = sc
            break

    dedup_result = dedup_chat_completions(
        {"messages": messages},
        system_content=system_content,
    )
    chars_saved += dedup_result.chars_saved

    # ── Step 6: Cache for next turn ──
    state.cached_messages = copy.deepcopy(messages)

    _total_chars_saved += chars_saved
    _total_reordered += docs_reordered

    if chars_saved > 0 or docs_reordered > 0:
        logger.info(
            "[ContextPilot] Call #%d: %d chars saved, %d blocks deduped, "
            "%d docs reordered (cumulative: %d chars ≈ %d tokens)",
            _total_calls,
            chars_saved,
            dedup_result.blocks_deduped,
            docs_reordered,
            _total_chars_saved,
            _total_chars_saved // 4,
        )


# ---------------------------------------------------------------------------
# OpenAI SDK monkey-patch (identical pattern to _sglang_hook / _vllm_hook)
# ---------------------------------------------------------------------------


def _apply_openai_patches(module):
    Completions = getattr(module, "Completions", None)
    AsyncCompletions = getattr(module, "AsyncCompletions", None)

    if Completions and not getattr(Completions, "_contextpilot_patched", False):
        _orig_create = Completions.create

        def _patched_create(self, *args, **kwargs):
            _optimize_messages(kwargs)
            return _orig_create(self, *args, **kwargs)

        Completions.create = _patched_create
        Completions._contextpilot_patched = True
        logger.info("[ContextPilot] Patched openai Completions.create")

    if AsyncCompletions and not getattr(
        AsyncCompletions, "_contextpilot_patched", False
    ):
        _orig_async_create = AsyncCompletions.create

        async def _patched_async_create(self, *args, **kwargs):
            _optimize_messages(kwargs)
            return await _orig_async_create(self, *args, **kwargs)

        AsyncCompletions.create = _patched_async_create
        AsyncCompletions._contextpilot_patched = True
        logger.info("[ContextPilot] Patched openai AsyncCompletions.create")


if _ENABLED:

    class _PatchingLoader(importlib.abc.Loader):
        def __init__(self, original_loader):
            self._original = original_loader

        def create_module(self, spec):
            if hasattr(self._original, "create_module"):
                return self._original.create_module(spec)
            return None

        def exec_module(self, module):
            self._original.exec_module(module)
            _apply_openai_patches(module)

    class _OpenAIImportHook(importlib.abc.MetaPathFinder):
        _target = "openai.resources.chat.completions"
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
            real_spec.loader = _PatchingLoader(real_spec.loader)
            return real_spec

    sys.meta_path.insert(0, _OpenAIImportHook())
    logger.debug("[ContextPilot] OpenAI import hook registered (CONTEXTPILOT=1)")

    # Patch eagerly if openai was already imported before us
    _already_loaded = sys.modules.get("openai.resources.chat.completions")
    if _already_loaded is not None:
        _apply_openai_patches(_already_loaded)
