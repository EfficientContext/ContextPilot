"""
ContextPilot Live Index HTTP Server

A FastAPI-based HTTP server that:
1. Exposes ContextPilot as a REST API
2. Proxies LLM requests to inference engine backend
3. Automatically tracks tokens and triggers eviction
4. Multi-turn conversation context deduplication

Usage:
    python -m contextpilot.server.http_server --port 8765 --infer-api-url http://localhost:30000

Environment variables (alternative to CLI args):
    CONTEXTPILOT_MAX_TOKENS: Maximum tokens allowed in index
    CONTEXTPILOT_INFER_API_URL: Inference backend URL (default: http://localhost:30000)
"""

import argparse
import copy
import hashlib
import json
import logging
import time
import asyncio
import os
import re
import uuid
from dataclasses import dataclass, field as dc_field
from typing import List, Dict, Any, Optional, cast
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn
    import aiohttp
except ImportError:
    raise ImportError(
        "FastAPI, uvicorn, and aiohttp are required for the HTTP server. "
        "Install with: pip install fastapi uvicorn pydantic aiohttp"
    )

from .live_index import ContextPilot
from .conversation_tracker import (
    ConversationTracker,
    DeduplicationResult,
    get_conversation_tracker,
    reset_conversation_tracker,
)
from .intercept_parser import (
    parse_intercept_headers,
    InterceptConfig,
    get_format_handler,
)
from .ttl_eviction import TTLEvictionPolicy, TTLTier, CacheMetrics
from .cloud_adapters import get_cloud_adapter, CloudProviderAdapter
from contextpilot.dedup import (
    dedup_chat_completions,
    dedup_responses_api,
    DedupResult,
)


logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
    logger.warning(
        "transformers not installed. Chat template functionality will be unavailable. "
        "Install with: pip install transformers"
    )


# Global state (initialized from env vars or CLI args)
_index: Optional[ContextPilot] = None
_max_tokens: Optional[int] = None
_infer_api_url: Optional[str] = None
_aiohttp_session: Optional[aiohttp.ClientSession] = None
_tokenizer = None  # AutoTokenizer instance for chat template
_model_name: Optional[str] = None  # Model name for tokenizer
_stateless_mode: bool = (
    False  # Stateless mode: just clustering/scheduling, no cache tracking
)
# Cloud proxy mode: forward to cloud LLM API with prompt cache optimization
_cloud_mode: bool = False
_chunk_modulus: int = 13
_cloud_adapter: Optional[CloudProviderAdapter] = None
_cloud_api_key: Optional[str] = None
_ttl_policy: Optional[TTLEvictionPolicy] = None
# Persistent string-to-ID mapping for string-input mode.
# Same string always gets the same integer ID across /reorder calls.
_str_to_id: Dict[str, int] = {}
_id_to_str: Dict[int, str] = {}
_next_str_id: int = 0

# Persistent index for the intercept path.  First request builds it
# (no reorder); subsequent requests use build_incremental to search
# the existing tree and reorder documents for prefix sharing.
_intercept_index: Optional[ContextPilot] = None

# ── Conversation-aware intercept state ────────────────────────────────────
# Tracks which tool results have already been processed, enabling
# skip-old / dedup-new / reorder-new behaviour.  Per-session model
# keyed by (system prompt + first user message).  Resets on compaction.

@dataclass
class _InterceptConvState:
    """Per-session intercept state for a single conversation."""
    # Cached copy of the full messages array after modification (reorder/dedup).
    # On subsequent turns, old messages are replaced with these cached versions
    # so the inference engine's prefix cache sees identical tokens.
    cached_messages: list = dc_field(default_factory=list)
    # Cached system prompt (Anthropic format only) after modification.
    cached_system: Any = None
    # Whether the first tool result (reorder candidate) has been processed.
    first_tool_result_done: bool = False
    # Hashes of individual document strings seen across all tool results.
    seen_doc_hashes: set = dc_field(default_factory=set)
    # Hashes of single-doc tool_results (file reads, etc.) → tool_call_id.
    # Used for cross-turn dedup of individual file reads like SKILL.md.
    single_doc_hashes: dict = dc_field(default_factory=dict)
    # Whether the system prompt has been processed (reordered) already.
    system_processed: bool = False
    # Number of messages in the last request.  Messages only grow in a
    # multi-turn conversation; if the count drops, it's a new session.
    last_message_count: int = 0


_intercept_states: dict[str, _InterceptConvState] = {}
_intercept_states_lock = asyncio.Lock()
_MAX_TRACKED_SESSIONS = 64  # LRU eviction threshold

# TTFT tracking for averages across a session
_ttft_history: List[float] = []
_ttft_chars_saved_total = 0


def _log_ttft(ttft_ms: float, slimmed: int, chars_saved: int) -> None:
    global _ttft_chars_saved_total
    _ttft_history.append(ttft_ms)
    _ttft_chars_saved_total += chars_saved
    avg = sum(_ttft_history) / len(_ttft_history)
    logger.info(
        f"TTFT: {ttft_ms:.0f}ms "
        f"(avg {avg:.0f}ms over {len(_ttft_history)} reqs, "
        f"slimmed {slimmed}, saved {chars_saved:,} chars, "
        f"total saved {_ttft_chars_saved_total:,} chars)"
    )


# Request ID normalization (engine -> ContextPilot canonical IDs)
_ENGINE_REQ_ID_PREFIX = re.compile(r"^(cmpl-|chatcmpl-|batch-)")
_VLLM_REQ_SUFFIX = re.compile(r"^(req-[^-]+)-\d+-[0-9a-f]+$")


def _normalize_request_id(request_id: str) -> str:
    """Normalize engine-specific request IDs to ContextPilot canonical form."""
    rid = _ENGINE_REQ_ID_PREFIX.sub("", request_id or "")
    m = _VLLM_REQ_SUFFIX.match(rid)
    if m:
        return m.group(1)
    return rid


def _init_config():
    """Initialize config from environment variables."""
    global _max_tokens, _infer_api_url, _tokenizer, _model_name, _stateless_mode
    global _cloud_mode, _cloud_adapter, _cloud_api_key, _ttl_policy

    # Check stateless mode first
    env_stateless = os.environ.get("CONTEXTPILOT_STATELESS_MODE", "0")
    _stateless_mode = env_stateless == "1"

    # Check cloud proxy mode
    cloud_provider = os.environ.get("CONTEXTPILOT_CLOUD_PROVIDER")
    if cloud_provider and _cloud_adapter is None:
        _cloud_mode = True
        _cloud_adapter = get_cloud_adapter(cloud_provider)
        _cloud_api_key = os.environ.get("CONTEXTPILOT_CLOUD_API_KEY", "")
        extended = os.environ.get("CONTEXTPILOT_EXTENDED_CACHE") == "1"
        if extended:
            ext_seconds = _cloud_adapter.get_extended_ttl_seconds()
            if ext_seconds is None:
                logger.warning(
                    f"{cloud_provider} does not support --extended-cache, ignoring"
                )
                ttl_seconds = _cloud_adapter.get_default_ttl_seconds()
            else:
                ttl_seconds = ext_seconds
                _cloud_adapter.configured_ttl = TTLTier.LONG
        else:
            ttl_seconds = _cloud_adapter.get_default_ttl_seconds()
        _ttl_policy = TTLEvictionPolicy(
            default_ttl_seconds=ttl_seconds,
        )
        logger.info(
            f"Cloud proxy mode: provider={cloud_provider}, "
            f"index_ttl={ttl_seconds}s"
            + (
                " (extended)"
                if extended and _cloud_adapter.supports_extended_cache
                else ""
            )
        )

    if _max_tokens is None and not _stateless_mode:
        env_max_tokens = os.environ.get("CONTEXTPILOT_MAX_TOKENS")
        if env_max_tokens:
            _max_tokens = int(env_max_tokens)

    if _infer_api_url is None:
        _infer_api_url = os.environ.get(
            "CONTEXTPILOT_INFER_API_URL", "http://localhost:30000"
        )

    # Initialize tokenizer for chat template if model is specified
    if _tokenizer is None:
        env_model = os.environ.get("CONTEXTPILOT_MODEL_NAME")
        if env_model and AutoTokenizer is not None:
            try:
                _model_name = env_model
                _tokenizer = AutoTokenizer.from_pretrained(_model_name)
                logger.info(f"Loaded tokenizer for chat template: {_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer for {env_model}: {e}")


# Request/Response Models
class BuildIndexRequest(BaseModel):
    """Request to build the index (legacy, use ReorderRequest instead)."""

    contexts: List[List[Any]] = Field(
        ..., description="List of contexts (each is a list of document IDs)"
    )
    initial_tokens_per_context: int = Field(
        0, description="Initial token count per context"
    )
    alpha: float = Field(0.001, description="Distance computation parameter")
    use_gpu: bool = Field(False, description="Use GPU for distance computation")
    linkage_method: str = Field("average", description="Linkage method for clustering")
    # Multi-turn deduplication fields
    parent_request_ids: Optional[List[Optional[str]]] = Field(
        None,
        description="List of parent request IDs for multi-turn deduplication. "
        "Each element corresponds to a context. None means turn 1 (no parent).",
    )
    deduplicate: bool = Field(
        False, description="If True, deduplicate contexts based on conversation history"
    )
    hint_template: Optional[str] = Field(
        None,
        description="Template for reference hints. Use {doc_id} and {turn_number} placeholders.",
    )


class ScheduleRequest(BaseModel):
    """Request to schedule a batch (legacy, use ReorderRequest instead)."""

    contexts: List[List[Any]] = Field(
        ...,
        description="List of contexts. Each context is a list of items (int doc IDs OR string doc contents). "
        "If strings are provided, identical strings are treated as the same document.",
    )
    alpha: float = Field(0.001, description="Distance computation parameter")
    use_gpu: bool = Field(False, description="Use GPU for distance computation")
    linkage_method: str = Field("average", description="Linkage method for clustering")


class ReorderRequest(BaseModel):
    """Unified request for context reordering (works in both stateless and stateful modes)."""

    contexts: List[List[Any]] = Field(
        ...,
        description="List of contexts. Each context is a list of items (int doc IDs OR string doc contents). "
        "If strings are provided, identical strings are treated as the same document.",
    )
    alpha: float = Field(0.001, description="Distance computation parameter")
    use_gpu: bool = Field(False, description="Use GPU for distance computation")
    linkage_method: str = Field("average", description="Linkage method for clustering")
    # Stateful-mode fields (ignored in stateless mode)
    initial_tokens_per_context: int = Field(
        0, description="Initial token count per context (stateful mode only)"
    )
    parent_request_ids: Optional[List[Optional[str]]] = Field(
        None,
        description="Parent request IDs for multi-turn deduplication (stateful mode only)",
    )
    deduplicate: bool = Field(
        False,
        description="If True, deduplicate contexts based on conversation history (stateful mode only)",
    )
    hint_template: Optional[str] = Field(
        None, description="Template for reference hints (stateful mode only)"
    )


class EvictRequest(BaseModel):
    """Request to evict (remove) requests from the index."""

    request_ids: List[str] = Field(
        ..., description="List of request IDs to evict/remove"
    )


class SearchRequest(BaseModel):
    """Request to search for a context."""

    context: List[Any] = Field(..., description="Query context (list of document IDs)")
    update_access: bool = Field(True, description="Whether to update LRU timestamp")


class InsertRequest(BaseModel):
    """Request to insert a new context."""

    context: List[Any] = Field(..., description="New context to insert")
    search_path: List[int] = Field(..., description="Search path from search operation")
    total_tokens: int = Field(0, description="Initial token count")


class DeduplicateRequest(BaseModel):
    """Request to deduplicate contexts for multi-turn conversations."""

    contexts: List[List[Any]] = Field(
        ..., description="List of contexts (each is a list of document IDs)"
    )
    parent_request_ids: List[Optional[str]] = Field(
        ...,
        description="List of parent request IDs. Each element corresponds to a context. "
        "None means turn 1 (no parent, will be registered for future dedup).",
    )
    hint_template: Optional[str] = Field(
        None,
        description="Template for reference hints. Use {doc_id} and {turn_number} placeholders.",
    )


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _aiohttp_session

    # Initialize config from environment variables
    _init_config()

    logger.info("ContextPilot Index Server starting...")
    logger.info(f"  stateless_mode: {_stateless_mode}")
    if not _stateless_mode:
        logger.info(f"  max_tokens: {_max_tokens}")
    logger.info(f"  infer_api_url: {_infer_api_url}")

    _aiohttp_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600))
    yield
    if _aiohttp_session:
        await _aiohttp_session.close()
    logger.info("ContextPilot Index Server shutting down...")


app = FastAPI(
    title="ContextPilot Live Index Server",
    description="HTTP API for ContextPilot with inference engine proxy and eviction synchronization",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "ContextPilot Live Index Server",
        "status": "running",
        "mode": "stateless" if _stateless_mode else "live",
        "index_initialized": _index is not None,
        "timestamp": time.time(),
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    global _max_tokens

    # Ensure config is initialized from env vars
    _init_config()

    # Stateless mode health check
    if _stateless_mode:
        return {
            "status": "ready",
            "mode": "stateless",
            "eviction_enabled": False,
            "message": "Stateless mode: clustering and scheduling only, no cache tracking",
            "timestamp": time.time(),
        }

    if _index is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "message": "Index not initialized. Call POST /reorder first.",
            },
        )

    stats = _index.get_stats()
    current_tokens = stats.get("total_tokens", 0)

    # max_tokens is guaranteed to be set
    return {
        "status": "ready",
        "mode": "live",
        "eviction_enabled": True,
        "max_tokens": _max_tokens,
        "current_tokens": current_tokens,
        "utilization_pct": (current_tokens / _max_tokens * 100) if _max_tokens else 0,
        "index_stats": stats,
        "timestamp": time.time(),
    }


@app.get("/metrics/ttft")
async def metrics_ttft(last: int = 0):
    """Return TTFT history for benchmarking.

    Query params:
        last: return only last N entries (0 = all).
    """
    history = list(_ttft_history)
    if last > 0:
        history = history[-last:]
    avg = sum(history) / len(history) if history else 0
    return {
        "ttft_ms": history,
        "count": len(history),
        "avg_ms": round(avg, 2),
        "total_chars_saved": _ttft_chars_saved_total,
    }


@app.post("/metrics/ttft/reset")
async def metrics_ttft_reset():
    """Reset TTFT history (call before a benchmark run)."""
    global _ttft_chars_saved_total
    _ttft_history.clear()
    _ttft_chars_saved_total = 0
    return {"status": "ok"}


@app.post("/reorder")
async def reorder(request: ReorderRequest):
    """
    Reorder contexts for optimal prefix sharing.

    **This is the primary endpoint.** It auto-dispatches based on server mode:

    - **Stateless mode** (``--stateless``): One-shot clustering and scheduling.
      Each call is independent — no state is kept between calls.
    - **Stateful mode** (default): Builds or incrementally updates a live index
      that tracks cached state across calls.  Supports multi-turn deduplication
      via ``deduplicate=True``.

    Supports two input formats:
    - ``List[List[int]]``: Each context is a list of document IDs.
    - ``List[List[str]]``: Each context is a list of document contents.
      Identical strings are treated as the same document.

    **Response (always present)**:
    - ``reordered_contexts``: Contexts with documents rearranged for maximum
      prefix cache sharing, in the optimal execution order.
    - ``original_indices``: List of original context indices so that
      ``reordered_contexts[i]`` corresponds to ``contexts[original_indices[i]]``.

    **Response (stateful mode only)**:
    - ``request_ids``: Ordered list matching input contexts.
    - ``deduplication``: Present when ``deduplicate=True``.
    """
    global _index

    _init_config()

    if _stateless_mode:
        return await _reorder_stateless(request)
    else:
        return await _reorder_stateful(request)


# ── internal helpers ─────────────────────────────────────────────────────────


async def _reorder_stateless(request: ReorderRequest):
    """Stateless reorder: one-shot clustering + scheduling, no state."""
    try:
        logger.info(f"Reordering {len(request.contexts)} contexts (stateless)...")

        contexts = request.contexts
        str_to_id = {}
        id_to_str = {}
        is_string_input = False

        if contexts and contexts[0] and isinstance(contexts[0][0], str):
            is_string_input = True
            logger.info("Detected string input, converting to integer IDs...")

            next_id = 0
            converted_contexts = []
            for ctx in contexts:
                converted_ctx = []
                for item in ctx:
                    sid = str_to_id.get(item)
                    if sid is None:
                        sid = next_id
                        str_to_id[item] = sid
                        id_to_str[sid] = item
                        next_id += 1
                    converted_ctx.append(sid)
                converted_contexts.append(converted_ctx)

            contexts = converted_contexts
            logger.info(f"Converted {len(str_to_id)} unique strings to IDs")

        temp_index = ContextPilot(
            alpha=request.alpha,
            use_gpu=request.use_gpu,
            linkage_method=request.linkage_method,
        )

        result = temp_index.schedule_only(contexts=contexts)

        scheduled_contexts = result["reordered_contexts"]
        if is_string_input:
            scheduled_contexts = [
                [id_to_str[item_id] for item_id in ctx] for ctx in scheduled_contexts
            ]

        logger.info(
            f"Reordered: {len(result['groups'])} groups, "
            f"{len(request.contexts)} contexts"
        )

        return {
            "status": "success",
            "message": "Contexts reordered successfully (stateless mode)",
            "mode": "stateless",
            "input_type": "string" if is_string_input else "integer",
            "num_contexts": len(request.contexts),
            "num_groups": len(result["groups"]),
            "reordered_contexts": scheduled_contexts,
            "original_indices": result["original_indices"],
            # Legacy aliases
            "scheduled_contexts": scheduled_contexts,
            "groups": result["groups"],
            "stats": result.get("stats", {}),
        }

    except Exception as e:
        logger.error(f"Error reordering (stateless): {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _reorder_stateful(request: ReorderRequest):
    """Stateful reorder: build/update live index, track cache state."""
    global _index

    try:
        global _str_to_id, _id_to_str, _next_str_id
        contexts = request.contexts
        is_string_input = False

        if contexts and contexts[0] and isinstance(contexts[0][0], str):
            is_string_input = True
            logger.info("Detected string input, converting to integer IDs...")
            converted_contexts = []
            for ctx in contexts:
                converted_ctx = []
                for item in ctx:
                    sid = _str_to_id.get(item)
                    if sid is None:
                        sid = _next_str_id
                        _str_to_id[item] = sid
                        _id_to_str[sid] = item
                        _next_str_id += 1
                    converted_ctx.append(sid)
                converted_contexts.append(converted_ctx)
            contexts = converted_contexts
            logger.info(f"Converted to IDs ({_next_str_id} unique strings total)")

        def _to_output(reordered):
            if is_string_input and reordered:
                return [[_id_to_str[i] for i in ctx] for ctx in reordered]
            return reordered

        # ── Incremental update ───────────────────────────────────────────
        if _index is not None and _index.is_live:
            logger.info(f"Incremental reorder with {len(contexts)} contexts...")

            result = _index.build_incremental(
                contexts=contexts,
                initial_tokens_per_context=request.initial_tokens_per_context,
            )

            logger.info(
                f"Incremental: {result['matched_count']} matched+inserted, "
                f"{result['merged_count']} built+merged"
            )

            dedup_results = None
            if request.deduplicate:
                tracker = get_conversation_tracker()
                docs_list = result.get("reordered_contexts") or contexts
                doc_contents_list = None
                if _id_to_str:
                    doc_contents_list = [
                        {did: _id_to_str[did] for did in ctx if did in _id_to_str}
                        for ctx in docs_list
                    ]
                dedup_results = tracker.deduplicate_batch(
                    request_ids=result["request_ids"],
                    docs_list=docs_list,
                    parent_request_ids=request.parent_request_ids,
                    hint_template=request.hint_template,
                    doc_contents_list=doc_contents_list,
                )
                if doc_contents_list:
                    for dc in doc_contents_list:
                        for did, content in dc.items():
                            if did in _id_to_str and content != _id_to_str[did]:
                                _id_to_str[did] = content
                logger.info(f"Deduplication: processed {len(dedup_results)} contexts")

            reordered = _to_output(result.get("reordered_contexts"))
            response = {
                "status": "success",
                "message": "Incremental reorder completed",
                "mode": "incremental",
                "input_type": "string" if is_string_input else "integer",
                "num_contexts": len(contexts),
                "matched_count": result["matched_count"],
                "merged_count": result["merged_count"],
                "request_ids": result["request_ids"],
                "reordered_contexts": reordered,
                "original_indices": result["original_indices"],
                "groups": result["groups"],
                "stats": _index.get_stats(),
            }

            if dedup_results:
                response["deduplication"] = {
                    "enabled": True,
                    "results": [
                        {
                            "request_id": result["request_ids"][i],
                            "original_docs": r.original_docs,
                            "deduplicated_docs": r.deduplicated_docs,
                            "overlapping_docs": r.overlapping_docs,
                            "new_docs": r.new_docs,
                            "reference_hints": r.reference_hints,
                            "blocks_deduped": r.blocks_deduped,
                            "blocks_total": r.blocks_total,
                            "block_chars_saved": r.block_chars_saved,
                        }
                        for i, r in enumerate(dedup_results)
                    ],
                    "total_docs_deduplicated": sum(
                        len(r.overlapping_docs) for r in dedup_results
                    ),
                    "total_blocks_deduped": sum(
                        r.blocks_deduped for r in dedup_results
                    ),
                }

            return response

        # ── Initial build ────────────────────────────────────────────────
        logger.info(f"Building index with {len(contexts)} contexts...")

        _index = ContextPilot(
            alpha=request.alpha,
            use_gpu=request.use_gpu,
            linkage_method=request.linkage_method,
        )

        result = _index.build_and_schedule(
            contexts=contexts,
            initial_tokens_per_context=request.initial_tokens_per_context,
        )

        request_id_mapping = result.get("request_id_mapping", {})
        request_ids = result.get("request_ids", [])

        logger.info(f"Index built. Auto-assigned {len(request_id_mapping)} request IDs")

        dedup_results = None
        if request.deduplicate:
            tracker = get_conversation_tracker()
            reordered_raw = result.get("reordered_contexts") or contexts
            doc_contents_list = None
            if _id_to_str:
                doc_contents_list = [
                    {did: _id_to_str[did] for did in ctx if did in _id_to_str}
                    for ctx in reordered_raw
                ]
            dedup_results = tracker.deduplicate_batch(
                request_ids=request_ids,
                docs_list=reordered_raw,
                parent_request_ids=request.parent_request_ids,
                hint_template=request.hint_template,
                doc_contents_list=doc_contents_list,
            )
            if doc_contents_list:
                for dc in doc_contents_list:
                    for did, content in dc.items():
                        if did in _id_to_str and content != _id_to_str[did]:
                            _id_to_str[did] = content
            logger.info(f"Deduplication: processed {len(dedup_results)} contexts")

        reordered = _to_output(result.get("reordered_contexts", contexts))
        order = result.get("original_indices", list(range(len(contexts))))
        response = {
            "status": "success",
            "message": "Index built successfully",
            "mode": "initial",
            "input_type": "string" if is_string_input else "integer",
            "num_contexts": len(contexts),
            "matched_count": 0,
            "inserted_count": len(contexts),
            "request_id_mapping": request_id_mapping,
            "request_ids": request_ids,
            "reordered_contexts": reordered,
            "original_indices": order,
            "stats": _index.get_stats(),
        }

        if dedup_results:
            response["deduplication"] = {
                "enabled": True,
                "results": [
                    {
                        "request_id": request_ids[i],
                        "original_docs": r.original_docs,
                        "deduplicated_docs": r.deduplicated_docs,
                        "overlapping_docs": r.overlapping_docs,
                        "new_docs": r.new_docs,
                        "reference_hints": r.reference_hints,
                        "blocks_deduped": r.blocks_deduped,
                        "blocks_total": r.blocks_total,
                        "block_chars_saved": r.block_chars_saved,
                    }
                    for i, r in enumerate(dedup_results)
                ],
                "total_docs_deduplicated": sum(
                    len(r.overlapping_docs) for r in dedup_results
                ),
                "total_blocks_deduped": sum(r.blocks_deduped for r in dedup_results),
            }

        return response

    except Exception as e:
        logger.error(f"Error reordering (stateful): {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Legacy aliases (deprecated, use /reorder) ────────────────────────────────


@app.post("/build", deprecated=True)
async def build_index(request: BuildIndexRequest):
    """Deprecated — use POST /reorder instead. Kept for backward compatibility."""
    unified = ReorderRequest(
        contexts=request.contexts,
        alpha=request.alpha,
        use_gpu=request.use_gpu,
        linkage_method=request.linkage_method,
        initial_tokens_per_context=request.initial_tokens_per_context,
        parent_request_ids=request.parent_request_ids,
        deduplicate=request.deduplicate,
        hint_template=request.hint_template,
    )
    return await reorder(unified)


@app.post("/schedule", deprecated=True)
async def schedule_batch(request: ScheduleRequest):
    """Deprecated — use POST /reorder instead. Kept for backward compatibility."""
    unified = ReorderRequest(
        contexts=request.contexts,
        alpha=request.alpha,
        use_gpu=request.use_gpu,
        linkage_method=request.linkage_method,
        initial_tokens_per_context=0,
        parent_request_ids=None,
        deduplicate=False,
        hint_template=None,
    )
    # Force stateless behaviour regardless of server mode
    return await _reorder_stateless(unified)


@app.post("/evict")
async def evict(request: EvictRequest):
    """
    Remove requests from the index (eviction callback integration).

    THIS IS THE MAIN ENDPOINT THAT THE INFERENCE ENGINE'S EVICTION CALLBACK SHOULD CALL.

    When the inference engine's cache evicts entries, it collects the request_ids
    from the evicted entries and invokes the registered callback. That callback
    calls this endpoint to remove the corresponding entries from ContextPilot.

    Supported engines (via zero-patch runtime hooks):
        - SGLang: contextpilot/_sglang_hook.py
        - vLLM:   contextpilot/_vllm_hook.py

    Both use the same protocol:
        POST /evict  {"request_ids": ["req-1", "req-2", ...]}
    """
    # Check if index is initialized
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /reorder first."
        )

    try:
        logger.debug(f"Eviction incoming IDs: {request.request_ids}")
        normalized_ids = [_normalize_request_id(rid) for rid in request.request_ids]
        normalized_ids = [
            rid for rid in normalized_ids if rid and not rid.startswith("HEALTH_CHECK")
        ]
        # Deduplicate while preserving order for deterministic logs/responses.
        normalized_ids = list(dict.fromkeys(normalized_ids))

        # Remove the evicted requests from our index
        result = _index.remove_requests(set(normalized_ids))

        # Also clear conversation history for evicted requests
        # This ensures ConversationTracker stays in sync with the engine's cache
        tracker = get_conversation_tracker()
        conversations_cleared = 0
        for req_id in normalized_ids:
            cleared = tracker.clear_conversation(req_id)
            conversations_cleared += cleared

        # Log eviction details
        logger.info(
            f"Eviction: removed {result['removed_count']} requests from index, "
            f"cleared {conversations_cleared} conversation entries, "
            f"not_found={len(result['not_found'])}, "
            f"incoming={len(request.request_ids)}, normalized={len(normalized_ids)}"
        )

        return {
            "status": "success",
            "conversations_cleared": conversations_cleared,
            "normalized_request_ids": normalized_ids,
            **result,
        }

    except Exception as e:
        logger.error(f"Error during eviction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_index():
    """
    Reset the index to initial state.

    Clears all nodes, metadata, request tracking, conversation history,
    and string-to-ID mappings.
    Use this to start fresh without restarting the server.

    After reset, you must call /reorder again before other operations.
    """
    global _index, _str_to_id, _id_to_str, _next_str_id, _intercept_index, _intercept_states

    # Reset conversation tracker
    reset_conversation_tracker()

    _intercept_states.clear()
    _intercept_index = None

    # Reset string-to-ID mapping
    _str_to_id = {}
    _id_to_str = {}
    _next_str_id = 0

    if _index is None:
        return {
            "status": "success",
            "message": "No index to reset (was not initialized)",
            "conversation_tracker": "reset",
        }

    try:
        _index.reset()
        logger.info(
            "Index, conversation tracker, and string mappings reset successfully"
        )

        return {
            "status": "success",
            "message": "Index reset to initial state",
            "conversation_tracker": "reset",
        }

    except Exception as e:
        logger.error(f"Error resetting index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(request: SearchRequest):
    """Search for a context in the index."""
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /reorder first."
        )

    try:
        search_path, node_id, prefix_length, has_prefix = _index.search(
            context=request.context, update_access=request.update_access
        )

        return {
            "status": "success",
            "search_path": search_path,
            "node_id": node_id,
            "prefix_length": prefix_length,
            "has_prefix": has_prefix,
        }

    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insert")
async def insert_context(request: InsertRequest):
    """
    Insert a new context into the index.

    Auto-generates a unique request_id for the new leaf node.
    The response includes the request_id that should be passed to the inference engine
    so that the engine can track it in its cache for eviction notifications.
    """
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /reorder first."
        )

    try:
        node_id, search_path, request_id = _index.insert(
            context=request.context,
            search_path=request.search_path,
            total_tokens=request.total_tokens,
        )

        return {
            "status": "success",
            "node_id": node_id,
            "search_path": search_path,
            "request_id": request_id,  # Pass this to inference engine for cache tracking
        }

    except Exception as e:
        logger.error(f"Error during insertion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/requests")
async def get_requests():
    """
    Get all tracked request IDs.

    Returns the list of all request_ids currently in the index.
    """
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /reorder first."
        )

    try:
        request_ids = list(_index.get_all_request_ids())

        return {
            "status": "success",
            "num_requests": len(request_ids),
            "request_ids": request_ids,
        }

    except Exception as e:
        logger.error(f"Error getting requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get index statistics."""
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /reorder first."
        )

    try:
        stats = _index.get_stats()

        return {
            "status": "success",
            "index_stats": stats,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Chat Template Helper
# ============================================================================


def apply_chat_template(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Apply chat template to a prompt using the configured tokenizer.

    Args:
        prompt: The user's message/prompt text
        system_prompt: Optional system prompt to prepend

    Returns:
        The formatted prompt string with chat template applied,
        or the original prompt if no tokenizer is configured.
    """
    if _tokenizer is None:
        return prompt

    # Build messages list
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        # Apply chat template with tokenize=False (return string, not tokens)
        # and add_generation_prompt=True (append the assistant prompt prefix)
        formatted = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return formatted
    except Exception as e:
        logger.warning(f"Failed to apply chat template: {e}. Using raw prompt.")
        return prompt


# ============================================================================
# Inference Engine Proxy Endpoints
# ============================================================================


@app.post("/v1/completions")
async def proxy_completions(request: Request):
    """
    Proxy /v1/completions to the inference engine and auto-update tokens.

    This endpoint:
    1. Forwards the request to the inference engine backend
    2. Tracks token usage from the response
    3. Automatically updates the eviction heap
    4. Returns the response to the client

    To associate a request with a context, include 'request_id' in the request body.
    """
    # Ensure config is loaded
    if _infer_api_url is None:
        _init_config()

    infer_api_url = _infer_api_url or os.environ.get(
        "CONTEXTPILOT_INFER_API_URL", "http://localhost:30000"
    )

    if not infer_api_url:
        raise HTTPException(
            status_code=503,
            detail="Inference API URL not configured. Set CONTEXTPILOT_INFER_API_URL env var or use --infer-api-url.",
        )

    session = _aiohttp_session
    if session is None:
        raise HTTPException(status_code=503, detail="HTTP session not initialized")

    try:
        # Parse request body
        body = await request.json()

        # Check for request_id (from manual calls) or rid (from RAGPipeline)
        # RAGPipeline sends 'rid' directly, manual calls may use 'request_id'
        request_id = body.pop("request_id", None) or body.get("rid", None)

        if not request_id:
            request_id = f"req-{uuid.uuid4().hex[:12]}"
            logger.debug(f"Auto-assigned request_id={request_id}")

        # Apply chat template if explicitly requested (default False - template should be applied at prompt generation)
        apply_template = body.pop("apply_chat_template", False)  # Default to False
        system_prompt = body.pop("system_prompt", None)  # Optional system prompt

        if apply_template and _tokenizer is not None and "prompt" in body:
            original_prompt = body["prompt"]
            body["prompt"] = apply_chat_template(original_prompt, system_prompt)
            logger.debug("Applied chat template to prompt")

        # Ensure request_id is tracked for eviction (even if no context node)
        if _index:
            _index.track_request(request_id)

        # Pass request_id to inference engine so it can use the same ID for request tracking
        # Engine will notify ContextPilot via /evict callback when this request is evicted
        if request_id:
            body["rid"] = request_id  # SGLang
            body["request_id"] = request_id  # vLLM
            logger.info(f"Proxy: forwarding request with request_id={request_id}")
        else:
            logger.info(
                "Proxy: forwarding request without rid (no ContextPilot tracking)"
            )

        # Forward to inference engine
        api_url = f"{infer_api_url}/v1/completions"
        logger.debug(f"Proxying to {api_url}")

        async with session.post(api_url, json=body) as response:
            result = await response.json()

            # Token tracking is handled by the inference engine via CONTEXTPILOT_INDEX_URL
            # The engine calls /evict after its internal cache eviction

            # Add request_id to response header (not body, to avoid
            # breaking strict API response parsers).
            cp_headers = {}
            if request_id and response.status == 200:
                usage = result.get("usage", {})
                cp_headers["X-ContextPilot-Result"] = json.dumps(
                    {
                        "request_id": request_id,
                        "tokens_reported": usage.get("total_tokens", 0),
                    }
                )

            return JSONResponse(
                content=result, status_code=response.status, headers=cp_headers
            )

    except aiohttp.ClientError as e:
        logger.error(f"Error proxying to inference engine: {e}")
        raise HTTPException(
            status_code=502, detail=f"Inference engine backend error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HTTP Intercept Proxy Endpoints
# ============================================================================


# ── Conversation-aware helpers ─────────────────────────────────────────────


def _hash_text(text: str) -> str:
    """Fast 16-hex-char hash for content comparison."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _session_fingerprint(body: Dict[str, Any]) -> str:
    """Stable session key from system prompt + first user message."""
    msgs = body.get("messages") or []
    parts_to_hash: list[str] = []

    system = body.get("system")
    if system:
        parts_to_hash.append(str(system)[:500])

    for msg in msgs[:5]:
        if isinstance(msg, dict) and msg.get("role") == "system":
            parts_to_hash.append(str(msg.get("content", ""))[:500])
        elif isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # OpenAI format: [{type: text, text: "..."}]
                text_parts = [p.get("text", "") for p in content
                              if isinstance(p, dict)]
                content = "".join(text_parts)
            parts_to_hash.append(str(content))
            break

    if not parts_to_hash:
        return _hash_text(json.dumps(msgs[:2], sort_keys=True))

    return _hash_text("\x00".join(parts_to_hash))


async def _get_intercept_state(body: Dict[str, Any]) -> _InterceptConvState:
    """Return per-session intercept state, creating or resetting as needed.

    If the message count drops, the host compacted old messages or a new
    session started — either way, reset: the old KV cache entries are gone.
    """
    session_key = _session_fingerprint(body)
    msg_count = len(body.get("messages") or [])

    async with _intercept_states_lock:
        state = _intercept_states.get(session_key)

        if state is None:
            state = _InterceptConvState()
            state.system_processed = True
            logger.info(
                f"Intercept: new session {session_key[:8]}… "
                f"({msg_count} msgs, {len(_intercept_states)} active sessions)"
            )
            if len(_intercept_states) >= _MAX_TRACKED_SESSIONS:
                oldest_key = next(iter(_intercept_states))
                del _intercept_states[oldest_key]
                logger.info(f"Intercept: evicted session {oldest_key[:8]}…")
            _intercept_states[session_key] = state
        elif msg_count < state.last_message_count:
            logger.info(
                f"Intercept: session {session_key[:8]}… message count dropped "
                f"({msg_count} < {state.last_message_count}), "
                f"resetting state (compaction or restart)"
            )
            state = _InterceptConvState()
            state.system_processed = True
            _intercept_states[session_key] = state

    state.last_message_count = msg_count
    return state


def _deduplicate_docs(docs: List[str], state: _InterceptConvState) -> tuple:
    """Remove documents already seen in previous tool results.

    Returns (new_docs, deduped_count).  Also registers all doc hashes
    (including duplicates) in state so future calls can dedup against them.
    """
    new_docs = []
    deduped_count = 0
    for doc in docs:
        h = _hash_text(doc)
        if h in state.seen_doc_hashes:
            deduped_count += 1
        else:
            new_docs.append(doc)
        state.seen_doc_hashes.add(h)
    return new_docs, deduped_count


# Regex for OpenClaw's EXTERNAL_UNTRUSTED_CONTENT security markers.
# These contain a random hex id that changes every request, preventing
# KV cache prefix sharing for identical content.
_EXTERNAL_MARKER_RE = re.compile(
    r'<<<((?:END_)?)EXTERNAL_UNTRUSTED_CONTENT\s+id=\\?"[0-9a-f]+\\?">>>'
)


def _strip_external_content_ids(body: Any) -> Any:
    """Remove random ids from EXTERNAL_UNTRUSTED_CONTENT markers in the body.

    Walks the body dict/list and applies the regex on every string value,
    turning ``<<<EXTERNAL_UNTRUSTED_CONTENT id="ab12cd34">>>`` into
    ``<<<EXTERNAL_UNTRUSTED_CONTENT>>>``.
    """
    if isinstance(body, str):
        return _EXTERNAL_MARKER_RE.sub(
            lambda m: f"<<<{m.group(1) or ''}EXTERNAL_UNTRUSTED_CONTENT>>>", body
        )
    if isinstance(body, dict):
        return {k: _strip_external_content_ids(v) for k, v in body.items()}
    if isinstance(body, list):
        return [_strip_external_content_ids(v) for v in body]
    return body


# API format constants
_OPENAI_CHAT = "openai_chat"
_ANTHROPIC_MESSAGES = "anthropic_messages"

_HOP_BY_HOP = frozenset((
    "host", "connection", "keep-alive", "transfer-encoding",
    "te", "trailer", "upgrade", "proxy-authorization",
    "proxy-authenticate", "content-length",
))


def _doc_preview(doc: str, max_len: int = 60) -> str:
    """Truncate a document string for log preview."""
    doc = doc.replace("\n", " ").strip()
    return doc[:max_len] + "…" if len(doc) > max_len else doc


def _reorder_documents(docs: List[str], config: InterceptConfig) -> tuple:
    """Reorder a list of document strings via the persistent intercept index.

    All documents form ONE context (a single list of doc strings).
    The first call (``_intercept_index is None``) builds the index —
    since there is only one context the order stays unchanged.
    Subsequent calls use ``build_incremental`` which searches the
    existing tree and reorders documents for prefix sharing with
    previously cached state.

    Returns (reordered_docs, original_order, reordered_order) where the
    order lists are 0-based indices suitable for logging/headers.
    """
    global _intercept_index

    contexts = [docs]  # All docs = one context
    original_order = list(range(len(docs)))

    if _intercept_index is None:
        # First call — build index only.  1 context → no reorder possible.
        _intercept_index = ContextPilot(
            alpha=config.alpha,
            use_gpu=False,
            linkage_method=config.linkage_method,
        )
        _intercept_index.build_and_schedule(contexts=cast(List[List[int]], contexts))
        logger.debug("Intercept index initialised (no reorder on first call)")
        return docs, original_order, original_order

    # Subsequent calls — search existing tree and reorder for prefix sharing.
    # build_incremental returns reordered_contexts with strings converted.
    result = _intercept_index.build_incremental(
        contexts=cast(List[List[int]], contexts)
    )
    reordered = result.get("reordered_contexts", [docs])[0]

    # Build order mapping: find where each reordered doc was in the original.
    doc_to_orig = {}
    for i, doc in enumerate(docs):
        doc_to_orig.setdefault(doc, []).append(i)
    reordered_order = []
    used = set()
    for doc in reordered:
        for idx in doc_to_orig.get(doc, []):
            if idx not in used:
                reordered_order.append(idx)
                used.add(idx)
                break
    reordered_docs = [docs[i] for i in reordered_order]

    if logger.isEnabledFor(logging.DEBUG):
        for label, order in [("BEFORE", original_order), ("AFTER", reordered_order)]:
            previews = [f"  [{i}] {_doc_preview(docs[i])}" for i in order]
            logger.debug(f"Reorder {label}:\n" + "\n".join(previews))

    return reordered_docs, original_order, reordered_order


async def _intercept_and_forward(request: Request, api_format: str):
    """Intercept an LLM API request, reorder documents, and forward.

    1. Parse X-ContextPilot-* headers → InterceptConfig
    2. Extract documents from system message/prompt and tool_results
    3. Reorder each extraction via ContextPilot clustering
    4. Reconstruct request body with reordered docs
    5. Forward to actual LLM backend, streaming or not
    If extraction fails at any step → forward original request unmodified.
    """
    if _infer_api_url is None:
        _init_config()

    infer_api_url = _infer_api_url or os.environ.get(
        "CONTEXTPILOT_INFER_API_URL", "http://localhost:30000"
    )
    if not infer_api_url:
        raise HTTPException(
            status_code=503,
            detail="Inference API URL not configured.",
        )

    session = _aiohttp_session
    if session is None:
        raise HTTPException(status_code=503, detail="HTTP session not initialized")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Strip random IDs from OpenClaw's EXTERNAL_UNTRUSTED_CONTENT markers
    # early, so extraction/clustering sees deterministic content and
    # identical documents share the same KV cache prefix.
    body = _strip_external_content_ids(body)

    # Parse intercept config from headers
    headers = dict(request.headers)
    config = parse_intercept_headers(headers)
    total_reordered = 0
    total_deduped = 0
    total_slimmed = 0
    chars_before_slim = 0
    chars_after_slim = 0
    system_count = 0
    tool_result_count = 0
    reorder_details = []
    _dedup_result = DedupResult()

    # ── Debug: log conversation shape, divergence, and tool_result details ──
    _debug_messages = body.get("messages") or []
    _debug_msg_count = len(_debug_messages)

    # Per-message hashes for this request
    _debug_msg_hashes = []
    for m in _debug_messages:
        h = hashlib.sha256(
            json.dumps(m, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:12]
        _debug_msg_hashes.append(h)

    # Build tool_call_id → function name mapping from assistant messages
    _tool_call_names = {}
    for m in _debug_messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                _tc_id = tc.get("id", "")
                _fn = (tc.get("function") or {}).get("name", "?")
                _args_raw = (tc.get("function") or {}).get("arguments", "")
                # Extract file path for read calls
                _path_hint = ""
                if _fn == "read" and isinstance(_args_raw, str):
                    try:
                        _args = json.loads(_args_raw)
                        _p = _args.get("path") or _args.get("file_path") or ""
                        if _p:
                            _path_hint = f" path={_p.split('/')[-1]}"
                    except Exception:
                        pass
                _tool_call_names[_tc_id] = f"{_fn}{_path_hint}"

    # Log all tool_result messages with size, function name, and content preview
    for idx, m in enumerate(_debug_messages):
        _role = m.get("role", "?")
        if _role in ("tool", "toolResult"):
            _tc_id = m.get("tool_call_id", "?")
            _fn_label = _tool_call_names.get(_tc_id, "?")
            _content = m.get("content", "")
            _content_str = str(_content)
            _chars = len(_content_str)
            _is_compacted = "[compacted:" in _content_str
            _preview = _content_str[:150].replace("\n", "\\n")
            logger.debug(
                f"  msg[{idx}] role={_role} fn={_fn_label} "
                f"tool_call_id={_tc_id} "
                f"chars={_chars} compacted={_is_compacted} "
                f"preview: {_preview}"
            )
        elif _role == "user" and isinstance(m.get("content"), list):
            for bi, block in enumerate(m["content"]):
                if isinstance(block, dict) and block.get("type") in (
                    "tool_result",
                    "toolResult",
                ):
                    _tu_id = block.get("tool_use_id", "?")
                    _tc = block.get("content", "")
                    _tc_str = str(_tc)
                    _chars = len(_tc_str)
                    _is_compacted = "[compacted:" in _tc_str
                    _preview = _tc_str[:150].replace("\n", "\\n")
                    logger.debug(
                        f"  msg[{idx}].content[{bi}] type=tool_result "
                        f"tool_use_id={_tu_id} chars={_chars} "
                        f"compacted={_is_compacted} preview: {_preview}"
                    )

    logger.info(
        f"Intercept: session={_session_fingerprint(body)[:8]} "
        f"{_debug_msg_count} msgs"
    )

    # ── Format handler (strategy pattern) ────────────────────────────
    handler = get_format_handler(api_format)

    if config.enabled:
        try:
            body = copy.deepcopy(body)

            # ── Conversation-aware state (single-conversation model) ──
            state = await _get_intercept_state(body)

            # ── Replace old messages with cached (modified) versions ──
            # On subsequent turns, the host sends original (unmodified)
            # messages.  Replace them with our cached modified versions
            # so the inference engine's prefix cache sees identical tokens.
            old_msg_count = len(state.cached_messages)
            if old_msg_count > 0:
                msgs = body.get("messages", [])
                if len(msgs) >= old_msg_count:
                    prefix_ok = True
                    for _ci in range(old_msg_count):
                        _cached_h = hashlib.sha256(
                            json.dumps(state.cached_messages[_ci],
                                       sort_keys=True,
                                       ensure_ascii=False).encode()
                        ).hexdigest()[:16]
                        _current_h = hashlib.sha256(
                            json.dumps(msgs[_ci],
                                       sort_keys=True,
                                       ensure_ascii=False).encode()
                        ).hexdigest()[:16]
                        if _cached_h != _current_h:
                            prefix_ok = False
                            break
                    if prefix_ok:
                        msgs[:old_msg_count] = copy.deepcopy(
                            state.cached_messages)
                        logger.info(
                            f"Intercept: replaced {old_msg_count} old "
                            f"messages with cached versions for prefix "
                            f"cache consistency"
                        )
                    else:
                        logger.info(
                            f"Intercept: prefix mismatch at msg[{_ci}], "
                            f"skipping cached message replay "
                            f"(different session/user)"
                        )
                        old_msg_count = 0
                handler.restore_system(body, state.cached_system)

            multi = handler.extract_all(body, config)

            # ── System prompt: reorder only on first turn ─────────────
            if multi.system_extraction and not state.system_processed:
                extraction, sys_idx = multi.system_extraction
                if len(extraction.documents) >= 2:
                    reordered_docs, orig_order, new_order = _reorder_documents(
                        extraction.documents, config
                    )
                    if orig_order != new_order:
                        reorder_details.append(
                            {
                                "source": "system",
                                "count": len(extraction.documents),
                                "original_order": orig_order,
                                "reordered_order": new_order,
                            }
                        )
                        handler.reconstruct_system(
                            body, extraction, reordered_docs, sys_idx
                        )
                        total_reordered += len(extraction.documents)
                        system_count = 1
                    state.system_processed = True

            # ── Tool results: skip cached old, dedup+reorder new ────────
            for extraction, location in multi.tool_extractions:
                if location.msg_index < old_msg_count:
                    continue
                if len(extraction.documents) < 2:
                    continue

                if not state.first_tool_result_done:
                    # First tool result in session → reorder for KV cache
                    state.first_tool_result_done = True
                    reordered_docs, orig_order, new_order = _reorder_documents(
                        extraction.documents, config
                    )
                    for doc in extraction.documents:
                        state.seen_doc_hashes.add(_hash_text(doc))
                    if orig_order != new_order:
                        reorder_details.append(
                            {
                                "source": f"tool_result[{location.msg_index}]",
                                "count": len(extraction.documents),
                                "original_order": orig_order,
                                "reordered_order": new_order,
                            }
                        )
                        handler.reconstruct_tool_result(
                            body, extraction, reordered_docs, location
                        )
                        total_reordered += len(extraction.documents)
                        tool_result_count += 1
                else:
                    # Subsequent tool results → dedup only
                    new_docs, deduped = _deduplicate_docs(extraction.documents, state)
                    total_deduped += deduped
                    if deduped > 0:
                        if not new_docs:
                            orig_chars = len(extraction.original_content)
                            new_docs = [
                                f"[All {deduped} documents identical to a "
                                f"previous tool result ({orig_chars} chars). "
                                f"Refer to the earlier result above.]"
                            ]
                            chars_before_slim += orig_chars
                            chars_after_slim += len(new_docs[0])
                            total_slimmed += deduped
                        reorder_details.append(
                            {
                                "source": f"tool_result[{location.msg_index}]",
                                "count": len(new_docs),
                                "deduped": deduped,
                            }
                        )
                        handler.reconstruct_tool_result(
                            body, extraction, new_docs, location
                        )
                        tool_result_count += 1

            # ── Single-doc tool results: cross-turn dedup ────────────
            for single_doc, location in multi.single_doc_extractions:
                if location.msg_index < old_msg_count:
                    continue
                if single_doc.content_hash in state.single_doc_hashes:
                    prev_tool_id = state.single_doc_hashes[single_doc.content_hash]
                    if single_doc.tool_call_id == prev_tool_id:
                        logger.debug(
                            f"Intercept: skipping old single-doc at "
                            f"msg[{location.msg_index}] "
                            f"({len(single_doc.content)} chars, "
                            f"preserving prefix cache)"
                        )
                        continue

                    if handler.tool_call_present(body, prev_tool_id):
                        hint = (
                            f"[Duplicate content — identical to a previous "
                            f"tool result ({prev_tool_id}). "
                            f"Refer to the earlier result above.]"
                        )
                        handler.replace_single_doc(body, location, hint)
                        total_deduped += 1
                        logger.debug(
                            f"Intercept: deduped single-doc at msg[{location.msg_index}] "
                            f"(hash={single_doc.content_hash[:12]}…, "
                            f"original={prev_tool_id})"
                        )
                    else:
                        state.single_doc_hashes[single_doc.content_hash] = (
                            single_doc.tool_call_id
                        )
                        logger.debug(
                            f"Intercept: original single-doc ({prev_tool_id}) "
                            f"compacted, keeping re-read at msg[{location.msg_index}]"
                        )
                else:
                    state.single_doc_hashes[single_doc.content_hash] = (
                        single_doc.tool_call_id
                    )

            if total_reordered > 0 or total_deduped > 0 or total_slimmed > 0:
                saved = chars_before_slim - chars_after_slim
                saved_tokens = saved // 4 if saved > 0 else 0
                logger.info(
                    f"Intercept ({api_format}): reordered {total_reordered}, "
                    f"deduped {total_deduped}, slimmed {total_slimmed} "
                    f"(saved {saved:,} chars ≈ {saved_tokens:,} tokens)"
                )

            _dedup_result = DedupResult()
            try:
                if api_format == _OPENAI_CHAT:
                    _dedup_result = dedup_chat_completions(body, chunk_modulus=_chunk_modulus)
                elif "input" in body and isinstance(body.get("input"), list):
                    _dedup_result = dedup_responses_api(body, chunk_modulus=_chunk_modulus)

                if _dedup_result.chars_saved > 0:
                    chars_before_slim += _dedup_result.chars_before
                    chars_after_slim += _dedup_result.chars_after
                    logger.info(
                        f"Dedup ({api_format}): "
                        f"blocks={_dedup_result.blocks_deduped}/{_dedup_result.blocks_total}, "
                        f"saved {_dedup_result.chars_saved:,} chars"
                    )
            except Exception as dedup_err:
                logger.warning(f"Dedup failed, continuing: {dedup_err}")

            # ── Cache the final messages array for next turn ──────────
            state.cached_messages = copy.deepcopy(body.get("messages", []))
            state.cached_system = handler.cache_system(body)

        except Exception as e:
            logger.warning(
                f"Intercept extraction/reorder failed, forwarding original: {e}"
            )
            total_reordered = 0
            total_deduped = 0
            total_slimmed = 0

    # In stateful mode, inject ContextPilot request_id as `rid` so SGLang
    # uses the same ID for cache tracking (enables eviction sync).
    if not _cloud_mode and not _stateless_mode and _index is not None:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        body["rid"] = request_id
        logger.debug(f"Intercept: injected rid={request_id}")

    # ── Cloud proxy mode: inject cache_control + compute content hash ──
    _cloud_content_hash = ""
    _cloud_request_id = ""
    if _cloud_mode and _cloud_adapter is not None and _ttl_policy is not None:
        _ttl_policy.evict_expired()
        cached_hashes = _ttl_policy.get_cached_hashes()
        body = _cloud_adapter.inject_cache_control(body, cached_hashes)
        _cloud_content_hash = hashlib.sha256(
            json.dumps(
                body.get("system", ""), sort_keys=True, ensure_ascii=False
            ).encode()
        ).hexdigest()[:24]
        _cloud_request_id = f"cloud-{uuid.uuid4().hex[:12]}"

    # Determine target URL
    if _cloud_mode and _cloud_adapter is not None:
        target_url = _cloud_adapter.get_api_url(_cloud_adapter.get_target_path())
    else:
        target_url = f"{infer_api_url}{handler.target_path()}"

    if _cloud_mode and _cloud_adapter is not None and _cloud_api_key:
        outbound_headers = _cloud_adapter.get_auth_headers(_cloud_api_key)
    else:
        outbound_headers = {}
        for k, v in headers.items():
            kl = k.lower()
            if kl.startswith("x-contextpilot-"):
                continue
            if kl in _HOP_BY_HOP:
                continue
            outbound_headers[k] = v

    # Build ContextPilot metadata as a response header (not in body,
    # which would break strict API response parsers like OpenClaw's SDK).
    cp_response_headers = {}
    _has_activity = (
        total_reordered > 0
        or total_deduped > 0
        or total_slimmed > 0
        or _dedup_result.chars_saved > 0
    )
    if _has_activity:
        cp_response_headers["X-ContextPilot-Result"] = json.dumps({
            "intercepted": True,
            "documents_reordered": total_reordered > 0,
            "total_documents": total_reordered,
            "documents_deduplicated": total_deduped,
            "documents_slimmed": total_slimmed,
            "chars_before_slim": chars_before_slim,
            "chars_after_slim": chars_after_slim,
            "chars_saved": chars_before_slim - chars_after_slim,
            "message_count": state.last_message_count,
            "sources": {
                "system": system_count,
                "tool_results": tool_result_count,
            },
            "reorder_details": reorder_details,
            "dedup": {
                "blocks_deduped": _dedup_result.blocks_deduped,
                "blocks_total": _dedup_result.blocks_total,
                "chars_saved": _dedup_result.chars_saved,
            },
        })

    is_stream = body.get("stream", False)

    _request_start = time.monotonic()

    try:
        if is_stream:
            # Streaming: passthrough SSE chunks, forwarding status & headers
            async def _stream_with_headers():
                _ttft_logged = False
                async with session.post(
                    target_url, json=body, headers=outbound_headers
                ) as resp:
                    # Collect response headers to forward
                    fwd_headers = dict(cp_response_headers)
                    for k, v in resp.headers.items():
                        kl = k.lower()
                        if kl in _HOP_BY_HOP or kl == "content-length":
                            continue
                        fwd_headers[k] = v
                    # Yield (headers_dict, status) as first item for the wrapper
                    yield resp.status, fwd_headers
                    async for chunk in resp.content.iter_any():
                        if not _ttft_logged:
                            _ttft_ms = (time.monotonic() - _request_start) * 1000
                            _saved = chars_before_slim - chars_after_slim
                            _log_ttft(_ttft_ms, total_slimmed, _saved)
                            _ttft_logged = True
                        yield chunk

            stream_iter = _stream_with_headers()
            first_event = await stream_iter.__anext__()
            status, fwd_headers = cast(tuple[int, Dict[str, str]], first_event)

            async def _stream_content_only():
                async for event in stream_iter:
                    if isinstance(event, bytes):
                        yield event

            return StreamingResponse(
                _stream_content_only(),
                status_code=status,
                headers=fwd_headers,
                media_type=fwd_headers.get("content-type", "text/event-stream"),
            )
        else:
            # Non-streaming: forward JSON with metadata in header only
            async with session.post(
                target_url, json=body, headers=outbound_headers
            ) as resp:
                _ttft_ms = (time.monotonic() - _request_start) * 1000
                _saved = chars_before_slim - chars_after_slim
                _log_ttft(_ttft_ms, total_slimmed, _saved)
                result = await resp.json()

                # ── Cloud mode: track cache metrics from response ──
                if (
                    _cloud_mode
                    and _cloud_adapter is not None
                    and _ttl_policy is not None
                    and _cloud_content_hash
                ):
                    metrics = _cloud_adapter.parse_cache_metrics(result)
                    _ttl_policy.update_from_response(
                        metrics, _cloud_request_id, content_hash=_cloud_content_hash
                    )
                    if (
                        metrics.cache_read_tokens > 0
                        and _index is not None
                        and _cloud_request_id
                    ):
                        node_id = _index._request_to_node.get(_cloud_request_id)
                        if node_id is not None and node_id in _index.metadata:
                            _index.metadata[node_id].update_access_time()
                    cp_response_headers["X-ContextPilot-Cloud-Cache"] = json.dumps(
                        {
                            "provider": _cloud_adapter.provider_name,
                            "cache_creation_tokens": metrics.cache_creation_tokens,
                            "cache_read_tokens": metrics.cache_read_tokens,
                            "ttl_stats": _ttl_policy.get_stats(),
                        }
                    )

                return JSONResponse(
                    content=result,
                    status_code=resp.status,
                    headers=cp_response_headers,
                )

    except aiohttp.ClientError as e:
        logger.error(f"Error forwarding intercepted request: {e}")
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")


@app.post("/v1/chat/completions")
async def intercept_openai_chat(request: Request):
    """Intercept OpenAI chat completions: extract docs, reorder, forward."""
    return await _intercept_and_forward(request, _OPENAI_CHAT)


@app.post("/v1/messages")
async def intercept_anthropic_messages(request: Request):
    """Intercept Anthropic messages: extract docs, reorder, forward."""
    return await _intercept_and_forward(request, _ANTHROPIC_MESSAGES)


@app.get("/cloud/stats")
async def cloud_cache_stats():
    """Get cloud prompt cache statistics (cloud proxy mode only)."""
    if not _cloud_mode or _ttl_policy is None:
        raise HTTPException(
            status_code=404,
            detail="Cloud proxy mode not enabled. Start with --cloud-provider.",
        )
    _ttl_policy.evict_expired()
    stats = _ttl_policy.get_stats()
    stats["provider"] = _cloud_adapter.provider_name if _cloud_adapter else None
    return JSONResponse(content=stats)


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_engine(path: str, request: Request):
    """
    Generic proxy for other /v1/* endpoints.

    Forwards requests to inference engine backend without modification.
    """
    # Ensure config is loaded
    if _infer_api_url is None:
        _init_config()

    infer_api_url = _infer_api_url or os.environ.get(
        "CONTEXTPILOT_INFER_API_URL", "http://localhost:30000"
    )

    if not infer_api_url:
        raise HTTPException(
            status_code=503,
            detail="Inference API URL not configured. Set CONTEXTPILOT_INFER_API_URL env var or use --infer-api-url.",
        )

    session = _aiohttp_session
    if session is None:
        raise HTTPException(status_code=503, detail="HTTP session not initialized")

    try:
        if _cloud_mode and _cloud_adapter is not None:
            target_url = _cloud_adapter.get_api_url(f"/v1/{path}")
            headers = _cloud_adapter.get_auth_headers(_cloud_api_key or "")
        else:
            target_url = f"{infer_api_url}/v1/{path}"
            headers = {}

        if request.method == "GET":
            async with session.get(target_url, headers=headers) as response:
                result = await response.json()
                return JSONResponse(content=result, status_code=response.status)
        else:
            body = await request.json()

            if not _cloud_mode:
                request_id = body.pop("request_id", None) or body.get("rid", None)
                if not request_id:
                    request_id = f"req-{uuid.uuid4().hex[:12]}"
                    logger.debug(f"Auto-assigned request_id={request_id}")
                if _index:
                    _index.track_request(request_id)
                if request_id:
                    body["rid"] = request_id
                    body["request_id"] = request_id

            body.setdefault("temperature", 0)
            if _cloud_mode:
                body.setdefault("top_p", 0)

            dedup_result = DedupResult()
            try:
                if path == "responses" or (
                    "input" in body and isinstance(body.get("input"), list)
                ):
                    # Log function_call_output stats before dedup
                    input_items = body.get("input", [])
                    fco_items = [
                        it
                        for it in input_items
                        if isinstance(it, dict)
                        and it.get("type") == "function_call_output"
                    ]
                    if fco_items:
                        import hashlib as _hl

                        fco_summary = []
                        for it in fco_items:
                            out = it.get("output", "")
                            h = _hl.sha256(
                                out.encode("utf-8", errors="replace")
                            ).hexdigest()[:12]
                            call_id = it.get("call_id", "?")
                            # Find the tool name from function_call items
                            fn_name = "?"
                            for fc in input_items:
                                if (
                                    isinstance(fc, dict)
                                    and fc.get("type") == "function_call"
                                    and fc.get("call_id") == it.get("call_id")
                                ):
                                    fn_name = fc.get("name", "?")
                                    break
                            content_preview = (
                                out[:60].replace("\n", "\\n") if len(out) < 100 else ""
                            )
                            fco_summary.append(
                                f"  call={call_id[:20]} fn={fn_name} len={len(out)} hash={h}"
                                + (f" [{content_preview}]" if content_preview else "")
                            )
                        logger.info(
                            f"Request /v1/{path}: {len(input_items)} items, "
                            f"{len(fco_items)} function_call_output:\n"
                            + "\n".join(fco_summary)
                        )
                    dedup_result = dedup_responses_api(body, chunk_modulus=_chunk_modulus)
                elif "messages" in body and isinstance(body.get("messages"), list):
                    dedup_result = dedup_chat_completions(body, chunk_modulus=_chunk_modulus)
                if dedup_result.chars_saved > 0:
                    logger.info(
                        f"Passthrough dedup /v1/{path}: "
                        f"block={dedup_result.blocks_deduped}/{dedup_result.blocks_total} "
                        f"(saved {dedup_result.chars_saved:,} chars)"
                    )
            except Exception as pe:
                logger.warning(f"Passthrough dedup failed: {pe}")

            response = await session.post(target_url, json=body, headers=headers)
            ct = response.headers.get("content-type", "")
            if "text/event-stream" in ct:

                async def _sse_passthrough():
                    try:
                        async for chunk in response.content.iter_any():
                            yield chunk
                    finally:
                        response.close()

                fwd_hdrs = {
                    k: v
                    for k, v in response.headers.items()
                    if k.lower()
                    not in (
                        "transfer-encoding",
                        "content-encoding",
                        "content-length",
                    )
                }
                return StreamingResponse(
                    _sse_passthrough(),
                    status_code=response.status,
                    headers=fwd_hdrs,
                )
            result = await response.json()
            response.close()
            return JSONResponse(content=result, status_code=response.status)

    except aiohttp.ClientError as e:
        logger.error(f"Error proxying to inference engine: {e}")
        raise HTTPException(
            status_code=502, detail=f"Inference engine backend error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the HTTP server."""
    parser = argparse.ArgumentParser(
        description="ContextPilot Live Index HTTP Server with Inference Engine Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live mode (with inference engine eviction callback):
  python -m contextpilot.server.http_server --port 8765 --infer-api-url http://localhost:30000

  # Stateless mode (just clustering/scheduling, no index maintained):
  python -m contextpilot.server.http_server --port 8765 --stateless --infer-api-url http://localhost:30000

  # Cloud proxy mode (forward to cloud LLM API with prompt cache optimization):
  python -m contextpilot.server.http_server --port 8765 --cloud-provider anthropic --cloud-api-key sk-ant-xxx
  python -m contextpilot.server.http_server --port 8765 --cloud-provider openai --cloud-api-key sk-xxx
  python -m contextpilot.server.http_server --port 8765 --cloud-provider minimax --cloud-api-key xxx

Live mode:
  - Build context index via POST /reorder
  - Receive eviction callbacks from inference engine at POST /evict
  - Engine notifies ContextPilot when requests are evicted from KV cache
  - For SGLang: set CONTEXTPILOT_INDEX_URL=http://localhost:8765

Stateless mode:
  - Use POST /reorder endpoint for one-off batch reordering
  - No index maintained, no eviction tracking
  - Each /reorder call is independent

Cloud proxy mode:
  - Forward to cloud LLM APIs (Anthropic, OpenAI, MiniMax)
  - Automatically inject cache_control for prompt cache optimization
  - Each provider uses its optimal TTL (Anthropic: 5min, OpenAI: 24hr, MiniMax: 5min)
  - GET /cloud/stats for cache hit/miss statistics
        """,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="(Deprecated) No longer required - eviction is now driven by engine callback",
    )
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Run in stateless mode: clustering/reordering only, no index maintained. "
        "Use POST /reorder endpoint for batch reordering.",
    )
    parser.add_argument(
        "--infer-api-url",
        type=str,
        default="http://localhost:30000",
        help="Inference backend URL (default: http://localhost:30000)",
    )
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name/path for chat template tokenizer (e.g., 'Qwen/Qwen3-32B')",
    )
    parser.add_argument(
        "--cloud-provider",
        type=str,
        default=None,
        choices=["anthropic", "openai", "minimax"],
        help="Cloud LLM provider for cloud proxy mode (anthropic/openai/minimax)",
    )
    parser.add_argument(
        "--cloud-api-key",
        type=str,
        default=None,
        help="API key for cloud provider (or set CONTEXTPILOT_CLOUD_API_KEY env var)",
    )
    parser.add_argument(
        "--extended-cache",
        action="store_true",
        default=False,
        help="Use extended cache (Anthropic: 1hr, OpenAI: 24hr, MiniMax: N/A)",
    )
    parser.add_argument(
        "--chunk-modulus",
        type=int,
        default=13,
        help="Content-level dedup block size (avg lines per block). "
        "Smaller = more fine-grained dedup but more pointer overhead. "
        "Larger = fewer blocks but may miss partial overlaps. "
        "Default 13. Range 7-30 recommended.",
    )

    args = parser.parse_args()

    # Set environment variables so they propagate to uvicorn workers
    if args.max_tokens is not None:
        os.environ["CONTEXTPILOT_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["CONTEXTPILOT_INFER_API_URL"] = args.infer_api_url.rstrip("/")
    os.environ["CONTEXTPILOT_STATELESS_MODE"] = "1" if args.stateless else "0"
    if args.model:
        os.environ["CONTEXTPILOT_MODEL_NAME"] = args.model
    if args.cloud_provider:
        os.environ["CONTEXTPILOT_CLOUD_PROVIDER"] = args.cloud_provider
    if args.extended_cache:
        os.environ["CONTEXTPILOT_EXTENDED_CACHE"] = "1"
    if args.cloud_api_key:
        os.environ["CONTEXTPILOT_CLOUD_API_KEY"] = args.cloud_api_key

    # Also set global config for direct access
    global _max_tokens, _infer_api_url, _tokenizer, _model_name, _stateless_mode, _chunk_modulus
    _max_tokens = args.max_tokens
    _infer_api_url = args.infer_api_url.rstrip("/")
    _stateless_mode = args.stateless
    _chunk_modulus = args.chunk_modulus

    # Initialize tokenizer for chat template
    if args.model and AutoTokenizer is not None:
        try:
            _model_name = args.model
            _tokenizer = AutoTokenizer.from_pretrained(_model_name)
            logger.info(f"Loaded tokenizer for chat template: {_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {args.model}: {e}")

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.cloud_provider:
        logger.info(
            f"Starting ContextPilot Index Server on {args.host}:{args.port} (CLOUD PROXY MODE)"
        )
        logger.info(f"Cloud provider: {args.cloud_provider}")
        logger.info("GET /cloud/stats for cache statistics")
    elif _stateless_mode:
        logger.info(
            f"Starting ContextPilot Index Server on {args.host}:{args.port} (STATELESS MODE)"
        )
        logger.info("Stateless mode: clustering/scheduling only, no cache tracking")
        logger.info("Use POST /reorder endpoint for batch reordering")
    else:
        logger.info(
            f"Starting ContextPilot Index Server on {args.host}:{args.port} (LIVE MODE)"
        )
        logger.info("Use POST /reorder endpoint for stateful reordering")
        logger.info("Eviction is driven by engine callback (CONTEXTPILOT_INDEX_URL)")
    if not args.cloud_provider:
        logger.info(f"Inference backend URL: {_infer_api_url}")

    # Run server
    uvicorn.run(
        "contextpilot.server.http_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
