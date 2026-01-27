"""
ContextPilot Live Index HTTP Server

A FastAPI-based HTTP server that:
1. Exposes the LiveContextIndex as a REST API
2. Proxies LLM requests to SGLang backend
3. Automatically tracks tokens and triggers eviction
4. Multi-turn conversation context deduplication

Usage:
    python -m contextpilot.server.http_server --port 8765 --infer-api-url http://localhost:30000

Environment variables (alternative to CLI args):
    RAGBOOST_MAX_TOKENS: Maximum tokens allowed in index
    RAGBOOST_INFER_API_URL: Inference backend URL (default: http://localhost:30000)
"""

import argparse
import logging
import time
import asyncio
import os
import uuid
from typing import List, Dict, Any, Optional
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

from .live_index import LiveContextIndex
from .conversation_tracker import (
    ConversationTracker, 
    DeduplicationResult,
    get_conversation_tracker,
    reset_conversation_tracker
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
_index: Optional[LiveContextIndex] = None
_max_tokens: Optional[int] = None
_infer_api_url: Optional[str] = None
_aiohttp_session: Optional[aiohttp.ClientSession] = None
_tokenizer = None  # AutoTokenizer instance for chat template
_model_name: Optional[str] = None  # Model name for tokenizer
_stateless_mode: bool = False  # Stateless mode: just clustering/scheduling, no cache tracking


def _init_config():
    """Initialize config from environment variables."""
    global _max_tokens, _infer_api_url, _tokenizer, _model_name, _stateless_mode

    # Check stateless mode first
    env_stateless = os.environ.get("RAGBOOST_STATELESS_MODE", "0")
    _stateless_mode = env_stateless == "1"

    if _max_tokens is None and not _stateless_mode:
        env_max_tokens = os.environ.get("RAGBOOST_MAX_TOKENS")
        if env_max_tokens:
            _max_tokens = int(env_max_tokens)

    if _infer_api_url is None:
        _infer_api_url = os.environ.get("RAGBOOST_INFER_API_URL", "http://localhost:30000")

    # Initialize tokenizer for chat template if model is specified
    if _tokenizer is None:
        env_model = os.environ.get("RAGBOOST_MODEL_NAME")
        if env_model and AutoTokenizer is not None:
            try:
                _model_name = env_model
                _tokenizer = AutoTokenizer.from_pretrained(_model_name)
                logger.info(f"Loaded tokenizer for chat template: {_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer for {env_model}: {e}")


# Request/Response Models
class BuildIndexRequest(BaseModel):
    """Request to build the index."""

    contexts: List[List[Any]] = Field(
        ..., description="List of contexts (each is a list of document IDs)"
    )
    initial_tokens_per_context: int = Field(
        0, description="Initial token count per context"
    )
    alpha: float = Field(0.005, description="Distance computation parameter")
    use_gpu: bool = Field(False, description="Use GPU for distance computation")
    linkage_method: str = Field("average", description="Linkage method for clustering")
    incremental: bool = Field(
        False, description="If True and index exists, do incremental build (search, reorder, merge)"
    )
    # Multi-turn deduplication fields
    parent_request_ids: Optional[List[Optional[str]]] = Field(
        None, 
        description="List of parent request IDs for multi-turn deduplication. "
                    "Each element corresponds to a context. None means turn 1 (no parent)."
    )
    deduplicate: bool = Field(
        False,
        description="If True, deduplicate contexts based on conversation history"
    )
    hint_template: Optional[str] = Field(
        None,
        description="Template for reference hints. Use {doc_id} and {turn_number} placeholders."
    )


class ScheduleRequest(BaseModel):
    """Request to schedule a batch (stateless mode - no cache tracking)."""

    contexts: List[List[Any]] = Field(
        ..., description="List of contexts (each is a list of document IDs)"
    )
    alpha: float = Field(0.005, description="Distance computation parameter")
    use_gpu: bool = Field(False, description="Use GPU for distance computation")
    linkage_method: str = Field("average", description="Linkage method for clustering")


class EvictRequest(BaseModel):
    """Request to evict (remove) requests from the index."""

    request_ids: List[str] = Field(..., description="List of request IDs to evict/remove")


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
                    "None means turn 1 (no parent, will be registered for future dedup)."
    )
    hint_template: Optional[str] = Field(
        None,
        description="Template for reference hints. Use {doc_id} and {turn_number} placeholders."
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
    description="HTTP API for ContextPilot LiveContextIndex with SGLang proxy and eviction synchronization",
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
                "message": "Index not initialized. Call POST /build first.",
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


@app.post("/build")
async def build_index(request: BuildIndexRequest):
    """
    Build and initialize the index, or incrementally update existing index.

    Two modes:
    1. Initial build (incremental=False or no existing index):
       - Creates new index from scratch
       - Performs clustering and scheduling
       - Auto-generates request_ids for all contexts
       
    2. Incremental build (incremental=True and index exists):
       - Reorders matched contexts to align with existing prefix structure
       - Builds a separate index for unmatched contexts
       - Merges new index subtrees under the global root
       - Returns NEW request_ids for all contexts

    Multi-turn deduplication (deduplicate=True):
       - Compares contexts against conversation history
       - Returns deduplicated_contexts with overlapping docs removed
       - Returns reference_hints for each deduplicated document

    The response includes an ordered list of request_ids matching input contexts order.
    These request_ids should be used when sending requests to SGLang so that
    the eviction callback can notify ContextPilot which contexts were evicted.
    """
    global _index

    # Ensure config is initialized from env vars
    _init_config()

    try:
        # Check if incremental mode is requested and index exists
        if request.incremental and _index is not None and _index.is_live:
            logger.info(f"Incremental build with {len(request.contexts)} contexts...")
            
            result = _index.build_incremental(
                contexts=request.contexts,
                initial_tokens_per_context=request.initial_tokens_per_context,
            )
            
            logger.info(
                f"Incremental build: {result['matched_count']} matched+inserted, "
                f"{result['merged_count']} built+merged"
            )
            
            # Multi-turn deduplication if requested
            dedup_results = None
            if request.deduplicate:
                tracker = get_conversation_tracker()
                dedup_results = tracker.deduplicate_batch(
                    request_ids=result['request_ids'],
                    docs_list=result.get('reordered_contexts') or request.contexts,
                    parent_request_ids=request.parent_request_ids,
                    hint_template=request.hint_template
                )
                logger.info(f"Deduplication: processed {len(dedup_results)} contexts")
            
            response = {
                "status": "success",
                "message": "Incremental build completed",
                "mode": "incremental",
                "num_contexts": len(request.contexts),
                "matched_count": result['matched_count'],
                "merged_count": result['merged_count'],
                "request_ids": result['request_ids'],
                "reordered_contexts": result.get('reordered_contexts'),
                "scheduled_order": result['scheduled_order'],
                "groups": result['groups'],
                "stats": _index.get_stats(),
            }
            
            # Add deduplication results if requested
            if dedup_results:
                response["deduplication"] = {
                    "enabled": True,
                    "results": [
                        {
                            "request_id": result['request_ids'][i],
                            "original_docs": r.original_docs,
                            "deduplicated_docs": r.deduplicated_docs,
                            "overlapping_docs": r.overlapping_docs,
                            "new_docs": r.new_docs,
                            "reference_hints": r.reference_hints,
                        }
                        for i, r in enumerate(dedup_results)
                    ],
                    "total_docs_deduplicated": sum(len(r.overlapping_docs) for r in dedup_results),
                }
            
            return response
        
        # Initial build
        logger.info(f"Building index with {len(request.contexts)} contexts...")

        _index = LiveContextIndex(
            alpha=request.alpha,
            use_gpu=request.use_gpu,
            linkage_method=request.linkage_method,
        )

        result = _index.build_and_schedule(
            contexts=request.contexts,
            initial_tokens_per_context=request.initial_tokens_per_context,
        )

        # Extract request_id mapping for SGLang integration
        request_id_mapping = result.get("request_id_mapping", {})
        # Ordered list of request_ids matching input contexts order
        request_ids = result.get("request_ids", [])

        logger.info(
            f"Index built successfully. Auto-assigned {len(request_id_mapping)} request IDs"
        )
        
        # Multi-turn deduplication if requested (for initial build too)
        dedup_results = None
        if request.deduplicate:
            tracker = get_conversation_tracker()
            reordered = result.get('scheduled_reordered') or request.contexts
            dedup_results = tracker.deduplicate_batch(
                request_ids=request_ids,
                docs_list=reordered,
                parent_request_ids=request.parent_request_ids,
                hint_template=request.hint_template
            )
            logger.info(f"Deduplication: processed {len(dedup_results)} contexts")

        response = {
            "status": "success",
            "message": "Index built successfully",
            "mode": "initial",
            "num_contexts": len(request.contexts),
            "matched_count": 0,
            "inserted_count": len(request.contexts),
            "request_id_mapping": request_id_mapping,
            "request_ids": request_ids,
            "stats": _index.get_stats(),
        }
        
        # Add deduplication results if requested
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
                    }
                    for i, r in enumerate(dedup_results)
                ],
                "total_docs_deduplicated": sum(len(r.overlapping_docs) for r in dedup_results),
            }
        
        return response

    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deduplicate")
async def deduplicate_contexts(request: DeduplicateRequest):
    """
    Deduplicate contexts for multi-turn conversations without index operations.
    
    This is a lightweight endpoint designed for Turn 2+ in multi-turn conversations.
    It only performs deduplication against conversation history - no index build/search.
    
    Flow:
    1. Turn 1: Client calls /build (no parent_request_id, builds index, registers in tracker)
    2. Turn 2+: Client calls /deduplicate (has parent_request_id, just deduplicates)
    
    For each context:
    - If parent_request_id is None: Registers as a new conversation root (Turn 1)
    - If parent_request_id is provided: Deduplicates against conversation history
    
    Returns deduplicated docs with reference hints for overlapping documents.
    """
    try:
        tracker = get_conversation_tracker()
        
        # Validate input lengths match
        if len(request.contexts) != len(request.parent_request_ids):
            raise HTTPException(
                status_code=400,
                detail=f"contexts and parent_request_ids must have same length "
                       f"(got {len(request.contexts)} vs {len(request.parent_request_ids)})"
            )
        
        # Generate request_ids for this batch
        import uuid
        request_ids = [f"dedup_{uuid.uuid4().hex[:8]}" for _ in request.contexts]
        
        # Perform deduplication
        dedup_results = tracker.deduplicate_batch(
            request_ids=request_ids,
            docs_list=request.contexts,
            parent_request_ids=request.parent_request_ids,
            hint_template=request.hint_template
        )
        
        logger.info(
            f"Deduplicated {len(request.contexts)} contexts, "
            f"removed {sum(len(r.overlapping_docs) for r in dedup_results)} overlapping docs"
        )
        
        return {
            "status": "success",
            "message": "Deduplication completed",
            "request_ids": request_ids,
            "results": [
                {
                    "request_id": request_ids[i],
                    "parent_request_id": request.parent_request_ids[i],
                    "original_docs": r.original_docs,
                    "deduplicated_docs": r.deduplicated_docs,
                    "overlapping_docs": r.overlapping_docs,
                    "new_docs": r.new_docs,
                    "reference_hints": r.reference_hints,
                    "is_new_conversation": r.is_new_conversation,
                }
                for i, r in enumerate(dedup_results)
            ],
            "summary": {
                "total_contexts": len(request.contexts),
                "new_conversations": sum(1 for r in dedup_results if r.is_new_conversation),
                "continued_conversations": sum(1 for r in dedup_results if not r.is_new_conversation),
                "total_docs_deduplicated": sum(len(r.overlapping_docs) for r in dedup_results),
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in deduplication: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/schedule")
async def schedule_batch(request: ScheduleRequest):
    """
    Schedule a batch of contexts (STATELESS MODE).

    This endpoint performs clustering and scheduling WITHOUT tracking cache state.
    Use this when you want to:
    1. Just reorder contexts for optimal prefix sharing
    2. Process batches independently without maintaining server state
    3. Send scheduled contexts directly to LLM engine without cache sync

    No cache tracking, eviction, or token updates required.
    Each call is independent - perfect for batch processing.

    Returns:
        - scheduled_contexts: Reordered contexts for optimal prefix sharing
        - original_indices: Mapping to original context indices
        - groups: Execution groups with prefix sharing info
    """
    try:
        logger.info(f"Scheduling batch with {len(request.contexts)} contexts (stateless mode)...")

        # Create a temporary index just for clustering/scheduling
        temp_index = LiveContextIndex(
            alpha=request.alpha,
            use_gpu=request.use_gpu,
            linkage_method=request.linkage_method,
            max_tokens=None,  # No max_tokens - stateless mode
        )

        # Build and schedule without live tracking
        result = temp_index.schedule_only(
            contexts=request.contexts,
        )

        logger.info(
            f"Batch scheduled: {len(result['groups'])} groups, "
            f"{len(request.contexts)} contexts reordered"
        )

        return {
            "status": "success",
            "message": "Batch scheduled successfully (stateless mode)",
            "mode": "stateless",
            "num_contexts": len(request.contexts),
            "num_groups": len(result["groups"]),
            "scheduled_contexts": result["scheduled_reordered"],
            "original_indices": result["final_mapping"],
            "groups": result["groups"],
            "stats": result.get("stats", {}),
        }

    except Exception as e:
        logger.error(f"Error scheduling batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evict")
async def evict(request: EvictRequest):
    """
    Remove requests from the index (SGLang eviction callback integration).

    THIS IS THE MAIN ENDPOINT THAT SGLANG'S EVICTION CALLBACK SHOULD CALL.

    When SGLang's radix_cache.evict() evicts nodes, it collects the request_ids
    from the evicted nodes and invokes the registered callback. That callback
    should call this endpoint to remove the corresponding entries from ContextPilot.

    Integration in SGLang:
        def eviction_callback(evicted_request_ids: set):
            if evicted_request_ids:
                try:
                    requests.post(
                        "http://localhost:8765/evict",
                        json={"request_ids": list(evicted_request_ids)},
                        timeout=1.0
                    )
                except Exception as e:
                    logger.warning(f"ContextPilot eviction sync failed: {e}")
        
        # Register callback when initializing radix cache
        tree_cache.set_eviction_callback(eviction_callback)
    """
    # Check if index is initialized
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /build first."
        )

    try:
        # Remove the evicted requests from our index
        result = _index.remove_requests(request.request_ids)
        
        # Also clear conversation history for evicted requests
        # This ensures ConversationTracker stays in sync with SGLang's cache
        tracker = get_conversation_tracker()
        conversations_cleared = 0
        for req_id in request.request_ids:
            cleared = tracker.clear_conversation(req_id)
            conversations_cleared += cleared

        # Log eviction details
        logger.info(
            f"Eviction: removed {result['removed_count']} requests from index, "
            f"cleared {conversations_cleared} conversation entries, "
            f"not_found={len(result['not_found'])}"
        )

        return {
            "status": "success",
            "conversations_cleared": conversations_cleared,
            **result,
        }

    except Exception as e:
        logger.error(f"Error during eviction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_index():
    """
    Reset the index to initial state.
    
    Clears all nodes, metadata, request tracking, and conversation history.
    Use this to start fresh without restarting the server.
    
    After reset, you must call /build again before other operations.
    """
    global _index
    
    # Reset conversation tracker
    reset_conversation_tracker()
    
    if _index is None:
        return {
            "status": "success",
            "message": "No index to reset (was not initialized)",
            "conversation_tracker": "reset",
        }
    
    try:
        _index.reset()
        logger.info("Index and conversation tracker reset successfully")
        
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
            status_code=503, detail="Index not initialized. Call POST /build first."
        )

    try:
        search_path, node_id, prefix_length = _index.search(
            context=request.context, update_access=request.update_access
        )

        return {
            "status": "success",
            "search_path": search_path,
            "node_id": node_id,
            "prefix_length": prefix_length,
        }

    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insert")
async def insert_context(request: InsertRequest):
    """
    Insert a new context into the index.

    Auto-generates a unique request_id for the new leaf node.
    The response includes the request_id that should be passed to SGLang
    so that SGLang can track it in its radix cache for eviction notifications.
    """
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /build first."
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
            "request_id": request_id,  # Pass this to SGLang for cache tracking
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
            status_code=503, detail="Index not initialized. Call POST /build first."
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
            status_code=503, detail="Index not initialized. Call POST /build first."
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


@app.post("/reset")
async def reset_index():
    """Reset the index (for debugging/testing). Note: max_tokens remains set from startup."""
    global _index

    try:
        _index = None
        logger.info("Index reset (max_tokens preserved from startup)")

        return {
            "status": "success",
            "message": "Index reset successfully. Call POST /build to reinitialize.",
            "max_tokens": _max_tokens,
        }

    except Exception as e:
        logger.error(f"Error resetting index: {e}")
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
# SGLang Proxy Endpoints
# ============================================================================


@app.post("/v1/completions")
async def proxy_completions(request: Request):
    """
    Proxy /v1/completions to SGLang and auto-update tokens.

    This endpoint:
    1. Forwards the request to SGLang backend
    2. Tracks token usage from the response
    3. Automatically updates the eviction heap
    4. Returns the response to the client

    To associate a request with a context, include 'request_id' in the request body.
    """
    # Ensure config is loaded
    if _infer_api_url is None:
        _init_config()

    infer_api_url = _infer_api_url or os.environ.get(
        "RAGBOOST_INFER_API_URL", "http://localhost:30000"
    )

    if not infer_api_url:
        raise HTTPException(
            status_code=503,
            detail="Inference API URL not configured. Set RAGBOOST_INFER_API_URL env var or use --infer-api-url.",
        )

    try:
        # Parse request body
        body = await request.json()
        
        # Check for request_id (from manual calls) or rid (from RAGPipeline)
        # RAGPipeline sends 'rid' directly, manual calls may use 'request_id'
        request_id = body.pop("request_id", None) or body.get("rid", None)
        
        # NOTE: We do NOT auto-generate request_id anymore.
        # The client should pass request_id from the /build response.
        # If not provided, ContextPilot token tracking is skipped for this request.
        if not request_id:
            logger.debug("No request_id provided, ContextPilot tracking disabled for this request")

        # Apply chat template if explicitly requested (default False - template should be applied at prompt generation)
        apply_template = body.pop("apply_chat_template", False)  # Default to False
        system_prompt = body.pop("system_prompt", None)  # Optional system prompt

        if apply_template and _tokenizer is not None and "prompt" in body:
            original_prompt = body["prompt"]
            body["prompt"] = apply_chat_template(original_prompt, system_prompt)
            logger.debug("Applied chat template to prompt")

        # Verify request_id exists in the index
        # Note: LRU tracking is now handled by SGLang's radix cache, not ContextPilot
        if request_id and _index:
            node = _index.get_request_node(request_id)
            if node is None:  # Use 'is None' because node_id=0 is valid but falsy
                logger.warning(f"Request ID '{request_id}' not found in index")
                request_id = None  # Clear so SGLang won't try to track

        # Pass request_id to SGLang so it can use the same ID for request tracking
        # SGLang will notify ContextPilot via /evict callback when this request is evicted
        # Only set rid if we have a valid request_id AND it's not already in the body
        if request_id:
            body["rid"] = request_id  # Ensure rid is set (may already be there from RAGPipeline)
            logger.info(f"Proxy: forwarding request with rid={request_id}")
        else:
            logger.info("Proxy: forwarding request without rid (no ContextPilot tracking)")

        # Forward to SGLang
        api_url = f"{infer_api_url}/v1/completions"
        logger.debug(f"Proxying to {api_url}")

        async with _aiohttp_session.post(api_url, json=body) as response:
            result = await response.json()

            # Token tracking is handled by SGLang via RAGBOOST_INDEX_URL
            # SGLang calls /update_tokens after request completion
            # SGLang calls /evict after its internal cache eviction
            
            # Add request_id to response for client reference
            if request_id and response.status == 200:
                usage = result.get("usage", {})
                result["_contextpilot"] = {
                    "request_id": request_id,
                    "tokens_reported": usage.get("total_tokens", 0),
                }

            return JSONResponse(content=result, status_code=response.status)

    except aiohttp.ClientError as e:
        logger.error(f"Error proxying to SGLang: {e}")
        raise HTTPException(status_code=502, detail=f"SGLang backend error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_sglang(path: str, request: Request):
    """
    Generic proxy for other SGLang /v1/* endpoints.

    Forwards requests to SGLang backend without modification.
    """
    # Ensure config is loaded
    if _infer_api_url is None:
        _init_config()

    infer_api_url = _infer_api_url or os.environ.get(
        "RAGBOOST_INFER_API_URL", "http://localhost:30000"
    )

    if not infer_api_url:
        raise HTTPException(
            status_code=503,
            detail="Inference API URL not configured. Set RAGBOOST_INFER_API_URL env var or use --infer-api-url.",
        )

    try:
        target_url = f"{infer_api_url}/v1/{path}"

        if request.method == "GET":
            async with _aiohttp_session.get(target_url) as response:
                result = await response.json()
                return JSONResponse(content=result, status_code=response.status)
        else:
            body = await request.json()
            async with _aiohttp_session.post(target_url, json=body) as response:
                result = await response.json()
                return JSONResponse(content=result, status_code=response.status)

    except aiohttp.ClientError as e:
        logger.error(f"Error proxying to SGLang: {e}")
        raise HTTPException(status_code=502, detail=f"SGLang backend error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the HTTP server."""
    parser = argparse.ArgumentParser(
        description="ContextPilot Live Index HTTP Server with SGLang Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live mode (with SGLang eviction callback integration):
  python -m contextpilot.server.http_server --port 8765 --infer-api-url http://localhost:30000

  # Stateless mode (just clustering/scheduling, no index maintained):
  python -m contextpilot.server.http_server --port 8765 --stateless --infer-api-url http://localhost:30000

Live mode:
  - Build context index via POST /build
  - Receive eviction callbacks from SGLang at POST /evict
  - SGLang notifies ContextPilot when requests are evicted from KV cache
  - Start SGLang with: RAGBOOST_INDEX_URL=http://localhost:8765

Stateless mode:
  - Use POST /schedule endpoint for one-off batch reordering
  - No index maintained, no eviction tracking
  - Each /schedule call is independent
        """,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="(Deprecated) No longer required - eviction is now driven by SGLang callback",
    )
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Run in stateless mode: clustering/scheduling only, no index maintained. "
             "Use POST /schedule endpoint for batch reordering.",
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

    args = parser.parse_args()

    # Note: --max-tokens is no longer required since eviction is now driven by
    # SGLang's callback, not by ContextPilot's internal tracking

    # Set environment variables so they propagate to uvicorn workers
    if args.max_tokens is not None:
        os.environ["RAGBOOST_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["RAGBOOST_INFER_API_URL"] = args.infer_api_url.rstrip("/")
    os.environ["RAGBOOST_STATELESS_MODE"] = "1" if args.stateless else "0"
    if args.model:
        os.environ["RAGBOOST_MODEL_NAME"] = args.model

    # Also set global config for direct access
    global _max_tokens, _infer_api_url, _tokenizer, _model_name, _stateless_mode
    _max_tokens = args.max_tokens
    _infer_api_url = args.infer_api_url.rstrip("/")
    _stateless_mode = args.stateless

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

    if _stateless_mode:
        logger.info(f"Starting ContextPilot Index Server on {args.host}:{args.port} (STATELESS MODE)")
        logger.info("Stateless mode: clustering/scheduling only, no cache tracking")
        logger.info("Use POST /schedule endpoint for batch reordering")
    else:
        logger.info(f"Starting ContextPilot Index Server on {args.host}:{args.port} (LIVE MODE)")
        logger.info("Eviction is driven by SGLang callback (RAGBOOST_INDEX_URL)")
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
