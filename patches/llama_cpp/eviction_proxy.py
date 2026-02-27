"""
llama.cpp ContextPilot Eviction Proxy
======================================
Drop-in proxy that sits between ContextPilot and a llama.cpp server, adding
KV-cache eviction sync so the ContextPilot index stays in sync with llama.cpp's
per-slot prefix cache.

Architecture:
  ContextPilot HTTP server  (or direct OpenAI client)
          |
  [This proxy  :8890]   ← also exposes /stats and /reset
          |
  [llama.cpp   :8889]   ← started with --cache-reuse N --parallel K

Eviction protocol (identical to patches/vllm and patches/sglang):
  llama.cpp uses N independent slots (--parallel N).  Each slot holds one
  sequence's KV state.  When slot S transitions from request A → request B,
  A's unique tokens are gone from cache.  The proxy detects this and sends:
      POST <CONTEXTPILOT_INDEX_URL>/evict  {"request_ids": ["req-A"]}

Environment variables:
  LLAMA_SERVER_URL         llama.cpp backend     (default: http://localhost:8889)
  CONTEXTPILOT_INDEX_URL   ContextPilot index server URL (enables eviction sync)
  PROXY_HOST               Bind host             (default: 0.0.0.0)
  PROXY_PORT               Bind port             (default: 8890)
  LOG_FILE                 JSONL log path        (default: query_log.jsonl)
  CHAT_TEMPLATE_FORMAT     "llama3" | "chatml"   (default: llama3)

Usage:
  # 1. Start llama.cpp with prefix caching and multi-slot support:
  llama-server -m model.gguf --port 8889 --cache-reuse 256 --parallel 4

  # 2. Start ContextPilot index server:
  python -m contextpilot.server.http_server --port 8765 \\
      --infer-api-url http://localhost:8890

  # 3. Start this proxy:
  CONTEXTPILOT_INDEX_URL=http://localhost:8765 python eviction_proxy.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

# ─── Configuration ─────────────────────────────────────────────────────────────

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8889").rstrip("/")
CONTEXTPILOT_INDEX_URL = (os.environ.get("CONTEXTPILOT_INDEX_URL") or "").rstrip("/") or None
_contextpilot_enabled = CONTEXTPILOT_INDEX_URL is not None

PROXY_HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8890"))
LOG_FILE = os.environ.get("LOG_FILE", "query_log.jsonl")
MAX_HISTORY = 1000
CHAT_TEMPLATE_FORMAT = os.environ.get("CHAT_TEMPLATE_FORMAT", "llama3").lower()

# ─── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Apple Silicon runtime check ───────────────────────────────────────────────
import platform as _platform
import subprocess as _subprocess

_IS_APPLE_SILICON = (_platform.system() == "Darwin" and _platform.machine() == "arm64")

def _check_powermetrics() -> bool:
    """Return True if powermetrics can run without a password (needed for GPU metrics)."""
    if not _IS_APPLE_SILICON:
        return False
    try:
        r = _subprocess.run(
            ["sudo", "-n", "/usr/bin/powermetrics", "--version"],
            capture_output=True, timeout=3,
        )
        return r.returncode == 0
    except Exception:
        return False

_POWERMETRICS_OK = _check_powermetrics()

# ─── Per-slot KV cache state ───────────────────────────────────────────────────
# Maps slot_id → ContextPilot request_id currently cached in that slot.
# Protected by _slot_lock.  Updated after every /completion response.
_slot_state: dict[int, Optional[str]] = {}
_slot_lock: Optional[asyncio.Lock] = None


# ─── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _slot_lock
    _slot_lock = asyncio.Lock()

    logger.info("llama.cpp ContextPilot Eviction Proxy starting")
    logger.info("  llama.cpp backend:   %s", LLAMA_SERVER_URL)
    logger.info(
        "  ContextPilot sync:   %s",
        CONTEXTPILOT_INDEX_URL if _contextpilot_enabled else "DISABLED",
    )
    logger.info("  Proxy listening:     %s:%d", PROXY_HOST, PROXY_PORT)
    logger.info("  Chat template:       %s", CHAT_TEMPLATE_FORMAT)

    if _IS_APPLE_SILICON:
        if _POWERMETRICS_OK:
            logger.info("  GPU metrics:         enabled (Apple Silicon / Metal)")
        else:
            logger.warning(
                "  GPU metrics:         DISABLED — powermetrics needs passwordless sudo.\n"
                "  Add to /etc/sudoers (run: sudo visudo):\n"
                "    %s ALL=(ALL) NOPASSWD: /usr/bin/powermetrics",
                _platform.node(),
            )
    else:
        logger.info("  GPU metrics:         N/A (not Apple Silicon)")

    yield
    logger.info("Proxy shutting down")


app = FastAPI(title="llama.cpp ContextPilot Eviction Proxy", lifespan=lifespan)
query_history: deque = deque(maxlen=MAX_HISTORY)

# ─── ContextPilot eviction helpers ─────────────────────────────────────────────


async def _fire_eviction(evicted_ids: set[str]) -> None:
    """POST evicted request IDs to ContextPilot index server."""
    if not _contextpilot_enabled or not evicted_ids:
        return

    filtered = {
        rid for rid in evicted_ids
        if rid and not rid.startswith("HEALTH_CHECK")
    }
    if not filtered:
        return

    logger.info(
        "[ContextPilot] Syncing eviction: %d requests [%s%s]",
        len(filtered),
        ", ".join(sorted(filtered)[:10]),
        " ..." if len(filtered) > 10 else "",
    )
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            await client.post(
                f"{CONTEXTPILOT_INDEX_URL}/evict",
                json={"request_ids": list(filtered)},
            )
    except Exception as e:
        logger.warning("[ContextPilot] Eviction sync failed: %s", e)


async def _update_slot(slot_id: int, new_rid: Optional[str]) -> Optional[str]:
    """
    Record that slot_id is now occupied by new_rid.

    Returns the previous occupant's request_id if it differs from new_rid
    (meaning that occupant's KV state has been replaced and should be evicted
    from ContextPilot's index), or None if nothing needs evicting.
    """
    assert _slot_lock is not None
    async with _slot_lock:
        prev_rid = _slot_state.get(slot_id)
        _slot_state[slot_id] = new_rid
        if prev_rid != new_rid:
            return prev_rid  # this request is no longer in this slot's cache
    return None


# ─── GPU Monitoring (Apple Silicon) ────────────────────────────────────────────

try:
    import plistlib
    import subprocess

    def _run_powermetrics(interval_ms: int, n_samples: int) -> list[dict]:
        """Blocking call to powermetrics; returns per-sample GPU metrics."""
        if not _POWERMETRICS_OK:
            return []
        try:
            result = subprocess.run(
                [
                    "sudo", "-n", "/usr/bin/powermetrics",
                    "--samplers", "gpu_power",
                    "-i", str(interval_ms),
                    "-n", str(n_samples),
                    "--format", "plist",
                ],
                capture_output=True,
                timeout=interval_ms * n_samples / 1000 + 3,
            )
            if result.returncode != 0:
                return []

            samples: list[dict] = []
            for rec in [b"<?xml" + p for p in result.stdout.split(b"<?xml") if p.strip()]:
                try:
                    gpu = plistlib.loads(rec.strip()).get("gpu", {})
                    idle = gpu.get("idle_ratio", 1.0)
                    freq = gpu.get("freq_hz", 0)
                    energy = gpu.get("gpu_energy", 0)
                    samples.append({
                        "gpu_id": 0,
                        "utilization_pct": round((1.0 - idle) * 100, 1),
                        "freq_mhz": round(freq / 1e6, 1),
                        "power_w": round(energy / interval_ms, 2),
                        "vendor": "apple",
                    })
                except Exception:
                    continue
            return samples
        except Exception:
            return []

except ImportError:
    def _run_powermetrics(interval_ms: int, n_samples: int) -> list[dict]:  # type: ignore[misc]
        return []


def _aggregate_gpu(samples: list[dict]) -> dict:
    if not samples:
        return {"gpus": [{"gpu_id": 0, "vendor": "apple"}], "available": False}
    avg_util = round(sum(s["utilization_pct"] for s in samples) / len(samples), 1)
    return {
        "gpus": [{
            "gpu_id": 0,
            "utilization_pct": avg_util,
            "peak_util_pct": max(s["utilization_pct"] for s in samples),
            "freq_mhz": round(sum(s["freq_mhz"] for s in samples) / len(samples), 1),
            "power_w": round(sum(s["power_w"] for s in samples) / len(samples), 2),
            "vendor": "apple",
            "n_samples": len(samples),
        }],
        "available": True,
    }


# ─── Chat template helpers ─────────────────────────────────────────────────────

def _llama3_prompt(messages: list[dict]) -> str:
    """Convert messages to Llama-3 instruct format."""
    parts = []
    for m in messages:
        parts.append(
            f"<|start_header_id|>{m.get('role', 'user')}<|end_header_id|>"
            f"\n\n{m.get('content', '')}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _chatml_prompt(messages: list[dict]) -> str:
    """Convert messages to ChatML format (Qwen, Mistral, etc.)."""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def messages_to_prompt(messages: list[dict]) -> str:
    if CHAT_TEMPLATE_FORMAT == "chatml":
        return _chatml_prompt(messages)
    return _llama3_prompt(messages)  # default: llama3


# ─── Log writer ────────────────────────────────────────────────────────────────

def _write_log(record: dict) -> None:
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("Log write failed: %s", e)


# ─── llama.cpp native /completion call ─────────────────────────────────────────

async def _native_completion(payload: dict) -> dict:
    """POST to llama.cpp /completion and return the JSON response."""
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(f"{LLAMA_SERVER_URL}/completion", json=payload)
        resp.raise_for_status()
        return resp.json()


def _extract_tracking(native: dict) -> tuple[int | None, dict, dict]:
    """
    Pull slot_id, cache_info, and gpu_info out of a /completion response.
    gpu_info is always empty here; the caller aggregates powermetrics separately.
    """
    slot_id: int | None = native.get("slot_id")
    timings = native.get("timings", {})
    prompt_n = int(timings.get("prompt_n", 0))
    cache_n = int(timings.get("cache_n", 0))
    predicted_n = int(timings.get("predicted_n", 0))
    total_prompt = prompt_n + cache_n
    cache_info = {
        "total_prompt_tokens": total_prompt,
        "reused_tokens": cache_n,
        "newly_computed": prompt_n,
        "reuse_ratio_pct": round(cache_n / total_prompt * 100, 1) if total_prompt > 0 else None,
        "prompt_eval_ms": round(timings.get("prompt_ms", 0), 2),
        "gen_ms": round(timings.get("predicted_ms", 0), 2),
        "prompt_tps": round(timings.get("prompt_per_second", 0), 2),
        "gen_tps": round(timings.get("predicted_per_second", 0), 2),
        "predicted_n": predicted_n,
    }
    return slot_id, cache_info, {}


# ─── Per-endpoint completion handlers ─────────────────────────────────────────

async def _handle_chat_completions(
    body: dict, query_id: str
) -> tuple[dict, dict, dict, int | None, str | None]:
    """
    Handle POST /v1/chat/completions.

    Translates to llama.cpp's native /completion to obtain slot_id and
    cache_n for eviction tracking, then returns an OpenAI-compatible response.
    """
    rid: str | None = body.get("rid") or body.get("request_id")
    messages = body.get("messages", [])
    model = body.get("model", "llama.cpp")
    max_tokens = body.get("max_tokens", -1)

    payload: dict[str, Any] = {
        "prompt": messages_to_prompt(messages),
        "n_predict": max_tokens if max_tokens and max_tokens > 0 else -1,
        "temperature": body.get("temperature", 0.8),
        "top_p": body.get("top_p", 0.95),
        "cache_prompt": True,
        "stop": ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"],
    }

    loop = asyncio.get_event_loop()
    native, gpu_samples = await asyncio.gather(
        _native_completion(payload),
        loop.run_in_executor(None, _run_powermetrics, 1500, 3),
    )
    gpu_info = _aggregate_gpu(gpu_samples)
    slot_id, cache_info, _ = _extract_tracking(native)
    predicted_n = cache_info.pop("predicted_n", 0)
    total = cache_info["total_prompt_tokens"]

    openai_resp: dict[str, Any] = {
        "id": f"chatcmpl-{query_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": native.get("content", "")},
            "finish_reason": "stop" if native.get("stop") else "length",
        }],
        "usage": {
            "prompt_tokens": total,
            "completion_tokens": predicted_n,
            "total_tokens": total + predicted_n,
        },
    }
    if _contextpilot_enabled and (rid or slot_id is not None):
        openai_resp["_contextpilot"] = {"request_id": rid, "slot_id": slot_id}

    return openai_resp, cache_info, gpu_info, slot_id, rid


async def _handle_text_completions(
    body: dict, query_id: str
) -> tuple[dict, dict, dict, int | None, str | None]:
    """
    Handle POST /v1/completions (OpenAI text completions format).

    This is the primary path used by ContextPilot's HTTP server
    (http_server.py adds 'rid' to the body before forwarding here).
    Translates to llama.cpp's native /completion to capture slot_id / cache_n.
    """
    rid: str | None = body.get("rid") or body.get("request_id")
    model = body.get("model", "llama.cpp")
    max_tokens = body.get("max_tokens", -1)

    # Build native /completion payload from OpenAI /v1/completions fields.
    # Fields unknown to llama.cpp (rid, request_id, model) are omitted.
    payload: dict[str, Any] = {
        "prompt": body.get("prompt", ""),
        "n_predict": max_tokens if max_tokens and max_tokens > 0 else -1,
        "temperature": body.get("temperature", 1.0),
        "top_p": body.get("top_p", 1.0),
        "cache_prompt": True,
    }
    if body.get("stop"):
        payload["stop"] = body["stop"]

    loop = asyncio.get_event_loop()
    native, gpu_samples = await asyncio.gather(
        _native_completion(payload),
        loop.run_in_executor(None, _run_powermetrics, 1500, 3),
    )
    gpu_info = _aggregate_gpu(gpu_samples)
    slot_id, cache_info, _ = _extract_tracking(native)
    predicted_n = cache_info.pop("predicted_n", 0)
    total = cache_info["total_prompt_tokens"]

    openai_resp: dict[str, Any] = {
        "id": f"cmpl-{query_id}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "text": native.get("content", ""),
            "index": 0,
            "finish_reason": "stop" if native.get("stop") else "length",
        }],
        "usage": {
            "prompt_tokens": total,
            "completion_tokens": predicted_n,
            "total_tokens": total + predicted_n,
        },
    }
    if _contextpilot_enabled and (rid or slot_id is not None):
        openai_resp["_contextpilot"] = {"request_id": rid, "slot_id": slot_id}

    return openai_resp, cache_info, gpu_info, slot_id, rid


async def _handle_native_completion(
    body: dict, query_id: str
) -> tuple[dict, dict, dict, int | None, str | None]:
    """
    Handle POST /completion (llama.cpp native API).

    Pass through to llama.cpp, but extract slot_id / cache_n / rid for
    eviction tracking before returning the unmodified response.
    """
    rid: str | None = body.get("rid") or body.get("request_id")
    # Strip ContextPilot-only fields; ensure cache_prompt is enabled
    forward = {k: v for k, v in body.items() if k not in ("rid", "request_id")}
    forward.setdefault("cache_prompt", True)

    loop = asyncio.get_event_loop()
    native, gpu_samples = await asyncio.gather(
        _native_completion(forward),
        loop.run_in_executor(None, _run_powermetrics, 1500, 3),
    )
    gpu_info = _aggregate_gpu(gpu_samples)
    slot_id, cache_info, _ = _extract_tracking(native)
    cache_info.pop("predicted_n", None)

    if _contextpilot_enabled and (rid or slot_id is not None):
        native["_contextpilot"] = {"request_id": rid, "slot_id": slot_id}

    return native, cache_info, gpu_info, slot_id, rid


# ─── Core proxy dispatcher ─────────────────────────────────────────────────────

async def _dispatch(request: Request, path: str) -> Response:
    """Route the request, handle eviction tracking, log, and return."""
    query_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()

    body_bytes = await request.body()
    try:
        body_json: dict = json.loads(body_bytes)
    except Exception:
        body_json = {}

    method = request.method
    is_chat = path in ("v1/chat/completions",) and method == "POST"
    is_text = path in ("v1/completions",) and method == "POST"
    is_native = path == "completion" and method == "POST"

    cache_info: dict = {}
    gpu_info: dict = {}
    slot_id: int | None = None
    rid: str | None = None
    status_code = 200

    try:
        if is_chat:
            resp_json, cache_info, gpu_info, slot_id, rid = await _handle_chat_completions(
                body_json, query_id
            )
            resp_body = json.dumps(resp_json).encode()

        elif is_text:
            resp_json, cache_info, gpu_info, slot_id, rid = await _handle_text_completions(
                body_json, query_id
            )
            resp_body = json.dumps(resp_json).encode()

        elif is_native:
            resp_json, cache_info, gpu_info, slot_id, rid = await _handle_native_completion(
                body_json, query_id
            )
            resp_body = json.dumps(resp_json).encode()

        else:
            # Generic pass-through for /health, /models, /tokenize, etc.
            async with httpx.AsyncClient(timeout=120) as client:
                upstream = await client.request(
                    method=method,
                    url=f"{LLAMA_SERVER_URL}/{path}",
                    headers={
                        k: v for k, v in request.headers.items()
                        if k.lower() not in ("host", "content-length")
                    },
                    content=body_bytes,
                )
            resp_body = upstream.content
            status_code = upstream.status_code
            resp_json = {}

    except Exception as e:
        logger.error("[%s] Error handling %s: %s", query_id, path, e)
        resp_body = json.dumps({"error": str(e)}).encode()
        status_code = 502
        resp_json = {}

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    # ── ContextPilot eviction tracking ────────────────────────────────────────
    if (is_chat or is_text or is_native) and slot_id is not None:
        evicted = await _update_slot(slot_id, rid)
        if evicted:
            asyncio.create_task(_fire_eviction({evicted}))

    # ── Logging ───────────────────────────────────────────────────────────────
    usage: dict = resp_json.get("usage", {}) if isinstance(resp_json, dict) else {}
    record = {
        "query_id": query_id,
        "timestamp": timestamp,
        "path": path,
        "latency_ms": latency_ms,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cache_reuse": cache_info,
        "gpu": gpu_info,
        "status_code": status_code,
        "model": body_json.get("model"),
        "n_messages": len(body_json.get("messages", [])),
        "slot_id": slot_id,
        "request_id": rid,
    }
    query_history.append(record)
    _write_log(record)

    # ── Console summary ───────────────────────────────────────────────────────
    parts = [f"[{query_id}] {path} | {latency_ms}ms"]
    if cache_info.get("reuse_ratio_pct") is not None:
        c = cache_info
        parts.append(
            f"cache {c['reused_tokens']}/{c['total_prompt_tokens']} "
            f"({c['reuse_ratio_pct']}%)"
        )
    gpus = gpu_info.get("gpus", [])
    if gpus and gpus[0].get("utilization_pct") is not None:
        g = gpus[0]
        parts.append(
            f"GPU {g['utilization_pct']}% {g.get('power_w', '?')}W"
        )
    if slot_id is not None:
        parts.append(f"slot={slot_id}")
    if rid:
        parts.append(f"rid={rid[:16]}")
    logger.info(" | ".join(parts))

    return Response(
        content=resp_body,
        status_code=status_code,
        headers={"content-type": "application/json"},
    )


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def openai_routes(request: Request, path: str):
    return await _dispatch(request, f"v1/{path}")


@app.api_route("/completion", methods=["POST"])
async def native_completion_route(request: Request):
    return await _dispatch(request, "completion")


@app.get("/stats")
async def stats():
    if not query_history:
        return JSONResponse({"message": "No queries recorded yet."})

    total = len(query_history)
    latencies = [q["latency_ms"] for q in query_history]
    reuse_qs = [
        q for q in query_history
        if q.get("cache_reuse", {}).get("reuse_ratio_pct") is not None
    ]
    sorted_lat = sorted(latencies)

    async with _slot_lock:  # type: ignore[union-attr]
        slot_summary = {str(sid): cur_rid for sid, cur_rid in _slot_state.items()}

    return JSONResponse({
        "total_queries": total,
        "latency_ms": {
            "avg": round(sum(latencies) / total, 2),
            "p50": sorted_lat[int(total * 0.50)],
            "p95": sorted_lat[min(int(total * 0.95), total - 1)],
            "min": min(latencies),
            "max": max(latencies),
        },
        "cache_reuse": {
            "queries_with_data": len(reuse_qs),
            "avg_reuse_ratio_pct": (
                round(
                    sum(q["cache_reuse"]["reuse_ratio_pct"] for q in reuse_qs)
                    / len(reuse_qs),
                    1,
                )
                if reuse_qs else None
            ),
        },
        "slot_state": slot_summary,
        "contextpilot_enabled": _contextpilot_enabled,
        "contextpilot_index_url": CONTEXTPILOT_INDEX_URL,
        "recent_queries": list(query_history)[-10:],
    })


@app.post("/reset")
async def reset_slots():
    """
    Reset slot state and notify ContextPilot of all evictions.
    Call this when the ContextPilot index is reset (POST /reset on the index
    server) or when llama.cpp is restarted.
    """
    assert _slot_lock is not None
    async with _slot_lock:
        evicted_ids = {rid for rid in _slot_state.values() if rid}
        cleared = len(_slot_state)
        _slot_state.clear()

    if evicted_ids:
        asyncio.create_task(_fire_eviction(evicted_ids))

    return JSONResponse({
        "status": "ok",
        "cleared_slots": cleared,
        "evicted_request_ids": list(evicted_ids),
    })


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def generic_proxy(request: Request, path: str):
    return await _dispatch(request, path)


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="warning")
