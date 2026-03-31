"""
llama.cpp Proxy Server with ContextPilot Integration (macOS / Apple Silicon)
=============================================================================
Sits between the client and llama-server.  Adds:

  - Apple GPU metrics via powermetrics (Apple Silicon only)
  - Cache-reuse stats parsed from llama-server's non-standard `timings` field
  - Per-request logging to query_log.jsonl and a /stats summary endpoint

ContextPilot integration:
    The native C++ hook (contextpilot._llamacpp_hook) injected into
    llama-server at launch handles eviction tracking automatically — no
    additional proxy-level registration is needed.

    CONTEXTPILOT_INDEX_URL=http://localhost:8765 contextpilot-llama-server ...
"""

import asyncio
import time
import json
import uuid
import httpx
import logging
import plistlib
import subprocess
from datetime import datetime, timezone
from collections import deque

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

# ─── Config ────────────────────────────────────────────────────────────────────
LLAMA_SERVER_URL      = "http://localhost:8889"
PROXY_HOST            = "0.0.0.0"
PROXY_PORT            = 8890
LOG_FILE              = "query_log.jsonl"
MAX_HISTORY           = 1000
CONTEXTPILOT_INDEX_URL = __import__("os").environ.get("CONTEXTPILOT_INDEX_URL")

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

if CONTEXTPILOT_INDEX_URL:
    logger.info(f"[ContextPilot] Eviction tracking active → {CONTEXTPILOT_INDEX_URL}")

app = FastAPI(title="llama.cpp Proxy with Metrics")
query_history: deque = deque(maxlen=MAX_HISTORY)


# ─── GPU Utils (Apple Silicon) ────────────────────────────────────────────────
def _run_powermetrics(interval_ms: int, n_samples: int) -> list[dict]:
    """
    Blocking call to powermetrics. Runs for interval_ms * n_samples milliseconds.
    Returns list of GPU samples parsed from plist output.
    M3 fields: idle_ratio, freq_hz, gpu_energy (mJ per interval)
    """
    try:
        result = subprocess.run(
            ["sudo", "-n", "/usr/bin/powermetrics",
             "--samplers", "gpu_power",
             "-i", str(interval_ms),
             "-n", str(n_samples),
             "--format", "plist"],
            capture_output=True, timeout=interval_ms * n_samples / 1000 + 3
        )
        if result.returncode != 0:
            logger.warning(f"[gpu] powermetrics failed: {result.stderr[:200]}")
            return []

        # Records are separated by <?xml headers (no null bytes on macOS 14+)
        raw     = result.stdout
        parts   = raw.split(b"<?xml")
        records = [b"<?xml" + p for p in parts if p.strip()]
        samples = []
        for rec in records:
            rec = rec.strip()
            if not rec:
                continue
            try:
                data          = plistlib.loads(rec)
                gpu           = data.get("gpu", {})
                idle_ratio    = gpu.get("idle_ratio", 1.0)
                freq_hz       = gpu.get("freq_hz", 0)
                gpu_energy_mj = gpu.get("gpu_energy", 0)
                power_w       = gpu_energy_mj / interval_ms   # mJ/ms = W
                samples.append({
                    "gpu_id":          0,
                    "utilization_pct": round((1.0 - idle_ratio) * 100, 1),
                    "freq_mhz":        round(freq_hz / 1e6, 1),
                    "power_w":         round(power_w, 2),
                    "vendor":          "apple",
                })
            except Exception:
                continue
        return samples
    except subprocess.TimeoutExpired:
        logger.warning("[gpu] powermetrics timed out")
        return []
    except Exception as e:
        logger.warning(f"[gpu] error: {e}")
        return []


def _static_gpu_name() -> dict:
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=5
        )
        data  = json.loads(result.stdout)
        cards = data.get("SPDisplaysDataType", [])
        if cards:
            return {"gpu_id": 0,
                    "name":   cards[0].get("sppci_model", "Apple GPU"),
                    "vendor": "apple"}
    except Exception:
        pass
    return {"gpu_id": 0, "name": "Apple GPU", "vendor": "apple"}


def aggregate_gpu_samples(samples: list) -> dict:
    if not samples:
        return {"gpus": [_static_gpu_name()], "available": True,
                "note": "no live samples captured"}
    avg_util  = round(sum(s["utilization_pct"] for s in samples) / len(samples), 1)
    peak_util = max(s["utilization_pct"] for s in samples)
    avg_power = round(sum(s["power_w"] for s in samples) / len(samples), 2)
    avg_freq  = round(sum(s["freq_mhz"] for s in samples) / len(samples), 1)
    return {
        "gpus": [{
            "gpu_id":          0,
            "utilization_pct": avg_util,
            "peak_util_pct":   peak_util,
            "freq_mhz":        avg_freq,
            "power_w":         avg_power,
            "vendor":          "apple",
            "n_samples":       len(samples),
        }],
        "available": True,
    }


# ─── Log writer ────────────────────────────────────────────────────────────────
def write_log(record: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ─── Forward to llama-server's built-in OpenAI API ────────────────────────────
async def _forward_chat_completions(body: dict) -> dict:
    """Pass the request directly to llama-server's /v1/chat/completions.

    llama-server handles chat-template formatting internally using the
    template embedded in the GGUF file, so no manual prompt construction
    is needed here.
    """
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()


# ─── Chat completions handler ──────────────────────────────────────────────────
async def handle_chat_completions(body_json: dict, query_id: str):
    loop = asyncio.get_event_loop()

    INTERVAL_MS = 1500
    N_SAMPLES   = 3   # covers up to ~4.5s of inference

    oai_resp, gpu_samples = await asyncio.gather(
        _forward_chat_completions(body_json),
        loop.run_in_executor(None, _run_powermetrics, INTERVAL_MS, N_SAMPLES),
    )

    gpu_info = aggregate_gpu_samples(gpu_samples)
    logger.info(f"[gpu] {len(gpu_samples)} samples captured")

    # llama-server includes a non-standard `timings` field alongside the
    # standard OpenAI response body.  Extract cache stats if present.
    timings      = oai_resp.get("timings", {})
    prompt_n     = int(timings.get("prompt_n", 0))
    cache_n      = int(timings.get("cache_n", 0))
    total_prompt = prompt_n + cache_n
    cache_info   = {
        "total_prompt_tokens": total_prompt,
        "reused_tokens":       cache_n,
        "newly_computed":      prompt_n,
        "reuse_ratio_pct":     round(cache_n / total_prompt * 100, 1) if total_prompt > 0 else None,
        "prompt_eval_ms":      round(timings.get("prompt_ms", 0), 2),
        "gen_ms":              round(timings.get("predicted_ms", 0), 2),
        "prompt_tps":          round(timings.get("prompt_per_second", 0), 2),
        "gen_tps":             round(timings.get("predicted_per_second", 0), 2),
    } if timings else {}

    return oai_resp, cache_info, gpu_info


# ─── Core proxy handler ────────────────────────────────────────────────────────
async def proxy_request(request: Request, path: str):
    query_id   = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()
    timestamp  = datetime.now(timezone.utc).isoformat()

    body_bytes = await request.body()
    try:
        body_json = json.loads(body_bytes)
    except Exception:
        body_json = {}

    is_chat = path.endswith("chat/completions") and request.method == "POST"

    if is_chat:
        resp_json, cache_info, gpu_info = await handle_chat_completions(body_json, query_id)
        resp_body   = json.dumps(resp_json).encode()
        status_code = 200
    else:
        async with httpx.AsyncClient(timeout=120) as client:
            upstream = await client.request(
                method  = request.method,
                url     = f"{LLAMA_SERVER_URL}/{path}",
                headers = {k: v for k, v in request.headers.items()
                           if k.lower() not in ("host", "content-length")},
                content = body_bytes,
            )
        resp_body   = upstream.content
        status_code = upstream.status_code
        cache_info  = {}
        gpu_info    = {}
        try:
            resp_json = json.loads(resp_body)
        except Exception:
            resp_json = {}

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    usage = resp_json.get("usage", {}) if isinstance(resp_json, dict) else {}
    record = {
        "query_id":          query_id,
        "timestamp":         timestamp,
        "path":              path,
        "latency_ms":        latency_ms,
        "prompt_tokens":     usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cache_reuse":       cache_info,
        "gpu":               gpu_info,
        "status_code":       status_code,
        "model":             body_json.get("model"),
        "n_messages":        len(body_json.get("messages", [])),
    }
    query_history.append(record)
    write_log(record)

    # ── Console log ───────────────────────────────────────────────────────────
    if cache_info and cache_info.get("reuse_ratio_pct") is not None:
        reused = cache_info["reused_tokens"]
        total  = cache_info["total_prompt_tokens"]
        new_c  = cache_info["newly_computed"]
        ratio  = cache_info["reuse_ratio_pct"]
        cache_str = f" | cache: {reused}/{total} reused, {new_c} computed ({ratio}%)"
    else:
        cache_str = " | cache: n/a"

    gpu_str = ""
    gpus = gpu_info.get("gpus", [])
    if gpus:
        g = gpus[0]
        if "utilization_pct" in g:
            gpu_str = (f" | GPU avg={g['utilization_pct']}%"
                       f" peak={g.get('peak_util_pct','?')}%"
                       f" {g.get('power_w','?')}W"
                       f" ({g.get('n_samples',0)} samples)")
        elif "name" in g:
            gpu_str = f" | GPU {g['name']}"

    logger.info(
        f"[{query_id}] {path} | {latency_ms}ms"
        f" | prompt={usage.get('prompt_tokens',0)} comp={usage.get('completion_tokens',0)}"
        f"{cache_str}{gpu_str}"
    )

    return Response(
        content     = resp_body,
        status_code = status_code,
        headers     = {"content-type": "application/json"},
    )


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def openai_proxy(request: Request, path: str):
    return await proxy_request(request, f"v1/{path}")

@app.get("/stats")
async def stats_endpoint():
    if not query_history:
        return JSONResponse({"message": "No queries yet."})
    total      = len(query_history)
    latencies  = [q["latency_ms"] for q in query_history]
    reuse_qs   = [q for q in query_history
                  if q.get("cache_reuse", {}).get("reuse_ratio_pct") is not None]
    sorted_lat = sorted(latencies)

    result = {
        "total_queries": total,
        "latency_ms": {
            "avg": round(sum(latencies) / total, 2),
            "p50": sorted_lat[int(total * 0.50)],
            "p95": sorted_lat[min(int(total * 0.95), total - 1)],
            "min": min(latencies),
            "max": max(latencies),
        },
        "cache_reuse": {
            "queries_with_data":   len(reuse_qs),
            "avg_reuse_ratio_pct": round(
                sum(q["cache_reuse"]["reuse_ratio_pct"] for q in reuse_qs) / len(reuse_qs), 1
            ) if reuse_qs else None,
        },
        "recent_queries": list(query_history)[-10:],
    }

    if CONTEXTPILOT_INDEX_URL:
        result["contextpilot"] = {"index_url": CONTEXTPILOT_INDEX_URL}

    return JSONResponse(result)

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def generic_proxy(request: Request, path: str):
    return await proxy_request(request, path)


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"Starting proxy on {PROXY_HOST}:{PROXY_PORT} → {LLAMA_SERVER_URL}")
    if CONTEXTPILOT_INDEX_URL:
        logger.info(f"ContextPilot eviction tracking → {CONTEXTPILOT_INDEX_URL}")
    else:
        logger.info("ContextPilot eviction tracking disabled (set CONTEXTPILOT_INDEX_URL to enable)")
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="warning")
