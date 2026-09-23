#!/usr/bin/env python3
"""
Video-RAG benchmark: does reordering retrieved frames for KV-cache prefix
sharing (plus a one-sentence "true order" hint) reduce prefill time while
keeping accuracy?

Each question gets its top-k retrieved frames (from retrieve_frames.py).
The *same* frames are sent under several prompt conditions:

    chrono_plain     frames in chronological order, no labels, no note   (natural baseline)
    chrono_labels    chronological + per-frame timestamp labels          (strong baseline)
    cp_sentence      ContextPilot reorder + one sentence giving true order
    cp_labels        ContextPilot reorder + timestamp labels only
    cp_both          ContextPilot reorder + labels + sentence
    cp_none          ContextPilot reorder, no order information          (ablation)
    shuffle_sentence random order (not cache-aware) + sentence           (isolates hint vs cache)

Per request we record TTFT (streaming), total latency, prompt/cached/completion
tokens (SGLang ``--enable-cache-report``) and the parsed answer.  The engine
cache is flushed before every condition so each starts cold.

Usage (inside the cluster, against the SGLang service):
    python run_video_rag_bench.py --questions q.jsonl --retrieval retrieval.jsonl \
        --frames /mnt/cp/data/lvb/frames --api http://sglang-qwen38-27b:8000 \
        --model Qwen/Qwen3.8-27B --k 64 --out /mnt/cp/results/lvb_qwen38_27b
"""

import argparse
import asyncio
import base64
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional

import aiohttp
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from contextpilot.multimodal import (  # noqa: E402
    ImageBlock,
    build_multimodal_messages,
    frame_label,
    group_execution_order,
    plan_canonical_batch,
    reorder_blocks_batch,
    true_order,
)
from contextpilot.server.live_index import ContextPilot  # noqa: E402

CONDITIONS = {
    #  name:              (ordering,   order_hint, in_order_note)
    "chrono_plain":       ("chrono",   "none",     False),
    "chrono_labels":      ("chrono",   "labels",   False),
    "cp_sentence":        ("cp",       "sentence", True),
    "cp_labels":          ("cp",       "labels",   False),
    "cp_both":            ("cp",       "both",     True),
    "cp_none":            ("cp",       "none",     False),
    "shuffle_sentence":   ("shuffle",  "sentence", True),
    "shuffle_none":       ("shuffle",  "none",     False),
    # Canonical per-video core prefix (see contextpilot.multimodal.canonical).
    # "canon"  = core padded so every request of a video shares one boundary
    # "canons" = soft, only the core frames this request retrieved
    "canon_sentence":     ("canon",    "sentence", True),
    "canon_labels":       ("canon",    "labels",   False),
    "canon_both":         ("canon",    "both",     True),
    "canon_none":         ("canon",    "none",     False),
    "canon_soft_sentence": ("canons",  "sentence", True),
}

# Official Video-MME / LVBench style answer instruction: without the trailing
# "The best answer is:" small models write a paragraph and the letter never
# appears inside the token budget.
ANSWER_SUFFIX = (
    "Select the best answer to the above multiple-choice question based on the "
    "video frames. Respond with only the letter of the correct option.\n\n"
    "The best answer is:"
)
LETTER_RE = re.compile(r"\b([A-J])\b")
_ANSWER_IS_RE = re.compile(r"(?:answer|option)\s*(?:is|:)\s*\(?([A-J])\)?", re.IGNORECASE)

_b64_cache: Dict[str, str] = {}


def _encode(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def _data_uri(path: str) -> str:
    if path not in _b64_cache:
        _b64_cache[path] = _encode(path)
    return _b64_cache[path]


def warm_frame_cache(paths, workers: int = 32) -> None:
    """Read and encode frames in parallel.

    Frames live on a network filesystem where a single-threaded read of a few
    thousand small files takes minutes; the requests themselves then only need
    memory.
    """
    todo = [p for p in dict.fromkeys(paths) if p not in _b64_cache]
    if not todo:
        return
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        for path, uri in zip(todo, ex.map(_encode, todo)):
            _b64_cache[path] = uri
    print(f"  prefetched {len(todo)} frames in {time.perf_counter() - t0:.1f}s", flush=True)


def all_frame_paths(questions, retrieval, frames_root, k, meta_cache):
    """Every frame file the run will need (top-k of each question, per video)."""
    paths = []
    for q in questions:
        ret = retrieval.get(q["qid"])
        if ret is None:
            continue
        vid = q["video_id"]
        if vid not in meta_cache:
            with open(os.path.join(frames_root, vid, "frames.json")) as f:
                meta_cache[vid] = json.load(f)
        meta = meta_cache[vid]
        for idx in ret["topk"][:k]:
            paths.append(os.path.join(frames_root, vid, meta["frames"][idx]))
    return paths


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def build_blocks(q, ret, frames_root, k, meta_cache) -> List[ImageBlock]:
    vid = q["video_id"]
    if vid not in meta_cache:
        meta_cache[vid] = json.load(open(os.path.join(frames_root, vid, "frames.json")))
    meta = meta_cache[vid]
    blocks = []
    for idx in ret["topk"][:k]:
        ts = float(meta["timestamps"][idx])
        path = os.path.join(frames_root, vid, meta["frames"][idx])
        blocks.append(ImageBlock(
            key=f"{vid}#{idx}", image_url=_data_uri(path), order=ts,
            label=frame_label(idx, ts), meta={"video_id": vid, "frame_index": idx},
        ))
    return blocks


def format_query(q) -> str:
    opts = q.get("options") or []
    lines = [q["question"]]
    for o in opts:
        lines.append(o)
    return "\n".join(lines)


def parse_letter(text: str, n_opts: int) -> Optional[str]:
    """Extract the chosen option letter, preferring an explicit statement."""
    text = text.strip()
    limit = max(n_opts, 1)
    candidates = [m.group(1) for m in _ANSWER_IS_RE.finditer(text)]
    if candidates:
        letter = candidates[-1]
        return letter if ord(letter) - ord("A") < limit else None
    m = LETTER_RE.search(text)
    if not m:
        m = re.search(r"\(?([A-J])\)?[\.\:\)]", text)
    if not m:
        return None
    letter = m.group(1)
    if ord(letter) - ord("A") >= limit:
        return None
    return letter


def make_requests(condition: str, questions, retrieval, frames_root, k, seed, meta_cache,
                  min_share: float = 0.5, core_size: Optional[int] = None,
                  budget: bool = True):
    ordering, hint, in_order_note = CONDITIONS[condition]
    all_blocks = [build_blocks(q, retrieval[q["qid"]], frames_root, k, meta_cache) for q in questions]
    queries = [format_query(q) for q in questions]
    in_order_template = (
        "Note: the frames above are shown in chronological order, from earliest to latest."
        if in_order_note else None
    )

    if ordering == "cp":
        pilot = ContextPilot(use_gpu=False)
        displayed_batch, order = reorder_blocks_batch(all_blocks, pilot=pilot)
    elif ordering in ("canon", "canons"):
        groups = [q["video_id"] for q in questions]
        displayed_batch, _cores = plan_canonical_batch(
            all_blocks, groups,
            min_share=min_share, core_size=core_size,
            include_missing=(ordering == "canon"),
            # keep the frame count equal to the baseline so a prefill saving
            # is not paid for with extra pinned frames
            budget=(k if budget else None),
        )
        # send each video's questions consecutively so the core is warm
        order = group_execution_order(groups)
        displayed_batch = [displayed_batch[i] for i in order]
    elif ordering == "chrono":
        displayed_batch = [true_order(b) for b in all_blocks]
        order = list(range(len(questions)))
    elif ordering == "shuffle":
        rng = random.Random(seed)
        displayed_batch = []
        for b in all_blocks:
            b = list(b)
            rng.shuffle(b)
            displayed_batch.append(b)
        order = list(range(len(questions)))
    else:
        raise ValueError(ordering)

    reqs = []
    for displayed, oi in zip(displayed_batch, order):
        q = questions[oi]
        msgs = build_multimodal_messages(
            displayed, queries[oi], order_hint=hint,
            in_order_template=in_order_template, extra_suffix=ANSWER_SUFFIX,
        )
        reqs.append({
            "qid": q["qid"], "video_id": q["video_id"], "orig_index": oi,
            "answer": q["answer"], "n_opts": len(q.get("options") or []),
            "display_frames": [b.meta["frame_index"] for b in displayed],
            "messages": msgs,
        })
    return reqs


async def send_one(session, api, model, req, max_tokens, extra_body, timeout):
    payload = {
        "model": model, "messages": req["messages"], "max_tokens": max_tokens,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
    }
    payload.update(extra_body)
    out = {"qid": req["qid"], "video_id": req["video_id"], "orig_index": req["orig_index"],
           "display_frames": req["display_frames"], "answer": req["answer"],
           "text": "", "ttft": None, "latency": None, "prompt_tokens": None,
           "cached_tokens": None, "completion_tokens": None, "error": None}
    t0 = time.perf_counter()
    try:
        async with session.post(f"{api}/v1/chat/completions", json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                out["error"] = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return out
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                for ch in chunk.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content") or ""
                    if delta:
                        if out["ttft"] is None:
                            out["ttft"] = time.perf_counter() - t0
                        out["text"] += delta
                usage = chunk.get("usage")
                if usage:
                    out["prompt_tokens"] = usage.get("prompt_tokens")
                    out["completion_tokens"] = usage.get("completion_tokens")
                    det = usage.get("prompt_tokens_details") or {}
                    out["cached_tokens"] = det.get("cached_tokens", usage.get("cached_tokens"))
            out["latency"] = time.perf_counter() - t0
            if out["ttft"] is None:
                out["ttft"] = out["latency"]
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["latency"] = time.perf_counter() - t0
    out["pred"] = parse_letter(out["text"], req["n_opts"]) if out["text"] else None
    out["correct"] = (out["pred"] == out["answer"]) if out["pred"] else False
    return out


async def run_condition(api, model, reqs, concurrency, max_tokens, extra_body, timeout, progress_every=25):
    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(reqs)
    done = 0

    async with aiohttp.ClientSession() as session:
        async def worker(i, req):
            nonlocal done
            async with sem:
                results[i] = await send_one(session, api, model, req, max_tokens, extra_body, timeout)
                done += 1
                if done % progress_every == 0:
                    ok = sum(1 for r in results if r and r.get("correct"))
                    print(f"    {done}/{len(reqs)} acc_so_far={ok / done:.3f}", flush=True)

        t0 = time.perf_counter()
        # Preserve submission order (matters for cache-aware scheduling)
        tasks = []
        for i, req in enumerate(reqs):
            tasks.append(asyncio.create_task(worker(i, req)))
            await asyncio.sleep(0)  # let the task start in order
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    return results, wall


async def flush_cache(api):
    async with aiohttp.ClientSession() as s:
        for path in ("/flush_cache",):
            try:
                async with s.post(f"{api}{path}", timeout=aiohttp.ClientTimeout(total=60)) as r:
                    return r.status
            except Exception as e:  # noqa: BLE001
                return f"error: {e}"


async def get_metrics(api):
    async with aiohttp.ClientSession() as s:
        try:
            async with s.get(f"{api}/metrics", timeout=aiohttp.ClientTimeout(total=30)) as r:
                return await r.text()
        except Exception:  # noqa: BLE001
            return ""


def parse_prefix_metrics(text: str) -> Dict[str, float]:
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key in ("sglang:cached_tokens_total", "sglang:prompt_tokens_total", "sglang:cache_hit_rate"):
            if line.startswith(key):
                try:
                    out[key] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def summarize(results, wall):
    ok = [r for r in results if not r["error"]]
    n = len(results)
    acc = sum(1 for r in ok if r["correct"]) / n if n else 0.0
    ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
    pt = sum(r["prompt_tokens"] or 0 for r in ok)
    ct = sum(r["cached_tokens"] or 0 for r in ok)
    lat = sorted(r["latency"] for r in ok)

    def pct(a, p):
        return a[min(len(a) - 1, int(p * len(a)))] if a else None

    return {
        "n": n, "errors": n - len(ok), "accuracy": acc,
        "unparsed": sum(1 for r in ok if r["pred"] is None),
        "ttft_mean": sum(ttfts) / len(ttfts) if ttfts else None,
        "ttft_p50": pct(ttfts, 0.5), "ttft_p90": pct(ttfts, 0.9),
        "latency_mean": sum(lat) / len(lat) if lat else None,
        "prompt_tokens_total": pt, "cached_tokens_total": ct,
        "cache_hit_ratio": ct / pt if pt else None,
        "prompt_tokens_mean": pt / len(ok) if ok else None,
        "wall_s": wall, "req_per_s": len(ok) / wall if wall else None,
        "prefill_tok_per_s_uncached": (pt - ct) / sum(r["ttft"] for r in ok) if ok and sum(r["ttft"] for r in ok) > 0 else None,
        # prefill-side aggregates (TTFT is prefill-dominated: max_tokens is tiny)
        "ttft_total_s": sum(r["ttft"] for r in ok if r["ttft"] is not None),
        "ttft_p99": pct(ttfts, 0.99),
        "uncached_tokens_per_req": (pt - ct) / len(ok) if ok else None,
        "cached_tokens_per_req": ct / len(ok) if ok else None,
        "frames_per_req": sum(len(r["display_frames"]) for r in ok) / len(ok) if ok else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--conditions", default="chrono_plain,cp_sentence,canon_sentence,canon_none,chrono_labels,cp_none,shuffle_sentence")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-share", type=float, default=0.5,
                    help="canonical core: minimum fraction of a video's questions that must retrieve a frame")
    ap.add_argument("--core-size", type=int, default=None,
                    help="canonical core: cap on the number of pinned frames")
    ap.add_argument("--no-budget", action="store_true",
                    help="canonical: do not cap the frame count at k (prompts grow)")
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--extra-body", default='{"chat_template_kwargs": {"enable_thinking": false}}')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    questions = load_jsonl(a.questions)
    retrieval = {r["qid"]: r for r in load_jsonl(a.retrieval)}
    questions = [q for q in questions if q["qid"] in retrieval]
    if a.limit:
        questions = questions[: a.limit]
    print(f"{len(questions)} questions, k={a.k}, conditions={a.conditions}")
    meta_cache: Dict[str, Any] = {}
    warm_frame_cache(all_frame_paths(questions, retrieval, a.frames, a.k, meta_cache))
    extra_body = json.loads(a.extra_body) if a.extra_body else {}

    summary_path = os.path.join(a.out, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    summary["_config"] = {k: v for k, v in vars(a).items()}
    summary["_config"]["n_questions"] = len(questions)

    for cond in a.conditions.split(","):
        cond = cond.strip()
        if not cond:
            continue
        res_path = os.path.join(a.out, f"{cond}.jsonl")
        if cond in summary and os.path.exists(res_path) and summary[cond].get("n") == len(questions):
            print(f"[{cond}] already done, skipping")
            continue
        print(f"[{cond}] building requests...", flush=True)
        t0 = time.perf_counter()
        reqs = make_requests(cond, questions, retrieval, a.frames, a.k, a.seed, meta_cache,
                             min_share=a.min_share, core_size=a.core_size,
                             budget=not a.no_budget)
        build_s = time.perf_counter() - t0
        if not a.no_flush:
            st = asyncio.run(flush_cache(a.api))
            print(f"[{cond}] flush_cache -> {st}")
        m0 = parse_prefix_metrics(asyncio.run(get_metrics(a.api)))
        print(f"[{cond}] sending {len(reqs)} requests (concurrency={a.concurrency})...", flush=True)
        results, wall = asyncio.run(run_condition(a.api, a.model, reqs, a.concurrency, a.max_tokens, extra_body, a.timeout))
        m1 = parse_prefix_metrics(asyncio.run(get_metrics(a.api)))
        with open(res_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        s = summarize(results, wall)
        s["build_s"] = build_s
        s["server_metrics_delta"] = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
        summary[cond] = s
        json.dump(summary, open(summary_path, "w"), indent=2)
        print(f"[{cond}] acc={s['accuracy']:.4f} ttft_mean={s['ttft_mean']:.3f}s "
              f"cache_hit={s['cache_hit_ratio']:.3f} errors={s['errors']} wall={wall:.1f}s", flush=True)

    print("\n=== SUMMARY ===")
    print(f"{'condition':<18}{'acc':>8}{'ttft_mean':>11}{'ttft_p50':>10}{'ttft_p99':>10}"
          f"{'hit':>8}{'uncach/req':>11}{'prefill_s':>11}{'errors':>7}{'wall_s':>9}")
    for cond, s in summary.items():
        if cond.startswith("_"):
            continue
        print(f"{cond:<18}{s['accuracy']:>8.4f}{s['ttft_mean'] or 0:>11.3f}{s['ttft_p50'] or 0:>10.3f}"
              f"{s.get('ttft_p99') or 0:>10.3f}{s['cache_hit_ratio'] or 0:>8.3f}"
              f"{s.get('uncached_tokens_per_req') or 0:>11.0f}{s.get('ttft_total_s') or 0:>11.0f}"
              f"{s['errors']:>7}{s['wall_s']:>9.1f}")


if __name__ == "__main__":
    main()
