"""
A/B Comparison: Baseline vs ContextPilot — Prefill Latency, Cache Hit Rate, Answer Quality

Runs the SAME set of mem0-retrieved requests twice, **one at a time** (batch_size=1):
  A) Baseline:      Original retrieval order, original query order, direct to SGLang
  B) ContextPilot:  Reordered memories + scheduled send order (from /build), direct to SGLang

Both runs send directly to SGLang — the only variables are:
  1) Send order:   Baseline sends queries 0→7 in order.
                   ContextPilot sends in scheduled_order (optimized for prefix reuse).
  2) Memory order: Baseline uses mem0's retrieval order within each prompt.
                   ContextPilot uses reordered_contexts from /build (shared prefixes first).

Between runs, SGLang's radix cache is flushed for a fair comparison.
Batch size = 1 ensures each request can benefit from the previous request's cached KV states.

Metrics compared:
  - Per-request cached_tokens (radix cache hits)
  - Total prefill tokens (prompt_tokens - cached_tokens = actual prefill work)
  - Cumulative wall-clock time
  - Answer quality (side-by-side)

SETUP:
  # Terminal 1: SGLang with cache report + LPM
  RAGBOOST_INDEX_URL=http://localhost:8765 python -m sglang.launch_server \
      --model-path Qwen/Qwen3-4B --tp-size 1 --schedule-policy lpm \
      --port 30000 --enable-cache-report

  # Terminal 2: ContextPilot HTTP server
  python -m contextpilot.server.http_server --port 8765 --infer-api-url http://localhost:30000

  # Terminal 3: Run this script
  export OPENAI_API_KEY=sk-...
  python examples/mem0_ab_comparison.py
"""

import os
import sys
import time
import requests
from typing import List, Dict, Any

from contextpilot.retriever import Mem0Retriever


# =============================================================================
# Config
# =============================================================================

SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:30000")
CONTEXTPILOT_URL = os.environ.get("CONTEXTPILOT_URL", "http://localhost:8765")
TOP_K = 10
MAX_TOKENS = 150


# =============================================================================
# Past conversations (same as main demo)
# =============================================================================

PAST_CONVERSATIONS = [
    [
        {"role": "user", "content": "I always fly ANA to Japan, I have mileage card NH-12345678 with 85000 miles. I prefer window seats."},
        {"role": "assistant", "content": "Noted! ANA with card NH-12345678, 85K miles, window seat preference."},
    ],
    [
        {"role": "user", "content": "For hotels in Tokyo I want to be near a subway station. My budget is about 2000 CNY per night. Must have a gym."},
        {"role": "assistant", "content": "Got it — subway proximity, ~2000 CNY/night, gym required."},
    ],
    [
        {"role": "user", "content": "Last time I stayed at Hyatt Regency Shinjuku. Location was great, right by the station, but the soundproofing was terrible. I'd rate it 4 out of 5."},
        {"role": "assistant", "content": "Hyatt Shinjuku: 4/5, good location, poor soundproofing. I'll note that for next time."},
    ],
    [
        {"role": "user", "content": "I'm allergic to peanuts and shrimp. I prefer vegetarian Japanese food. I absolutely cannot eat natto, I hate fermented soybeans."},
        {"role": "assistant", "content": "Understood: peanut & shrimp allergy, vegetarian Japanese preference, no natto."},
    ],
    [
        {"role": "user", "content": "I loved Ain Soph Ripple in Ginza for vegan food. Also T's TanTan at Tokyo Station has amazing veggie ramen."},
        {"role": "assistant", "content": "Great picks! Ain Soph Ripple (Ginza) and T's TanTan (Tokyo Station) saved."},
    ],
    [
        {"role": "user", "content": "For group dinners I prefer izakaya style, usually around Shinjuku area. My favorite ramen is at Fuunji near Shinjuku station."},
        {"role": "assistant", "content": "Izakaya in Shinjuku for groups, Fuunji for ramen. Noted!"},
    ],
    [
        {"role": "user", "content": "I once had wagyu at Gyukatsu Motomura in Shibuya — one of the best meals of my life."},
        {"role": "assistant", "content": "Gyukatsu Motomura Shibuya — a must-revisit for wagyu!"},
    ],
    [
        {"role": "user", "content": "I visited Tokyo in October 2024, stayed 5 days. Went to Akihabara for electronics and Shibuya for shopping. Got a 72-hour Metro pass which was super convenient."},
        {"role": "assistant", "content": "Oct 2024 trip: 5 days, Akihabara + Shibuya, 72h Metro pass was great."},
    ],
    [
        {"role": "user", "content": "TeamLab Borderless in Odaiba was incredible, I highly recommend it. I also hiked Takao-san trail — beautiful views of Mt Fuji."},
        {"role": "assistant", "content": "TeamLab Borderless at Odaiba and Takao-san hike—both highly recommended!"},
    ],
    [
        {"role": "user", "content": "I use Suica card linked to my Apple Watch for transit in Tokyo. Much easier than buying individual tickets."},
        {"role": "assistant", "content": "Suica on Apple Watch — contactless transit. Very efficient!"},
    ],
    [
        {"role": "user", "content": "I work as an ML engineer at a tech startup in Shanghai. I'm building a RAG system using LLMs for document QA. My team has 6 engineers and our manager is Zhang Wei."},
        {"role": "assistant", "content": "ML engineer in Shanghai, RAG/LLM project, team of 6, manager Zhang Wei."},
    ],
    [
        {"role": "user", "content": "We have weekly standup Monday at 10am. I prefer morning meetings. Our project deadline is March 15 for the RAG demo."},
        {"role": "assistant", "content": "Monday 10am standup, morning meetings preferred, March 15 deadline."},
    ],
    [
        {"role": "user", "content": "I have the AI Summit Tokyo conference on March 5th. I'm presenting a paper on context optimization for LLM inference. I need to prepare my slides."},
        {"role": "assistant", "content": "AI Summit Tokyo Mar 5, presenting on context optimization. Slides needed."},
    ],
    [
        {"role": "user", "content": "I'm learning Japanese, currently at JLPT N3 level. I practice daily. I also collect Ghibli merchandise and visited the Ghibli Museum in Mitaka last time."},
        {"role": "assistant", "content": "Japanese N3 level, daily practice. Ghibli collector, visited Mitaka museum."},
    ],
    [
        {"role": "user", "content": "I run 5km every morning so I always look for hotels with a gym or a nice running path nearby. I also enjoy onsen hotels for weekend trips."},
        {"role": "assistant", "content": "5km morning runs, hotel gym/running path needed. Onsen fan for weekends."},
    ],
    [
        {"role": "user", "content": "I have a 3-day trip to Tokyo coming up, March 4 to 7. My flight is ANA NH920 Shanghai-Tokyo departing 8am on March 4th."},
        {"role": "assistant", "content": "Tokyo trip Mar 4-7, ANA NH920 departs 8am Mar 4."},
    ],
    [
        {"role": "user", "content": "I want to meet my college friend Tanaka-san while I'm in Tokyo. We usually meet in Shinjuku for dinner."},
        {"role": "assistant", "content": "Meeting Tanaka-san in Shinjuku for dinner during Tokyo trip."},
    ],
]

AGENT_QUERIES = [
    "Plan my upcoming Tokyo trip next week, what do I need to prepare?",
    "What hotel should I book? Consider my past experience and preferences.",
    "Find me restaurants that match my dietary restrictions and food preferences in Tokyo.",
    "Check my flight details and mileage status for the Tokyo trip.",
    "What should I prepare for the AI Summit conference presentation?",
    "What fun activities and sightseeing can I do after the conference in Tokyo?",
    "I want to meet my friend Tanaka, what do you know about our past meetups?",
    "Give me a complete summary of everything relevant to this Tokyo trip.",
]


# =============================================================================
# Helpers
# =============================================================================

def build_prompt(query: str, memory_texts: List[str]) -> str:
    mem_block = "\n".join(f"[Memory {i+1}] {m}" for i, m in enumerate(memory_texts))
    return (
        f"You are a helpful travel assistant with access to the user's long-term memory.\n\n"
        f"## Relevant memories from past conversations:\n{mem_block}\n\n"
        f"## User's question:\n{query}\n\n"
        f"Provide a concise, helpful response based on the memories above:"
    )


def flush_sglang_cache():
    """Flush SGLang's radix cache."""
    r = requests.post(f"{SGLANG_URL}/flush_cache", timeout=10)
    return r.status_code == 200


def send_one(prompt: str) -> dict:
    """Send a single request to SGLang and return response + latency."""
    body = {
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    t0 = time.time()
    resp = requests.post(f"{SGLANG_URL}/v1/completions", json=body, timeout=120).json()
    latency = time.time() - t0
    return {**resp, "_latency": latency}


def extract_metrics(resp: dict) -> dict:
    """Extract cache/latency metrics from SGLang response."""
    usage = resp.get("usage", {})
    details = usage.get("prompt_tokens_details") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached = details.get("cached_tokens", 0) or 0
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "prefill_tokens": prompt_tokens - cached,
        "completion_tokens": usage.get("completion_tokens", 0),
        "latency": resp.get("_latency", 0),
        "answer": resp.get("choices", [{}])[0].get("text", "").strip(),
    }


def run_sequential(prompts: List[str], send_order: List[int], label: str) -> tuple:
    """
    Send prompts one at a time in the given order.
    Returns (metrics_in_original_order, wall_time).
    """
    n = len(prompts)
    metrics = [None] * n
    t0 = time.time()
    for step, orig_idx in enumerate(send_order):
        p = prompts[orig_idx]
        print(f"    [{step+1}/{n}] Sending query {orig_idx} ...", end=" ", flush=True)
        resp = send_one(p)
        m = extract_metrics(resp)
        metrics[orig_idx] = m
        print(f"prompt={m['prompt_tokens']:>4d}  cached={m['cached_tokens']:>4d}  "
              f"prefill={m['prefill_tokens']:>4d}  latency={m['latency']:.2f}s")
    wall_time = time.time() - t0
    return metrics, wall_time


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print(" A/B COMPARISON: Baseline vs ContextPilot  (batch_size=1, sequential)")
    print(" Prefill Latency | Cache Hit Rate | Answer Quality")
    print("=" * 80)

    # ---- Check services ----
    try:
        r = requests.get(f"{SGLANG_URL}/v1/models", timeout=3).json()
        model = r["data"][0]["id"]
        print(f"\n  SGLang:       running ({model})")
    except Exception as e:
        print(f"\n  SGLang:       NOT RUNNING — {e}")
        sys.exit(1)

    try:
        r = requests.get(f"{CONTEXTPILOT_URL}/health", timeout=3).json()
        print(f"  ContextPilot: {r['status']}")
    except Exception as e:
        print(f"  ContextPilot: NOT RUNNING — {e}")
        sys.exit(1)

    # ---- Step 1: Populate mem0 ----
    print(f"\n{'─' * 80}")
    print("STEP 1: Populate mem0 & retrieve memories")
    print(f"{'─' * 80}")

    retriever = Mem0Retriever(
        config={
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
        },
        use_integer_ids=True,
    )

    user_id = "demo_travel_user"
    agent_id = "travel_assistant"

    try:
        retriever.delete_all_memories(user_id=user_id, agent_id=agent_id)
    except Exception:
        pass

    print(f"  Adding {len(PAST_CONVERSATIONS)} conversations to mem0...")
    for conv in PAST_CONVERSATIONS:
        retriever.add_memory(conv, user_id=user_id, agent_id=agent_id)

    time.sleep(1)
    corpus = retriever.load_corpus_from_memories(user_id=user_id, agent_id=agent_id, limit=200)
    corpus_map = retriever.get_corpus_map()
    print(f"  {len(corpus)} memories extracted")

    # Retrieve contexts for all queries
    contexts = []
    for i, q in enumerate(AGENT_QUERIES):
        results = retriever.search_queries(
            query_data=[{"qid": i, "text": q}],
            user_id=user_id, agent_id=agent_id, top_k=TOP_K,
        )
        contexts.append(results[0]["top_k_doc_id"])

    # Build prompts in ORIGINAL retrieval order
    baseline_prompts = []
    for i, q in enumerate(AGENT_QUERIES):
        mem_texts = [corpus_map.get(str(d), {}).get("text", f"[doc {d}]") for d in contexts[i]]
        baseline_prompts.append(build_prompt(q, mem_texts))

    print(f"  {len(AGENT_QUERIES)} queries x top-{TOP_K} memories retrieved")

    # Show overlap stats
    all_docs = [d for c in contexts for d in c]
    unique_docs = set(all_docs)
    overlap_pct = (len(all_docs) - len(unique_docs)) / len(all_docs)
    print(f"  Retrieval redundancy: {overlap_pct:.0%} "
          f"({len(all_docs)} total refs, {len(unique_docs)} unique docs)")

    # ══════════════════════════════════════════════════════════════════════════
    # RUN A: Baseline — original order, one at a time
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 80}")
    print("RUN A: BASELINE — original order, original memory arrangement")
    print(f"{'─' * 80}")

    print("  Flushing SGLang radix cache...")
    if not flush_sglang_cache():
        print("  WARNING: cache flush failed")
    time.sleep(1)

    original_order = list(range(len(AGENT_QUERIES)))
    print(f"  Sending {len(baseline_prompts)} requests sequentially (batch_size=1):")
    baseline_metrics, baseline_wall = run_sequential(
        baseline_prompts, original_order, label="baseline"
    )

    print(f"\n  Completed in {baseline_wall:.2f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD: ContextPilot index to get optimal order + reordered memories
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 80}")
    print("BUILD: ContextPilot /build — compute optimal send order & memory arrangement")
    print(f"{'─' * 80}")

    requests.post(f"{CONTEXTPILOT_URL}/reset", json={})
    build_resp = requests.post(f"{CONTEXTPILOT_URL}/build", json={
        "contexts": contexts,
        "initial_tokens_per_context": 0,
        "alpha": 0.005,
        "use_gpu": False,
        "linkage_method": "average",
    }, timeout=30).json()

    request_ids = build_resp["request_ids"]
    scheduled_order = build_resp.get("scheduled_order", original_order)
    scheduled_reordered = build_resp.get("scheduled_reordered", None)

    print(f"  Request IDs: {request_ids}")
    print(f"  Original order:  {original_order}")
    print(f"  Scheduled order: {scheduled_order}")
    if scheduled_reordered:
        for i, (orig, reord) in enumerate(zip(contexts, scheduled_reordered)):
            if orig != reord:
                print(f"    Query {i}: memory order changed")

    # Build ContextPilot prompts with reordered memories
    cp_prompts = []
    for i in range(len(AGENT_QUERIES)):
        if scheduled_reordered and len(scheduled_reordered) == len(contexts):
            # Use reordered doc IDs for this query
            doc_ids = scheduled_reordered[i]
        else:
            # Fallback: same as baseline
            doc_ids = contexts[i]
        mem_texts = [corpus_map.get(str(d), {}).get("text", f"[doc {d}]") for d in doc_ids]
        cp_prompts.append(build_prompt(AGENT_QUERIES[i], mem_texts))

    # Verify prompts are non-empty
    for i, p in enumerate(cp_prompts):
        if len(p) < 50:
            print(f"  WARNING: query {i} prompt suspiciously short ({len(p)} chars)")

    # Show what changed
    same_count = sum(1 for i in range(len(contexts))
                     if not scheduled_reordered or contexts[i] == scheduled_reordered[i])
    print(f"  Memory reordering: {len(contexts) - same_count}/{len(contexts)} queries have different memory order")

    # ══════════════════════════════════════════════════════════════════════════
    # RUN B: ContextPilot — scheduled order, reordered memories, one at a time
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 80}")
    print("RUN B: CONTEXTPILOT — scheduled order, reordered memories")
    print(f"{'─' * 80}")

    print("  Flushing SGLang radix cache...")
    if not flush_sglang_cache():
        print("  WARNING: cache flush failed")
    time.sleep(1)

    print(f"  Sending {len(cp_prompts)} requests sequentially in scheduled order (batch_size=1):")
    cp_metrics, cp_wall = run_sequential(
        cp_prompts, scheduled_order, label="contextpilot"
    )

    print(f"\n  Completed in {cp_wall:.2f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print(" COMPARISON RESULTS")
    print(f"{'=' * 80}")

    b_cached = sum(m["cached_tokens"] for m in baseline_metrics)
    b_prefill = sum(m["prefill_tokens"] for m in baseline_metrics)
    b_prompt = sum(m["prompt_tokens"] for m in baseline_metrics)

    c_cached = sum(m["cached_tokens"] for m in cp_metrics)
    c_prefill = sum(m["prefill_tokens"] for m in cp_metrics)
    c_prompt = sum(m["prompt_tokens"] for m in cp_metrics)

    print(f"\n  {'Metric':<30s}  {'Baseline':>10s}  {'ContextPilot':>12s}  {'Delta':>10s}")
    print(f"  {'─' * 70}")

    # Cache hit rate
    b_hit_rate = b_cached / b_prompt * 100 if b_prompt else 0
    c_hit_rate = c_cached / c_prompt * 100 if c_prompt else 0
    print(f"  {'Total prompt tokens':<30s}  {b_prompt:>10d}  {c_prompt:>12d}")
    print(f"  {'Total cached tokens':<30s}  {b_cached:>10d}  {c_cached:>12d}  "
          f"{'+' if c_cached >= b_cached else ''}{c_cached - b_cached:>9d}")
    print(f"  {'Cache hit rate':<30s}  {b_hit_rate:>9.1f}%  {c_hit_rate:>11.1f}%  "
          f"{'+' if c_hit_rate >= b_hit_rate else ''}{c_hit_rate - b_hit_rate:>8.1f}%")

    # Prefill tokens (actual compute work)
    prefill_saved = b_prefill - c_prefill
    prefill_pct = prefill_saved / b_prefill * 100 if b_prefill else 0
    print(f"  {'Total prefill tokens':<30s}  {b_prefill:>10d}  {c_prefill:>12d}  "
          f"{'-' if prefill_saved > 0 else '+'}{abs(prefill_saved):>9d}")
    print(f"  {'Prefill reduction':<30s}  {'':>10s}  {'':>12s}  "
          f"{prefill_pct:>9.1f}%")

    # Wall time
    time_saved = baseline_wall - cp_wall
    print(f"  {'Total wall time':<30s}  {baseline_wall:>9.2f}s  {cp_wall:>11.2f}s  "
          f"{'-' if time_saved > 0 else '+'}{abs(time_saved):>8.2f}s")
    if cp_wall > 0:
        print(f"  {'Speedup':<30s}  {'':>10s}  {'':>12s}  "
              f"{baseline_wall / cp_wall:>9.2f}x")

    # Per-request comparison
    print(f"\n  {'─' * 80}")
    print(f"  Per-request detail:")
    print(f"  {'Q':<3s}  {'Query':<40s}  {'B.cache':>7s}  {'CP.cache':>8s}  "
          f"{'Delta':>6s}  {'B.lat':>6s}  {'CP.lat':>6s}")
    print(f"  {'─' * 80}")
    for i, q in enumerate(AGENT_QUERIES):
        bm = baseline_metrics[i]
        cm = cp_metrics[i]
        delta = cm["cached_tokens"] - bm["cached_tokens"]
        sign = "+" if delta >= 0 else ""
        print(f"  {i:<3d}  {q[:38]:<40s}  {bm['cached_tokens']:>7d}  {cm['cached_tokens']:>8d}  "
              f"{sign}{delta:>5d}  {bm['latency']:>5.2f}s  {cm['latency']:>5.2f}s")

    # ── Answer Quality (side-by-side) ──
    print(f"\n{'=' * 80}")
    print(" ANSWER QUALITY COMPARISON")
    print(f"{'=' * 80}")

    for i, q in enumerate(AGENT_QUERIES):
        bm = baseline_metrics[i]
        cm = cp_metrics[i]
        print(f"\n  ┌─ Q{i}: {q}")
        print(f"  │")
        print(f"  ├─ [Baseline] (cached={bm['cached_tokens']}):")
        for line in bm["answer"][:400].split("\n"):
            print(f"  │    {line}")
        print(f"  │")
        print(f"  ├─ [ContextPilot] (cached={cm['cached_tokens']}):")
        for line in cm["answer"][:400].split("\n"):
            print(f"  │    {line}")
        print(f"  └─────")

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print(" SUMMARY")
    print(f"{'=' * 80}")
    print(f"""
  Mode:               Sequential (batch_size=1)
  Queries:            {len(AGENT_QUERIES)}
  Memories per query: {TOP_K}
  Retrieval overlap:  {overlap_pct:.0%}

  Baseline send order:      {original_order}
  ContextPilot send order:  {scheduled_order}

  Cache hit rate:     {b_hit_rate:.1f}% (baseline) -> {c_hit_rate:.1f}% (ContextPilot)
  Prefill saved:      {prefill_saved} tokens ({prefill_pct:.1f}% reduction)
  Wall time:          {baseline_wall:.2f}s -> {cp_wall:.2f}s

  ContextPilot reorders memories WITHIN each prompt so that shared content
  appears at the front (prefix), and schedules the SEND ORDER so that requests
  with the most shared prefixes are sent consecutively. This maximizes radix
  cache reuse — each request benefits from KV states cached by previous ones.
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
