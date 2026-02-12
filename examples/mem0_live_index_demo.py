"""
Demo: mem0 Long-Term Memory × ContextPilot Live Index — End-to-End with SGLang

Full production path:
  mem0 retrieval → ContextPilot HTTP Server (/build) → SGLang inference (/v1/completions)
  with eviction tracking, multi-turn dedup, and incremental builds.

Shows:
1. **Retrieval overlap**: Real mem0 vector search returns heavily overlapping memories.
2. **Reorder benefit**: ContextPilot reorders contexts for maximum prefix sharing.
3. **E2E inference**: Requests go through ContextPilot proxy → SGLang with request_id tracking.
4. **Eviction sync**: SGLang's radix cache eviction calls back to ContextPilot.
5. **Incremental build**: Follow-up turns use incremental index update + deduplication.

SETUP (3 terminals):
  # Terminal 1: Apply SGLang patch & start SGLang
  cd /home/sicheng/ContextPilot
  bash patches/sglang/apply_patch.sh
  RAGBOOST_INDEX_URL=http://localhost:8765 python -m sglang.launch_server \
      --model-path Qwen/Qwen3-4B --tp-size 1 --schedule-policy lpm --port 30000

  # Terminal 2: Start ContextPilot server
  python -m contextpilot.server.http_server --port 8765 --infer-api-url http://localhost:30000

  # Terminal 3: Run this demo
  export OPENAI_API_KEY=sk-...
  python examples/mem0_live_index_demo.py

Requirements:
    pip install mem0ai openai
"""

import os
import sys
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from contextpilot.retriever import Mem0Retriever


# =============================================================================
# Config
# =============================================================================

CONTEXTPILOT_URL = os.environ.get("CONTEXTPILOT_URL", "http://localhost:8765")
TOP_K = 10  # memories per query

# =============================================================================
# 1. Real mem0 Memory Store — populate from past agent conversations
# =============================================================================

PAST_CONVERSATIONS = [
    # --- Travel preferences ---
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
    # --- Food & dietary ---
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
    # --- Past Tokyo trips ---
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
    # --- Work context ---
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
    # --- Personal interests ---
    [
        {"role": "user", "content": "I'm learning Japanese, currently at JLPT N3 level. I practice daily. I also collect Ghibli merchandise and visited the Ghibli Museum in Mitaka last time."},
        {"role": "assistant", "content": "Japanese N3 level, daily practice. Ghibli collector, visited Mitaka museum."},
    ],
    [
        {"role": "user", "content": "I run 5km every morning so I always look for hotels with a gym or a nice running path nearby. I also enjoy onsen hotels for weekend trips."},
        {"role": "assistant", "content": "5km morning runs, hotel gym/running path needed. Onsen fan for weekends."},
    ],
    # --- Upcoming trip ---
    [
        {"role": "user", "content": "I have a 3-day trip to Tokyo coming up, March 4 to 7. My flight is ANA NH920 Shanghai-Tokyo departing 8am on March 4th."},
        {"role": "assistant", "content": "Tokyo trip Mar 4-7, ANA NH920 departs 8am Mar 4."},
    ],
    [
        {"role": "user", "content": "I want to meet my college friend Tanaka-san while I'm in Tokyo. We usually meet in Shinjuku for dinner."},
        {"role": "assistant", "content": "Meeting Tanaka-san in Shinjuku for dinner during Tokyo trip."},
    ],
]


# =============================================================================
# 2. Helper functions
# =============================================================================

def check_services():
    """Check if ContextPilot server (and SGLang behind it) are running."""
    print("Checking services...")
    try:
        r = requests.get(f"{CONTEXTPILOT_URL}/health", timeout=3)
        health = r.json()
        print(f"  ContextPilot server: {health['status']} (mode={health.get('mode', '?')})")
    except Exception as e:
        print(f"  ContextPilot server: NOT RUNNING ({e})")
        print(f"\n  Start it with:")
        print(f"    python -m contextpilot.server.http_server --port 8765 --infer-api-url http://localhost:30000")
        return False

    # Check SGLang via the proxy
    try:
        r = requests.get(f"{CONTEXTPILOT_URL}/v1/models", timeout=5)
        models = r.json()
        model_id = models.get("data", [{}])[0].get("id", "unknown")
        print(f"  SGLang backend: running (model={model_id})")
    except Exception as e:
        print(f"  SGLang backend: NOT RUNNING ({e})")
        print(f"\n  Start it with:")
        print(f"    RAGBOOST_INDEX_URL=http://localhost:8765 python -m sglang.launch_server \\")
        print(f"        --model-path Qwen/Qwen3-4B --tp-size 1 --schedule-policy lpm --port 30000")
        return False

    return True


def compute_prefix_sharing(contexts: List[List[int]]) -> Dict[str, Any]:
    """Compute prefix sharing metrics between consecutive contexts."""
    pairs = []
    total_prefix = 0
    total_overlap = 0
    for i in range(len(contexts) - 1):
        a, b = contexts[i], contexts[i + 1]
        prefix_len = 0
        for x, y in zip(a, b):
            if x == y:
                prefix_len += 1
            else:
                break
        overlap = len(set(a) & set(b))
        pairs.append({"pair": (i, i + 1), "prefix_len": prefix_len, "set_overlap": overlap})
        total_prefix += prefix_len
        total_overlap += overlap
    return {
        "pairs": pairs,
        "total_prefix_sharing": total_prefix,
        "total_set_overlap": total_overlap,
        "avg_prefix_sharing": total_prefix / max(len(pairs), 1),
    }


def build_prompt(query: str, memory_texts: List[str]) -> str:
    """Build a prompt from memories + query."""
    mem_block = "\n".join(f"[Memory {i+1}] {m}" for i, m in enumerate(memory_texts))
    return (
        f"You are a helpful travel assistant with access to the user's long-term memory.\n\n"
        f"## Relevant memories from past conversations:\n{mem_block}\n\n"
        f"## User's question:\n{query}\n\n"
        f"Provide a concise, helpful response based on the memories above:"
    )


# =============================================================================
# 3. Main E2E demo
# =============================================================================

def main():
    print("=" * 75)
    print(" mem0 × ContextPilot × SGLang — End-to-End Demo")
    print("=" * 75)

    # ---- Check services ----
    if not check_services():
        print("\nPlease start both services and re-run.")
        sys.exit(1)

    # ---- Step 1: Populate mem0 ----
    print("\n" + "-" * 75)
    print("STEP 1: Populate mem0 with past agent conversations")
    print("-" * 75)

    retriever = Mem0Retriever(
        config={
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
        },
        use_integer_ids=True,
    )

    user_id = "demo_travel_user"
    agent_id = "travel_assistant"

    print(f"  Clearing old memories for user={user_id}...")
    try:
        retriever.delete_all_memories(user_id=user_id, agent_id=agent_id)
    except Exception:
        pass

    print(f"  Adding {len(PAST_CONVERSATIONS)} past conversations to mem0...")
    for i, conv in enumerate(PAST_CONVERSATIONS):
        retriever.add_memory(conv, user_id=user_id, agent_id=agent_id)
    print(f"    ✓ All {len(PAST_CONVERSATIONS)} conversations added")

    time.sleep(1)

    corpus = retriever.load_corpus_from_memories(user_id=user_id, agent_id=agent_id, limit=200)
    print(f"\n  ✓ mem0 extracted {len(corpus)} memories from conversations\n")
    for doc in corpus[:5]:
        print(f"    ID {doc['chunk_id']:>3d}: {doc['text'][:80]}...")
    if len(corpus) > 5:
        print(f"    ... and {len(corpus) - 5} more")

    # ---- Step 2: Retrieve memories for batch of agent turns ----
    print("\n" + "-" * 75)
    print("STEP 2: Retrieve memories for 8 agent conversation turns")
    print("-" * 75)

    agent_queries = [
        "Plan my upcoming Tokyo trip next week, what do I need to prepare?",
        "What hotel should I book? Consider my past experience and preferences.",
        "Find me restaurants that match my dietary restrictions and food preferences in Tokyo.",
        "Check my flight details and mileage status for the Tokyo trip.",
        "What should I prepare for the AI Summit conference presentation?",
        "What fun activities and sightseeing can I do after the conference in Tokyo?",
        "I want to meet my friend Tanaka, what do you know about our past meetups?",
        "Give me a complete summary of everything relevant to this Tokyo trip.",
    ]

    contexts = []
    for turn_idx, query in enumerate(agent_queries):
        results = retriever.search_queries(
            query_data=[{"qid": turn_idx, "text": query}],
            user_id=user_id, agent_id=agent_id, top_k=TOP_K,
        )
        doc_ids = results[0]["top_k_doc_id"]
        contexts.append(doc_ids)
        print(f"\n  Turn {turn_idx}: \"{query[:60]}...\"")
        print(f"    Retrieved IDs: {doc_ids}")

    corpus_map = retriever.get_corpus_map()

    # ---- Step 3: Show retrieval overlap ----
    print("\n" + "-" * 75)
    print("STEP 3: Retrieval Overlap Analysis (this is why ContextPilot helps)")
    print("-" * 75)

    n = len(contexts)
    print(f"\n  Pairwise set overlap (each turn retrieves {TOP_K} memories):\n")
    header = "        " + "  ".join(f"T{i}" for i in range(n))
    print(header)
    for i in range(n):
        row = f"  T{i}    "
        for j in range(n):
            ovl = len(set(contexts[i]) & set(contexts[j]))
            row += f" · " if i == j else f"{ovl:2d} "
        print(row)

    all_docs = [d for c in contexts for d in c]
    unique_docs = set(all_docs)
    print(f"\n  Total retrievals: {len(all_docs)}, Unique: {len(unique_docs)}, "
          f"Redundancy: {(len(all_docs) - len(unique_docs)) / len(all_docs):.0%}")

    baseline = compute_prefix_sharing(contexts)
    print(f"  Baseline prefix sharing (original order): {baseline['total_prefix_sharing']} "
          f"(avg {baseline['avg_prefix_sharing']:.1f})")

    # ---- Step 4: Reset ContextPilot + Build Live Index via HTTP ----
    print("\n" + "-" * 75)
    print("STEP 4: Build Live Index via ContextPilot HTTP Server (/build)")
    print("-" * 75)

    # Reset any prior state
    requests.post(f"{CONTEXTPILOT_URL}/reset", json={})
    print("  ✓ Index reset")

    # Build index — this does clustering, reordering, scheduling, and assigns request_ids
    build_resp = requests.post(f"{CONTEXTPILOT_URL}/build", json={
        "contexts": contexts,
        "initial_tokens_per_context": 0,
        "alpha": 0.005,
        "use_gpu": False,
        "linkage_method": "average",
    }, timeout=30).json()

    request_ids = build_resp["request_ids"]
    stats = build_resp.get("stats", {})

    print(f"  ✓ Index built: {stats.get('total_nodes', '?')} nodes, "
          f"{stats.get('leaf_nodes', '?')} leaves")
    print(f"  ✓ Request IDs assigned: {request_ids}")

    # Get current tracked requests
    req_resp = requests.get(f"{CONTEXTPILOT_URL}/requests").json()
    print(f"  ✓ Server tracking {req_resp['num_requests']} requests")

    # ---- Step 5: Send inference requests through ContextPilot proxy ----
    print("\n" + "-" * 75)
    print("STEP 5: E2E Inference — ContextPilot proxy → SGLang (/v1/completions)")
    print("-" * 75)
    print("  Sending ALL requests concurrently so SGLang's LPM scheduler can")
    print("  batch them and exploit prefix sharing from ContextPilot's reordering.")
    print("  Each request carries request_id → SGLang tracks it in radix cache")
    print("  → eviction callback notifies ContextPilot → index stays in sync\n")

    # Build all prompts first
    prompts = []
    for turn_idx, (query, rid) in enumerate(zip(agent_queries, request_ids)):
        doc_ids = contexts[turn_idx]
        mem_texts = [corpus_map.get(str(did), {}).get("text", f"[memory {did}]") for did in doc_ids]
        prompts.append(build_prompt(query, mem_texts))

    # Submit all requests concurrently — SGLang batches them with LPM scheduling
    def send_request(turn_idx, prompt, rid):
        resp = requests.post(f"{CONTEXTPILOT_URL}/v1/completions", json={
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.7,
            "request_id": rid,
        }, timeout=120).json()
        return turn_idx, resp

    t0 = time.time()
    responses = [None] * len(agent_queries)
    with ThreadPoolExecutor(max_workers=len(agent_queries)) as pool:
        futures = {
            pool.submit(send_request, i, p, r): i
            for i, (p, r) in enumerate(zip(prompts, request_ids))
        }
        for future in as_completed(futures):
            turn_idx, resp = future.result()
            responses[turn_idx] = resp
    batch_time = time.time() - t0

    print(f"  All {len(agent_queries)} requests completed in {batch_time:.2f}s (concurrent batch)\n")
    for turn_idx, (query, rid) in enumerate(zip(agent_queries, request_ids)):
        resp = responses[turn_idx]
        if resp:
            answer = resp.get("choices", [{}])[0].get("text", "").strip()
            usage = resp.get("usage", {})
            print(f"  Turn {turn_idx} (rid={rid}): \"{query[:55]}...\"")
            print(f"    → tokens: prompt={usage.get('prompt_tokens', '?')}, "
                  f"completion={usage.get('completion_tokens', '?')}, "
                  f"total={usage.get('total_tokens', '?')}")
            print(f"    → answer: {answer[:120]}...")
        else:
            print(f"  Turn {turn_idx} (rid={rid}): ERROR")

    # ---- Step 6: Check index state after inference ----
    print("\n" + "-" * 75)
    print("STEP 6: Index State After Inference")
    print("-" * 75)

    stats_resp = requests.get(f"{CONTEXTPILOT_URL}/stats").json()
    idx_stats = stats_resp.get("index_stats", {})
    print(f"  Total nodes:    {idx_stats.get('total_nodes', '?')}")
    print(f"  Leaf nodes:     {idx_stats.get('leaf_nodes', '?')}")
    print(f"  Live requests:  {idx_stats.get('live_requests', '?')}")
    print(f"  Total tokens:   {idx_stats.get('total_tokens', '?')}")

    req_resp = requests.get(f"{CONTEXTPILOT_URL}/requests").json()
    print(f"  Tracked request IDs: {req_resp.get('request_ids', [])}")

    # ---- Step 7: Simulate SGLang eviction callback ----
    print("\n" + "-" * 75)
    print("STEP 7: Simulate SGLang Eviction (POST /evict)")
    print("-" * 75)
    print("  In production, SGLang's patched radix_cache calls /evict automatically.")
    print("  Here we simulate evicting the first 2 requests.\n")

    evict_ids = request_ids[:2]
    evict_resp = requests.post(f"{CONTEXTPILOT_URL}/evict", json={
        "request_ids": evict_ids,
    }).json()

    print(f"  Evicted: {evict_ids}")
    print(f"  Result: removed={evict_resp.get('removed_count', '?')}, "
          f"not_found={evict_resp.get('not_found', [])}")

    # Check remaining
    req_resp = requests.get(f"{CONTEXTPILOT_URL}/requests").json()
    print(f"  Remaining requests: {req_resp.get('num_requests', '?')} → {req_resp.get('request_ids', [])}")

    # ---- Step 8: Incremental build + deduplication (follow-up turn) ----
    print("\n" + "-" * 75)
    print("STEP 8: Incremental Build + Dedup (follow-up conversation turn)")
    print("-" * 75)

    follow_up_queries = [
        "Which transit card should I use in Tokyo? Suica or Pasmo?",
        "Remind me about the AI Summit schedule and my presentation topic.",
    ]

    follow_contexts = []
    for q in follow_up_queries:
        results = retriever.search_queries(
            query_data=[{"qid": 99, "text": q}],
            user_id=user_id, agent_id=agent_id, top_k=TOP_K,
        )
        follow_contexts.append(results[0]["top_k_doc_id"])
        print(f"  Query: \"{q[:60]}\"")
        print(f"    Retrieved IDs: {results[0]['top_k_doc_id']}")

    # Use the last surviving request_id as parent (for dedup chain)
    surviving_rids = req_resp.get("request_ids", [])
    parent_ids = [surviving_rids[-1] if surviving_rids else None] * len(follow_contexts)

    incr_resp = requests.post(f"{CONTEXTPILOT_URL}/build", json={
        "contexts": follow_contexts,
        "incremental": True,
        "deduplicate": True,
        "parent_request_ids": parent_ids,
    }, timeout=30).json()

    new_rids = incr_resp.get("request_ids", [])
    print(f"\n  ✓ Incremental build: matched={incr_resp.get('matched_count', '?')}, "
          f"merged={incr_resp.get('merged_count', '?')}")
    print(f"  ✓ New request IDs: {new_rids}")

    if "deduplication" in incr_resp:
        dedup = incr_resp["deduplication"]
        print(f"  ✓ Deduplication: {dedup.get('total_docs_deduplicated', 0)} docs deduplicated")
        for r in dedup.get("results", []):
            print(f"    rid={r['request_id']}: "
                  f"original={len(r['original_docs'])}, "
                  f"deduped={len(r['deduplicated_docs'])}, "
                  f"overlap={len(r['overlapping_docs'])}")

    # Send follow-up inference concurrently
    print(f"\n  Sending follow-up inference requests concurrently...")
    follow_prompts = [
        build_prompt(q, [corpus_map.get(str(d), {}).get("text", f"[{d}]") for d in ctx])
        for q, ctx in zip(follow_up_queries, follow_contexts)
    ]

    follow_responses = [None] * len(follow_up_queries)
    with ThreadPoolExecutor(max_workers=len(follow_up_queries)) as pool:
        futures = {
            pool.submit(send_request, i, p, r): i
            for i, (p, r) in enumerate(zip(follow_prompts, new_rids))
        }
        for future in as_completed(futures):
            idx, resp = future.result()
            follow_responses[idx] = resp

    for i, (q, rid) in enumerate(zip(follow_up_queries, new_rids)):
        resp = follow_responses[i]
        if resp:
            answer = resp.get("choices", [{}])[0].get("text", "").strip()
            usage = resp.get("usage", {})
            print(f"    [{rid}] \"{q[:50]}...\"")
            print(f"      → {answer[:100]}...")
            print(f"      → tokens: {usage.get('total_tokens', '?')}")
        else:
            print(f"    [{rid}] ERROR")

    # ---- Final state ----
    print("\n" + "-" * 75)
    print("STEP 9: Final Index State")
    print("-" * 75)

    final_stats = requests.get(f"{CONTEXTPILOT_URL}/stats").json()
    final_reqs = requests.get(f"{CONTEXTPILOT_URL}/requests").json()
    idx = final_stats.get("index_stats", {})
    print(f"  Total nodes:     {idx.get('total_nodes', '?')}")
    print(f"  Leaf nodes:      {idx.get('leaf_nodes', '?')}")
    print(f"  Live requests:   {final_reqs.get('num_requests', '?')}")
    print(f"  Request IDs:     {final_reqs.get('request_ids', [])}")

    # ---- Summary ----
    print("\n" + "=" * 75)
    print(" E2E DEMO SUMMARY")
    print("=" * 75)
    print(f"""
  1. mem0 stored {len(corpus)} long-term memories from {len(PAST_CONVERSATIONS)} conversations
  2. 8 agent turns each retrieved top-{TOP_K} memories (redundancy: {(len(all_docs) - len(unique_docs)) / len(all_docs):.0%})
  3. ContextPilot /build clustered + reordered + scheduled:
     - Baseline prefix sharing:    {baseline['total_prefix_sharing']}
  4. Inference sent via proxy with request_id → SGLang radix cache tracks them
  5. Eviction callback /evict removed {len(evict_ids)} requests → index auto-pruned
  6. Incremental /build added {len(follow_up_queries)} follow-up turns with deduplication
  7. Full loop: mem0 → ContextPilot → SGLang → eviction → incremental update
""")
    print("=" * 75)


if __name__ == "__main__":
    main()
