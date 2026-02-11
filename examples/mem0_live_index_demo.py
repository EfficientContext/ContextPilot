"""
Demo: mem0 Long-Term Memory with Live Index — Real Retrieval Overlap & Reorder Benefit

This demo uses REAL mem0 retrieval to show two key things:
1. **Retrieval overlap**: Multiple turns of an agent conversation retrieve
   heavily overlapping memories from mem0's vector store.
2. **Reorder benefit**: ContextPilot's intra-context reordering + inter-context
   scheduling significantly improves prefix sharing → KV cache hit rate.

Requirements:
    - pip install mem0ai openai
    - export OPENAI_API_KEY=sk-...

Usage:
    cd /home/sicheng/ContextPilot
    python examples/mem0_live_index_demo.py
"""

import os
import sys
import time
from typing import List, Dict, Any

from contextpilot.retriever import Mem0Retriever
from contextpilot.context_index import ContextIndex
from contextpilot.context_ordering import InterContextScheduler


# =============================================================================
# 1. Real mem0 Memory Store — populate from past agent conversations
# =============================================================================

# These are real conversations that a travel assistant agent has had with the
# user over weeks/months. mem0 extracts facts and stores embeddings.
PAST_CONVERSATIONS = [
    # --- Travel preferences (across multiple past sessions) ---
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

    # --- Food & dietary (multiple sessions) ---
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

    # --- Upcoming trip planning ---
    [
        {"role": "user", "content": "I have a 3-day trip to Tokyo coming up, March 4 to 7. My flight is ANA NH920 Shanghai-Tokyo departing 8am on March 4th."},
        {"role": "assistant", "content": "Tokyo trip Mar 4-7, ANA NH920 departs 8am Mar 4."},
    ],
    [
        {"role": "user", "content": "I want to meet my college friend Tanaka-san while I'm in Tokyo. We usually meet in Shinjuku for dinner."},
        {"role": "assistant", "content": "Meeting Tanaka-san in Shinjuku for dinner during Tokyo trip."},
    ],
]


# Agent's new multi-turn conversation queries (these are the actual search queries)
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

TOP_K = 10  # memories per query


# =============================================================================
# 2. Metric: compute prefix sharing between contexts
# =============================================================================

def compute_prefix_sharing(contexts: List[List[int]]) -> Dict[str, Any]:
    """
    Compute prefix sharing metrics for an ordered list of contexts.
    
    Prefix sharing = longest common prefix between consecutive contexts.
    This directly maps to KV cache reuse in SGLang.
    """
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
        pairs.append({
            "pair": (i, i + 1),
            "prefix_len": prefix_len,
            "set_overlap": overlap,
            "len_a": len(a),
            "len_b": len(b),
        })
        total_prefix += prefix_len
        total_overlap += overlap

    return {
        "pairs": pairs,
        "total_prefix_sharing": total_prefix,
        "total_set_overlap": total_overlap,
        "avg_prefix_sharing": total_prefix / max(len(pairs), 1),
        "avg_set_overlap": total_overlap / max(len(pairs), 1),
        "num_contexts": len(contexts),
    }


# =============================================================================
# 3. Main demo
# =============================================================================

def main():
    print("=" * 75)
    print(" mem0 Long-Term Memory × ContextPilot Live Index Demo")
    print(" REAL mem0 retrieval — showing overlap + reorder benefit")
    print("=" * 75)

    # ---- Step 1: Initialize mem0 and populate long-term memories ----
    print("\n" + "-" * 75)
    print("STEP 1: Populate mem0 with past agent conversations")
    print("-" * 75)

    retriever = Mem0Retriever(
        config={
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-4o-mini"},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"},
            },
        },
        use_integer_ids=True,
    )

    user_id = "demo_travel_user"
    agent_id = "travel_assistant"

    # Clear any stale data from prior runs
    print(f"  Clearing old memories for user={user_id}...")
    try:
        retriever.delete_all_memories(user_id=user_id, agent_id=agent_id)
    except Exception:
        pass  # may not exist yet

    print(f"  Adding {len(PAST_CONVERSATIONS)} past conversations to mem0...")
    for i, conv in enumerate(PAST_CONVERSATIONS):
        retriever.add_memory(conv, user_id=user_id, agent_id=agent_id)
        print(f"    [{i+1}/{len(PAST_CONVERSATIONS)}] added")

    # Let mem0 process
    time.sleep(1)

    # Verify stored memories
    corpus = retriever.load_corpus_from_memories(
        user_id=user_id, agent_id=agent_id, limit=200
    )
    print(f"\n  ✓ mem0 now has {len(corpus)} extracted memories for this user")
    print(f"    (mem0 extracts & deduplicates facts from conversations)\n")
    for doc in corpus[:5]:
        print(f"    ID {doc['chunk_id']:>3d}: {doc['text'][:80]}...")
    if len(corpus) > 5:
        print(f"    ... and {len(corpus) - 5} more memories")

    # ---- Step 2: Real retrieval — 8 agent turns ----
    print("\n" + "-" * 75)
    print(f"STEP 2: Real mem0 Retrieval — {len(AGENT_QUERIES)} agent turns (top_k={TOP_K})")
    print("-" * 75)

    contexts_original = []
    query_results = []

    for turn_idx, query in enumerate(AGENT_QUERIES):
        results = retriever.search_queries(
            query_data=[{"qid": turn_idx, "text": query}],
            user_id=user_id,
            agent_id=agent_id,
            top_k=TOP_K,
        )
        doc_ids = results[0]["top_k_doc_id"]
        contexts_original.append(doc_ids)
        query_results.append(results[0])

        print(f"\n  Turn {turn_idx}: \"{query}\"")
        print(f"    Retrieved IDs: {doc_ids}")
        # Show top 3 memory texts
        for mem in results[0].get("memories", [])[:3]:
            print(f"      • {mem.get('memory', '')[:75]}...")

    corpus_map = retriever.get_corpus_map()

    # ---- Step 3: Show retrieval overlap ----
    print("\n" + "-" * 75)
    print("STEP 3: Retrieval Overlap Analysis")
    print("-" * 75)

    n_turns = len(contexts_original)
    ctx_sizes = [len(c) for c in contexts_original]

    print(f"\n  Pairwise set overlap:")
    print(f"  (each turn retrieved {TOP_K} memories from mem0's vector store)\n")
    header = "        " + "  ".join(f"T{i}" for i in range(n_turns))
    print(header)
    for i in range(n_turns):
        row = f"  T{i}    "
        for j in range(n_turns):
            overlap = len(set(contexts_original[i]) & set(contexts_original[j]))
            if i == j:
                row += f" · "
            else:
                row += f"{overlap:2d} "
        print(row)

    all_retrieved = [doc for ctx in contexts_original for doc in ctx]
    unique_retrieved = set(all_retrieved)
    total_memories = len(corpus)
    print(f"\n  Total retrieved across all turns: {len(all_retrieved)}")
    print(f"  Unique memories retrieved:        {len(unique_retrieved)} / {total_memories}")
    print(f"  Redundant retrievals:             {len(all_retrieved) - len(unique_retrieved)}")
    print(f"  Redundancy rate:                  "
          f"{(len(all_retrieved) - len(unique_retrieved)) / len(all_retrieved):.1%}")

    # ---- Step 4: Baseline prefix sharing (original retrieval order) ----
    print("\n" + "-" * 75)
    print("STEP 4: Baseline — Prefix Sharing WITHOUT Reordering")
    print("-" * 75)

    baseline_metrics = compute_prefix_sharing(contexts_original)
    print(f"\n  Contexts in original retrieval order:\n")
    for i, ctx in enumerate(contexts_original):
        print(f"    [{i}] {ctx}")

    print(f"\n  Consecutive prefix sharing:")
    for p in baseline_metrics["pairs"]:
        i, j = p["pair"]
        print(f"    ({i},{j}): prefix_len={p['prefix_len']:2d}, "
              f"set_overlap={p['set_overlap']:2d}/{min(p['len_a'], p['len_b'])}")

    print(f"\n  ► Total prefix sharing:  {baseline_metrics['total_prefix_sharing']}")
    print(f"  ► Avg prefix per pair:   {baseline_metrics['avg_prefix_sharing']:.2f}")
    print(f"  ► Total set overlap:     {baseline_metrics['total_set_overlap']}")

    # ---- Step 5: ContextPilot reordering ----
    print("\n" + "-" * 75)
    print("STEP 5: ContextPilot — Build Index + Reorder + Schedule")
    print("-" * 75)

    index = ContextIndex(use_gpu=False, alpha=0.005, linkage_method="average")
    result = index.fit_transform(contexts_original)

    scheduler = InterContextScheduler()
    scheduled_reordered, scheduled_originals, final_mapping, groups = \
        scheduler.schedule_contexts(result)

    print(f"\n  Clustering: {result.stats['total_nodes']} nodes, "
          f"{result.stats['leaf_nodes']} leaves")
    print(f"  Execution groups: {len(groups)}")
    print(f"  Scheduled order: {final_mapping}")
    print(f"\n  Contexts after intra-reorder + inter-schedule:\n")
    for idx, ctx in enumerate(scheduled_reordered):
        orig_idx = final_mapping[idx]
        query_short = AGENT_QUERIES[orig_idx][:45]
        print(f"    [{idx}] (Turn {orig_idx}: {query_short}...)")
        print(f"        {ctx}")

    # ---- Step 6: Reordered prefix sharing ----
    print("\n" + "-" * 75)
    print("STEP 6: Prefix Sharing AFTER Reordering + Scheduling")
    print("-" * 75)

    reordered_metrics = compute_prefix_sharing(scheduled_reordered)

    print(f"\n  Consecutive prefix sharing:")
    for p in reordered_metrics["pairs"]:
        i, j = p["pair"]
        orig_i = final_mapping[i]
        orig_j = final_mapping[j]
        print(f"    ({i},{j}) [Turn {orig_i}→{orig_j}]: "
              f"prefix_len={p['prefix_len']:2d}, set_overlap={p['set_overlap']:2d}")

    print(f"\n  ► Total prefix sharing:  {reordered_metrics['total_prefix_sharing']}")
    print(f"  ► Avg prefix per pair:   {reordered_metrics['avg_prefix_sharing']:.2f}")
    print(f"  ► Total set overlap:     {reordered_metrics['total_set_overlap']}")

    # ---- Step 7: Comparison ----
    print("\n" + "=" * 75)
    print(" COMPARISON SUMMARY")
    print("=" * 75)

    b_pfx = baseline_metrics["total_prefix_sharing"]
    r_pfx = reordered_metrics["total_prefix_sharing"]
    b_ovl = baseline_metrics["total_set_overlap"]
    r_ovl = reordered_metrics["total_set_overlap"]

    pfx_delta = r_pfx - b_pfx
    pfx_pct = (pfx_delta / b_pfx * 100) if b_pfx > 0 else float('inf')

    print(f"""
                           Baseline    ContextPilot    Improvement
                           --------    ------------    -----------
  Total prefix sharing:    {b_pfx:>6d}      {r_pfx:>8d}        {'+' if pfx_delta >= 0 else ''}{pfx_delta:d}{f' ({pfx_pct:+.0f}%)' if b_pfx > 0 else ' (∞)'}
  Avg prefix per pair:     {baseline_metrics['avg_prefix_sharing']:>6.2f}      {reordered_metrics['avg_prefix_sharing']:>8.2f}
  Total set overlap:       {b_ovl:>6d}      {r_ovl:>8d}        {'+' if r_ovl >= b_ovl else ''}{r_ovl - b_ovl:d}
""")

    print("  Prefix sharing = KV cache reuse.")
    print("  Higher prefix sharing → fewer tokens recomputed → faster inference.\n")

    # ---- Step 8: Show memory content for one overlap pair ----
    print("-" * 75)
    print("STEP 7: Example — Overlapping memories between two turns")
    print("-" * 75)

    # Pick two turns with highest overlap
    best_overlap = 0
    best_pair = (0, 1)
    for i in range(n_turns):
        for j in range(i + 1, n_turns):
            ovl = len(set(contexts_original[i]) & set(contexts_original[j]))
            if ovl > best_overlap:
                best_overlap = ovl
                best_pair = (i, j)

    ti, tj = best_pair
    shared_ids = set(contexts_original[ti]) & set(contexts_original[tj])
    print(f"\n  Turn {ti}: \"{AGENT_QUERIES[ti][:60]}...\"")
    print(f"  Turn {tj}: \"{AGENT_QUERIES[tj][:60]}...\"")
    print(f"  Shared memories: {len(shared_ids)} / {TOP_K}\n")
    for sid in list(shared_ids)[:6]:
        mem_data = corpus_map.get(str(sid), {})
        text = mem_data.get("text", f"[ID {sid}]")
        print(f"    ID {sid:>3d}: {text[:80]}...")

    print(f"\n  → With ContextPilot, these shared memories are placed at the FRONT")
    print(f"    of both contexts (intra-reorder), and the two turns are scheduled")
    print(f"    consecutively (inter-schedule) → maximum KV cache prefix reuse.\n")

    # ---- Step 9: Tree structure ----
    print("-" * 75)
    print("STEP 8: Cluster Tree Structure")
    print("-" * 75)
    result.print_tree()

    print("\n" + "=" * 75)
    print(" Demo complete. To use with SGLang in production:")
    print("   1. bash patches/sglang/apply_patch.sh")
    print("   2. python -m contextpilot.server.http_server --port 8765")
    print("   3. RAGBOOST_INDEX_URL=http://localhost:8765 python -m sglang.launch_server \\")
    print("        --model-path Qwen/Qwen3-4B --tp-size 1 --schedule-policy lpm --port 30000")
    print("   4. POST /build with your mem0-retrieved doc_id contexts")
    print("   5. POST /v1/completions with request_id for each turn")
    print("=" * 75)


if __name__ == "__main__":
    main()
