#!/usr/bin/env python3
"""
PageIndex + ContextPilot End-to-End Example
============================================

Demonstrates the complete workflow using a real PageIndex tree
(Walt Disney Q1 FY25 Earnings Report) with ContextPilot
optimization for efficient LLM inference.

Key insight: When multiple queries retrieve overlapping document nodes,
ContextPilot reorders documents within each context so shared documents
form the longest possible common prefix. A radix-tree KV cache can then
serve those prefix tokens from cache instead of recomputing them.

Workflow:
    1. Load a PageIndex tree (pre-built from PDF via PageIndex)
    2. Use PageIndexRetriever for LLM tree search (or simulated queries in demo mode)
    3. Feed contexts (lists of node IDs) to ContextPilot
    4. ContextPilot clusters, reorders within-context, and schedules
    5. Measure prefix sharing improvement (LCP metric)

Run:
    python examples/pageindex_e2e_example.py                         # demo (no API)
    python examples/pageindex_e2e_example.py --tree tree.json        # demo with custom tree
    python examples/pageindex_e2e_example.py --tree tree.json -q "query"  # full LLM pipeline

Tree data:
    The repo includes a pre-built tree at examples/data/disney_q1_fy25_tree.json
    (41 nodes from the Walt Disney Q1 FY25 earnings report).
    Generate your own with PageIndex:
        pip install pageindex  # cloud API SDK
        # See https://github.com/yinsicheng/PageIndex for usage
"""

import json
import os
import time
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# ContextPilot imports
import contextpilot as cp


# ============================================================================
# Helper: Tree Utilities
# ============================================================================

def flatten_tree(structure) -> List[Dict[str, Any]]:
    """Flatten a PageIndex tree to a list of nodes."""
    results = []
    def traverse(node):
        if isinstance(node, dict):
            results.append(node)
            for child in node.get("nodes", []):
                traverse(child)
        elif isinstance(node, list):
            for item in node:
                traverse(item)
    traverse(structure)
    return results


# ============================================================================
# ContextPilot: Build + Schedule + Measure
# ============================================================================

def longest_common_prefix(a: List[int], b: List[int]) -> int:
    """Count matching tokens from the start of two lists."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def measure_prefix_sharing(
    order: List[int],
    reordered_contexts: List[List[int]],
) -> Tuple[int, int]:
    """
    Compute total prefix tokens reusable from KV cache.

    For each consecutive pair in execution order, count how many tokens
    at the start of the current context match the previous one
    (= KV cache hits in a radix-tree cache).

    Returns (total_reused, total_new).
    """
    total_reused = 0
    total_new = 0
    for i in range(len(order)):
        ctx = reordered_contexts[i]
        if i == 0:
            lcp = 0
        else:
            lcp = longest_common_prefix(reordered_contexts[i - 1], ctx)
        total_reused += lcp
        total_new += len(ctx) - lcp
    return total_reused, total_new


def run_contextpilot(
    contexts: List[List[int]],
    query_labels: List[str],
    use_gpu: bool = False,
    alpha: float = 0.005,
) -> Dict[str, Any]:
    """
    Build a ContextPilot index and schedule contexts.

    Args:
        contexts: Each element is one query's document-ID list,
                  e.g. [[2, 8, 9, 31], [2, 12, 14, 34], ...]
        query_labels: Human-readable label per context (for display).
        use_gpu: GPU acceleration for distance matrix.
        alpha: Position weight in distance metric.

    Returns:
        Dictionary with scheduled order, groups, reordered contexts,
        and prefix-sharing metrics (scheduled vs naive vs random).
    """
    n = len(contexts)
    total_docs = sum(len(c) for c in contexts)

    # ── Document-level overlap ──
    all_ids = [d for c in contexts for d in c]
    unique_ids = set(all_ids)
    overlap = {d: cnt for d, cnt in Counter(all_ids).items() if cnt > 1}
    overlap_ratio = 1 - len(unique_ids) / len(all_ids) if all_ids else 0

    print(f"  Contexts:       {n}")
    print(f"  Total doc refs: {len(all_ids)}")
    print(f"  Unique docs:    {len(unique_ids)}")
    print(f"  Overlap ratio:  {overlap_ratio:.1%}")
    if overlap:
        print(f"  Shared docs:    {overlap}")

    # ── Reorder with ContextPilot ──
    engine = cp.ContextPilot(use_gpu=use_gpu, alpha=alpha)
    t0 = time.time()
    reordered_contexts, original_indices = engine.reorder(contexts)
    reorder_time = time.time() - t0
    print(f"\n  Reorder time:   {reorder_time:.3f}s")

    # ── Display scheduled order ──
    print(f"\n  Scheduled execution order:")
    for i, q_idx in enumerate(original_indices):
        label = query_labels[q_idx]
        reordered = reordered_contexts[i]
        original = contexts[q_idx]
        changed = reordered != original
        lcp = 0
        if i > 0:
            lcp = longest_common_prefix(reordered_contexts[i - 1], reordered)
        bar = "█" * (lcp * 3) if lcp > 0 else ""
        mark = " ← reordered" if changed else ""
        print(
            f"    [{q_idx}] {label:35s}  "
            f"docs={reordered!s:28s}  LCP={lcp}  {bar}{mark}"
        )

    # ── Prefix sharing: scheduled ──
    sched_reused, sched_new = measure_prefix_sharing(
        original_indices, reordered_contexts
    )

    # ── Prefix sharing: naive (original order, no reordering) ──
    naive_reused, naive_new = measure_prefix_sharing(
        list(range(n)), contexts
    )

    # ── Prefix sharing: random baseline (avg 100 shuffles) ──
    rng = random.Random(42)
    rand_total = 0
    for _ in range(100):
        perm = list(range(n))
        rng.shuffle(perm)
        shuffled = [contexts[i] for i in perm]
        r, _ = measure_prefix_sharing(perm, shuffled)
        rand_total += r
    rand_avg = rand_total / 100

    # ── Summary ──
    print(f"\n  Prefix sharing (Longest Common Prefix):")
    print(f"    {'Method':<20s} {'Reused':>8s} {'New':>8s} {'% Saved':>8s} {'vs Random':>10s}")
    print(f"    {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for label, reused in [
        ("ContextPilot", sched_reused),
        ("Naive", naive_reused),
        ("Random (avg)", rand_avg),
    ]:
        pct = reused / total_docs * 100 if total_docs else 0
        vs_rand = (
            f"+{(reused - rand_avg) / rand_avg * 100:.0f}%"
            if rand_avg > 0
            else "n/a"
        )
        if label == "Random (avg)":
            vs_rand = "baseline"
        print(
            f"    {label:<20s} {reused:>8.0f} "
            f"{total_docs - reused:>8.0f} {pct:>7.1f}% {vs_rand:>10s}"
        )

    return {
        "original_indices": original_indices,
        "reordered_contexts": reordered_contexts,
        "reorder_time": reorder_time,
        "overlap_ratio": overlap_ratio,
        "prefix_sharing": {
            "scheduled": sched_reused,
            "naive": naive_reused,
            "random_avg": rand_avg,
            "total_docs": total_docs,
        },
    }


# ============================================================================
# Tree Loading
# ============================================================================

# Default tree data bundled with the repo
DEFAULT_TREE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "disney_q1_fy25_tree.json"
)


def load_tree(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a PageIndex tree JSON.

    If no path is given, loads the bundled Disney Q1 FY25 tree
    (41 nodes, generated from the actual SEC filing).

    Generate your own tree with PageIndex:
        pip install pageindex
        # See https://github.com/yinsicheng/PageIndex
    """
    tree_path = path or DEFAULT_TREE_PATH
    if not os.path.isfile(tree_path):
        raise FileNotFoundError(
            f"Tree file not found: {tree_path}\n"
            f"Generate one with PageIndex: pip install pageindex\n"
            f"See https://github.com/yinsicheng/PageIndex"
        )
    with open(tree_path) as f:
        return json.load(f)


def demo_disney(tree_path: Optional[str] = None):
    """
    Demo using a PageIndex tree (defaults to the bundled Disney tree).

    Simulates 6 analyst queries with overlapping node retrieval.
    Each context is a list of integer node IDs — exactly how a real
    PageIndex pipeline would feed them to ContextPilot.
    """
    tree = load_tree(tree_path)
    all_nodes = flatten_tree(tree["structure"])
    node_map = {n["node_id"]: n for n in all_nodes if "node_id" in n}

    print("=" * 70)
    print("  PageIndex + ContextPilot Demo")
    print(f"  Document: {tree['doc_name']}")
    print(f"  Nodes: {len(all_nodes)}")
    print("=" * 70)

    # ── Simulated queries (what an LLM tree-search would return) ──
    # Node IDs map to the bundled Disney tree (examples/data/disney_q1_fy25_tree.json).
    # The tree uses zero-padded string IDs like "0001"; we convert to ints.
    #
    # Intentionally: doc order within each context is NOT pre-sorted,
    # and queries with high overlap are NOT adjacent in the list.
    # This lets the demo show both of ContextPilot's optimizations:
    #   1. Intra-context reordering: move shared nodes to the front
    #   2. Inter-context scheduling: run similar queries consecutively
    queries = [
        ("Revenue & EPS growth",        [8, 31, 2, 1]),       # shared 1,2 buried at end
        ("FY2025 outlook & CapEx",      [29, 5, 6, 3]),       # no overlap — breaks naive streaks
        ("Streaming (DTC) performance", [14, 12, 1, 10, 2]),  # shared 1,2,10 scattered
        ("Theme parks performance",     [20, 10, 2, 1]),      # shared 1,2,10
        ("Content licensing results",   [15, 12, 1, 2]),      # shared 1,2
        ("ESPN & Sports results",       [17, 16, 2, 10, 1]),  # shared 1,2,10 scattered
    ]

    # Validate node IDs against the loaded tree
    valid_ids = {int(n["node_id"]) for n in all_nodes if "node_id" in n}
    for label, nids in queries:
        missing = [n for n in nids if n not in valid_ids]
        if missing:
            print(f"  Warning: query '{label}' has unknown node IDs: {missing}")

    print(f"\n  Queries ({len(queries)}):")
    for label, node_ids in queries:
        titles = []
        for n in node_ids:
            nid_str = str(n).zfill(4)  # tree uses zero-padded IDs like "0001"
            if nid_str in node_map:
                titles.append(node_map[nid_str]["title"][:30])
            elif str(n) in node_map:
                titles.append(node_map[str(n)]["title"][:30])
            else:
                titles.append(f"?{n}")
        print(f"    {label:35s} -> nodes {node_ids}  ({', '.join(titles)})")

    # ── Run ContextPilot ──
    contexts = [nids for _, nids in queries]
    labels = [label for label, _ in queries]

    print(f"\n{'─' * 70}")
    print("  ContextPilot Analysis")
    print(f"{'─' * 70}")

    result = run_contextpilot(contexts, labels, use_gpu=False, alpha=0.005)

    # ── Show reordering explanation ──
    print(f"\n{'─' * 70}")
    print("  What happened:")
    print(f"{'─' * 70}")
    print("  Notice that the original queries had shared nodes (1, 2, 10)")
    print("  buried in the middle or end of each context. ContextPilot:")
    print("    1. Reordered docs WITHIN each context → shared nodes moved to front")
    print("    2. Scheduled queries so overlapping ones run consecutively")
    print("  Both optimizations together maximize radix-tree KV cache reuse.")
    print("  Compare with Naive (original order, no reordering) → 0 prefix reuse.")
    print()
    print("  Tree data:  examples/data/disney_q1_fy25_tree.json (41 nodes)")
    print("  Generate:   pip install pageindex  (see https://github.com/yinsicheng/PageIndex)")

    return result


# ============================================================================
# Full pipeline with PageIndexRetriever + ContextPilot
# ============================================================================

def run_pipeline(
    tree_path: str,
    queries: List[str],
    model: str = "gpt-4o",
    top_k: int = 5,
    use_gpu: bool = False,
) -> Dict[str, Any]:
    """
    Full pipeline: PageIndexRetriever tree-search -> ContextPilot -> answer generation.

    Requires:
        pip install openai
        export OPENAI_API_KEY="your-key"
    """
    from contextpilot.retriever import PageIndexRetriever, PAGEINDEX_AVAILABLE

    if not PAGEINDEX_AVAILABLE:
        raise ImportError(
            "PageIndexRetriever not available. Install openai: pip install openai"
        )

    # Phase 0: Load tree via PageIndexRetriever
    retriever = PageIndexRetriever(model=model, verbose=True)
    retriever.load_tree_structures([tree_path])

    all_nodes = flatten_tree(
        list(retriever.documents.values())[0].get("structure", {})
    )
    node_map = {n["node_id"]: n for n in all_nodes if "node_id" in n}

    doc_name = list(retriever.documents.keys())[0]
    print("=" * 70)
    print(f"  PageIndex + ContextPilot Pipeline")
    print(f"  Document: {doc_name}")
    print(f"  Model: {model}")
    print("=" * 70)

    # Phase 1: Tree search via PageIndexRetriever
    print(f"\n  Phase 1: Tree Search ({len(queries)} queries)")
    query_data = [{"question": q} for q in queries]
    search_results = retriever.search_queries(query_data=query_data, top_k=top_k)

    for i, sr in enumerate(search_results):
        print(f"    [{i}] {sr['text'][:50]:50s} -> {sr['top_k_doc_id']}")

    # Convert chunk_ids to integer node IDs for ContextPilot
    corpus_map = retriever.get_corpus_map()
    contexts = []
    for sr in search_results:
        node_ids = []
        for chunk_id in sr["top_k_doc_id"]:
            item = corpus_map.get(chunk_id, {})
            nid = item.get("node_id", "")
            if nid:
                node_ids.append(int(nid))
        contexts.append(node_ids)

    labels = [sr["text"][:35] for sr in search_results]

    # Phase 2: ContextPilot optimization
    print(f"\n  Phase 2: ContextPilot Optimization")
    cp_result = run_contextpilot(contexts, labels, use_gpu=use_gpu)

    # Phase 3: Generate answers using optimized order
    print(f"\n  Phase 3: Answer Generation")
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI()
    answers = []
    for i, q_idx in enumerate(cp_result["original_indices"]):
        sr = search_results[q_idx]
        query = sr["text"]
        reordered_ids = cp_result["reordered_contexts"][i]

        # Build context text from node summaries/text
        context_parts = []
        for nid in reordered_ids:
            nid_str = str(nid).zfill(4)
            node = node_map.get(nid_str) or node_map.get(str(nid))
            if node:
                text = node.get("text") or node.get("summary", "")
                context_parts.append(f"[{node['title']}]\n{text}")

        context_text = "\n\n".join(context_parts)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": (
                    f"Answer based on the context.\n\n"
                    f"Question: {query}\n\nContext:\n{context_text}"
                )},
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content
        answers.append({"query": query, "answer": answer})
        print(f"    [{q_idx}] {query[:50]}")
        print(f"        -> {answer[:120]}...")

    return {"answers": answers, "statistics": cp_result["prefix_sharing"]}


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PageIndex + ContextPilot E2E Example"
    )
    parser.add_argument(
        "--tree", "-t", type=str,
        help="Path to PageIndex tree JSON (default: bundled Disney tree)",
    )
    parser.add_argument(
        "--query", "-q", type=str, action="append",
        help="Query for LLM tree search + answer generation (repeatable)",
    )
    parser.add_argument("--model", default="gpt-4o", help="LLM model")
    parser.add_argument("--top-k", type=int, default=5, help="Nodes per query")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    args = parser.parse_args()

    if args.query:
        # Full LLM pipeline: PageIndexRetriever tree search + ContextPilot + answer generation
        run_pipeline(
            args.tree or DEFAULT_TREE_PATH, args.query,
            model=args.model, top_k=args.top_k, use_gpu=args.gpu,
        )
    else:
        # Demo mode: simulated queries, no API key needed
        demo_disney(args.tree)
