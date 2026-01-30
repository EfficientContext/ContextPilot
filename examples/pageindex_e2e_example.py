#!/usr/bin/env python3
"""
PageIndex + ContextPilot End-to-End Example

This example demonstrates the complete workflow of using PageIndex for
document retrieval with ContextPilot optimization for efficient LLM inference.

The workflow:
1. Load a PageIndex tree structure (pre-built from PDF)
2. Execute tree search for multiple queries
3. Build a SHARED ContextPilot index for all retrieved contexts
4. Reorder contexts for optimal LLM inference
5. Generate answers using the optimized context order

Key benefit: ContextPilot identifies overlapping contexts across queries
and builds an optimized shared index, reducing redundant computation
and improving LLM cache utilization.
"""

import json
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

# ContextPilot imports
from contextpilot.context_index import build_context_index
from contextpilot.context_ordering import InterContextScheduler

# Optional: OpenAI for actual LLM calls
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================================
# Helper Functions
# ============================================================================

def flatten_tree(structure) -> List[Dict[str, Any]]:
    """Flatten a PageIndex tree structure to a list of nodes."""
    results = []
    
    def traverse(node):
        if isinstance(node, dict):
            results.append(node)
            for child in node.get('nodes', []):
                traverse(child)
        elif isinstance(node, list):
            for item in node:
                traverse(item)
    
    traverse(structure)
    return results


def get_node_by_id(tree_structure: Dict[str, Any], node_ids: List[str]) -> List[Dict[str, Any]]:
    """Get node contents by their IDs."""
    structure = tree_structure.get('structure', tree_structure)
    nodes = flatten_tree(structure)
    node_map = {n.get('node_id'): n for n in nodes if n.get('node_id')}
    
    return [
        {
            'node_id': nid,
            'title': node_map[nid].get('title', ''),
            'text': node_map[nid].get('text', ''),
            'summary': node_map[nid].get('summary', ''),
        }
        for nid in node_ids if nid in node_map
    ]


def remove_text_field(structure):
    """Remove 'text' field from tree for search (keep only structure and summaries)."""
    def clean(node):
        if isinstance(node, dict):
            return {
                k: ([clean(c) for c in v] if k in ['nodes', 'children'] else v)
                for k, v in node.items() if k != 'text'
            }
        elif isinstance(node, list):
            return [clean(item) for item in node]
        return node
    return clean(structure)


# ============================================================================
# PageIndex Tree Search (LLM-based)
# ============================================================================

async def tree_search(
    query: str,
    tree_structure: Dict[str, Any],
    client: "AsyncOpenAI",
    model: str = "gpt-4o",
    top_k: int = 5
) -> List[str]:
    """
    Use LLM to search the tree structure and find relevant nodes.
    
    This is the core of PageIndex's reasoning-based retrieval:
    instead of vector similarity, we use LLM reasoning to navigate
    the document hierarchy.
    """
    structure = tree_structure.get('structure', tree_structure)
    tree_for_search = remove_text_field(structure)
    
    search_prompt = f"""You are given a question and a tree structure of a document.
Each node contains a node_id, title, and summary.
Find the nodes most likely to contain the answer.

Question: {query}

Document tree:
{json.dumps(tree_for_search, indent=2)}

Reply in JSON format:
{{
    "thinking": "<your reasoning>",
    "node_list": ["node_id_1", "node_id_2", ...]
}}
Return at most {top_k} nodes. Only output the JSON."""

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": search_prompt}],
        temperature=0.0,
    )
    
    result = response.choices[0].message.content.strip()
    
    # Parse response (handle markdown code blocks)
    if result.startswith('```'):
        lines = result.split('\n')[1:-1]
        result = '\n'.join(lines)
    
    try:
        parsed = json.loads(result)
        return parsed.get('node_list', [])[:top_k]
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse search result: {result[:100]}")
        return []


# ============================================================================
# ContextPilot Optimization
# ============================================================================

def build_shared_context_index(
    contexts: List[Dict[str, Any]],
    use_gpu: bool = False
) -> tuple[Any, Dict[str, int], float]:
    """
    Build a shared ContextPilot index for all unique contexts.
    
    Returns:
        - clustering_result: The clustering result for scheduling
        - node_id_to_idx: Mapping from node_id to index
        - overlap_ratio: Ratio of duplicated contexts
    """
    # Deduplicate by node_id
    unique_contexts = {}
    for ctx in contexts:
        node_id = ctx.get('node_id')
        if node_id and node_id not in unique_contexts:
            unique_contexts[node_id] = ctx
    
    unique_list = list(unique_contexts.values())
    node_id_to_idx = {ctx['node_id']: i for i, ctx in enumerate(unique_list)}
    
    # Calculate overlap
    overlap_ratio = 1 - (len(unique_list) / len(contexts)) if contexts else 0
    
    print(f"📊 Context Analysis:")
    print(f"   Total retrieved: {len(contexts)}")
    print(f"   Unique contexts: {len(unique_list)}")
    print(f"   Overlap ratio: {overlap_ratio*100:.1f}%")
    
    # Convert to token lists for ContextPilot
    # In practice, use a proper tokenizer; here we use character count as proxy
    context_tokens = []
    for ctx in unique_list:
        text = ctx.get('text', '') or ctx.get('summary', '')
        tokens = list(range(len(text)))
        context_tokens.append(tokens)
    
    # Build index (only once for all queries!)
    print(f"\n🔧 Building ContextPilot index...")
    clustering_result = build_context_index(
        contexts=context_tokens,
        use_gpu=use_gpu,
        alpha=0.001
    )
    
    return clustering_result, node_id_to_idx, overlap_ratio


def reorder_contexts(
    contexts: List[Dict[str, Any]],
    clustering_result: Any,
    node_id_to_idx: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Reorder contexts for a single query using the shared index.
    
    This optimizes context order for better LLM cache utilization
    by grouping similar/overlapping contexts together.
    """
    if not contexts or clustering_result is None:
        return contexts
    
    scheduler = InterContextScheduler()
    scheduled = scheduler.schedule_contexts(clustering_result)
    
    if scheduled and len(scheduled) > 2:
        # Get global ordering from scheduler
        global_order = scheduled[2]
        
        # Map to local context indices
        local_indices = set(node_id_to_idx.get(ctx.get('node_id')) for ctx in contexts)
        local_indices.discard(None)
        
        # Reorder based on global scheduling
        idx_to_ctx = {node_id_to_idx.get(ctx.get('node_id')): ctx for ctx in contexts}
        reordered = [idx_to_ctx[i] for i in global_order if i in local_indices and i in idx_to_ctx]
        
        # Add any missing (shouldn't happen)
        seen = set(node_id_to_idx.get(ctx.get('node_id')) for ctx in reordered)
        for ctx in contexts:
            if node_id_to_idx.get(ctx.get('node_id')) not in seen:
                reordered.append(ctx)
        
        return reordered if reordered else contexts
    
    return contexts


# ============================================================================
# Answer Generation
# ============================================================================

async def generate_answer(
    query: str,
    contexts: List[Dict[str, Any]],
    client: "AsyncOpenAI",
    model: str = "gpt-4o"
) -> str:
    """Generate an answer using the retrieved and optimized contexts."""
    context_text = "\n\n".join([
        f"[{ctx.get('title', 'Section')}]\n{ctx.get('text', ctx.get('summary', ''))}"
        for ctx in contexts
    ])
    
    prompt = f"""Answer the question based on the context provided.

Question: {query}

Context:
{context_text}

Provide a clear, concise answer based only on the context."""

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    
    return response.choices[0].message.content


# ============================================================================
# Main End-to-End Pipeline
# ============================================================================

async def run_pageindex_with_contextpilot(
    tree_structure: Dict[str, Any],
    queries: List[str],
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    top_k: int = 5,
    use_gpu: bool = False
) -> Dict[str, Any]:
    """
    Run the complete PageIndex + ContextPilot pipeline.
    
    This demonstrates the optimal workflow:
    1. Execute all tree searches first
    2. Build ONE shared ContextPilot index
    3. Reorder contexts for each query using the shared index
    4. Generate answers
    
    Args:
        tree_structure: PageIndex tree structure (from JSON)
        queries: List of questions to answer
        api_key: OpenAI API key (uses env var if not provided)
        model: LLM model to use
        top_k: Number of nodes to retrieve per query
        use_gpu: Whether to use GPU for ContextPilot
    
    Returns:
        Dictionary with results and statistics
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package required. Install with: pip install openai")
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
    
    client = AsyncOpenAI(api_key=api_key)
    
    print("=" * 60)
    print("  PageIndex + ContextPilot End-to-End Pipeline")
    print("=" * 60)
    
    # Phase 1: Tree Search for all queries
    print(f"\n📍 Phase 1: Tree Search ({len(queries)} queries)")
    print("-" * 40)
    
    search_results = []
    all_contexts = []
    
    for i, query in enumerate(queries):
        print(f"  🔍 Query {i+1}: {query[:50]}...")
        
        node_ids = await tree_search(query, tree_structure, client, model, top_k)
        contexts = get_node_by_id(tree_structure, node_ids)
        
        print(f"     → Retrieved {len(node_ids)} nodes: {node_ids}")
        
        search_results.append({
            'query': query,
            'node_ids': node_ids,
            'contexts': contexts
        })
        all_contexts.extend(contexts)
    
    # Phase 2: Build SHARED ContextPilot Index
    print(f"\n📍 Phase 2: Build Shared ContextPilot Index")
    print("-" * 40)
    
    clustering_result, node_id_to_idx, overlap_ratio = build_shared_context_index(
        all_contexts, use_gpu=use_gpu
    )
    
    # Phase 3: Generate Answers with Optimized Context Order
    print(f"\n📍 Phase 3: Generate Answers")
    print("-" * 40)
    
    answers = []
    for i, sr in enumerate(search_results):
        query = sr['query']
        contexts = sr['contexts']
        
        # Reorder using shared index
        optimized_contexts = reorder_contexts(contexts, clustering_result, node_id_to_idx)
        
        print(f"  💬 Query {i+1}: Generating answer...")
        answer = await generate_answer(query, optimized_contexts, client, model)
        
        answers.append({
            'query': query,
            'node_ids': sr['node_ids'],
            'answer': answer,
            'num_contexts': len(contexts)
        })
        print(f"     ✓ Done")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("  Summary")
    print("=" * 60)
    print(f"  Queries processed: {len(queries)}")
    print(f"  Total contexts retrieved: {len(all_contexts)}")
    print(f"  Unique contexts: {len(node_id_to_idx)}")
    print(f"  Context overlap: {overlap_ratio*100:.1f}%")
    print(f"  Index built: 1 time (shared)")
    
    return {
        'answers': answers,
        'statistics': {
            'num_queries': len(queries),
            'total_contexts': len(all_contexts),
            'unique_contexts': len(node_id_to_idx),
            'overlap_ratio': overlap_ratio
        }
    }


# ============================================================================
# Demo with Sample Data
# ============================================================================

def create_sample_tree() -> Dict[str, Any]:
    """Create a sample tree structure for demo purposes."""
    return {
        "doc_name": "sample_report.pdf",
        "doc_description": "A sample quarterly report for demonstration.",
        "structure": [
            {
                "title": "Executive Summary",
                "node_id": "0001",
                "summary": "Overview of Q4 performance showing 15% revenue growth.",
                "text": "Q4 2024 was a strong quarter with 15% year-over-year revenue growth. Key highlights include successful product launches, expansion into new markets, and improved operational efficiency.",
                "nodes": []
            },
            {
                "title": "Financial Results",
                "node_id": "0002",
                "summary": "Detailed financial metrics and performance indicators.",
                "text": "Revenue reached $500M, up from $435M last year. Operating income increased 20% to $75M. Net income was $60M with EPS of $2.50.",
                "nodes": [
                    {
                        "title": "Revenue Breakdown",
                        "node_id": "0003",
                        "summary": "Revenue by segment and region.",
                        "text": "Product revenue: $350M (70%). Services revenue: $150M (30%). North America: 60%, Europe: 25%, Asia: 15%.",
                        "nodes": []
                    },
                    {
                        "title": "Cost Analysis",
                        "node_id": "0004",
                        "summary": "Cost structure and margins.",
                        "text": "Gross margin improved to 45% from 42%. Operating expenses were $425M, with R&D at $100M and S&M at $200M.",
                        "nodes": []
                    }
                ]
            },
            {
                "title": "Business Highlights",
                "node_id": "0005",
                "summary": "Key achievements and milestones.",
                "text": "Launched 3 new products. Acquired TechStartup Inc. Expanded to 5 new countries. Customer base grew 25% to 10,000 enterprise clients.",
                "nodes": []
            },
            {
                "title": "Outlook",
                "node_id": "0006",
                "summary": "Guidance for next quarter and full year.",
                "text": "Q1 2025 revenue expected at $520-540M. Full year 2025 guidance: 12-15% growth. Investing heavily in AI capabilities.",
                "nodes": []
            }
        ]
    }


async def demo_without_api():
    """Demo the workflow without making actual API calls."""
    print("=" * 60)
    print("  PageIndex + ContextPilot Demo (No API)")
    print("=" * 60)
    
    tree = create_sample_tree()
    
    # Simulate search results (what LLM would return)
    simulated_searches = [
        {'query': 'What was the revenue growth?', 'nodes': ['0001', '0002', '0003']},
        {'query': 'What are the cost margins?', 'nodes': ['0002', '0004']},
        {'query': 'What is the outlook?', 'nodes': ['0001', '0006']},
        {'query': 'What were the business achievements?', 'nodes': ['0005', '0001']},
    ]
    
    print(f"\n📊 Simulating {len(simulated_searches)} queries with overlap...")
    
    # Collect all contexts
    all_contexts = []
    for search in simulated_searches:
        contexts = get_node_by_id(tree, search['nodes'])
        all_contexts.extend(contexts)
        print(f"  Query: {search['query'][:40]}...")
        print(f"    → Nodes: {search['nodes']}")
    
    # Build shared index
    print("\n🔧 Building ContextPilot index...")
    clustering_result, node_id_to_idx, overlap_ratio = build_shared_context_index(
        all_contexts, use_gpu=False
    )
    
    print(f"\n✅ Demo complete!")
    print(f"   Overlap ratio: {overlap_ratio*100:.1f}%")
    print(f"   (In production, this means {overlap_ratio*100:.1f}% less redundant computation)")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PageIndex + ContextPilot E2E Example")
    parser.add_argument("--tree", "-t", type=str, help="Path to PageIndex tree JSON")
    parser.add_argument("--demo", action="store_true", help="Run demo without API")
    parser.add_argument("--query", "-q", type=str, action="append", help="Query to run")
    
    args = parser.parse_args()
    
    if args.demo:
        asyncio.run(demo_without_api())
    elif args.tree and args.query:
        with open(args.tree) as f:
            tree = json.load(f)
        
        results = asyncio.run(run_pageindex_with_contextpilot(
            tree_structure=tree,
            queries=args.query,
            use_gpu=False
        ))
        
        print("\n📝 Answers:")
        for ans in results['answers']:
            print(f"\nQ: {ans['query']}")
            print(f"A: {ans['answer'][:200]}...")
    else:
        # Run demo by default
        asyncio.run(demo_without_api())
