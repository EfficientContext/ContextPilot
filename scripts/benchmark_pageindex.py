#!/usr/bin/env python3
"""
PageIndex + ContextPilot Performance Benchmark

This script benchmarks the integration of PageIndex (reasoning-based tree search)
with ContextPilot (context optimization) on real PDF documents.

Usage:
    python benchmark_pageindex.py                    # Run default benchmark
    python benchmark_pageindex.py --pdf path/to.pdf  # Benchmark specific PDF
    python benchmark_pageindex.py --use-cached       # Use cached tree structures
    python benchmark_pageindex.py --compare          # Compare with/without ContextPilot

Requirements:
    - PageIndex: pip install pageindex
    - ContextPilot: pip install -e /path/to/ContextPilot
    - OpenAI API key: export OPENAI_API_KEY=your-key
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Install with: pip install openai")

# Try to import pageindex for building trees (optional - not needed if using pre-built trees)
PAGEINDEX_AVAILABLE = False
page_index = None
try:
    from pageindex import page_index
    from pageindex.utils import structure_to_list, remove_fields, print_tree
    PAGEINDEX_AVAILABLE = True
except ImportError:
    # pageindex not available for building trees, but we can still use pre-built trees
    pass

# Define fallback helper functions for when pageindex isn't fully available
def _fallback_structure_to_list(structure):
    """Fallback implementation of structure_to_list - flattens tree to list of nodes."""
    results = []
    
    def traverse(node):
        if isinstance(node, dict):
            results.append(node)
            # Check for child nodes (could be 'nodes' or 'children')
            for key in ['nodes', 'children']:
                if key in node and node[key]:
                    for child in node[key]:
                        traverse(child)
        elif isinstance(node, list):
            for item in node:
                traverse(item)
    
    traverse(structure)
    return results

def _fallback_remove_fields(structure, fields=None, fields_to_remove=None):
    """Fallback implementation to remove fields from tree structure."""
    import copy
    
    # Support both 'fields' and 'fields_to_remove' parameter names
    remove_list = fields if fields is not None else (fields_to_remove if fields_to_remove is not None else [])
    
    def remove_from_node(node):
        if isinstance(node, dict):
            result = {}
            for k, v in node.items():
                if k in remove_list:
                    continue
                elif k in ['nodes', 'children']:
                    result[k] = [remove_from_node(child) for child in v] if v else v
                elif isinstance(v, dict):
                    result[k] = remove_from_node(v)
                else:
                    result[k] = v
            return result
        elif isinstance(node, list):
            return [remove_from_node(item) for item in node]
        return node
    
    return remove_from_node(structure)

def _fallback_print_tree(structure, indent=0):
    """Fallback implementation to print tree structure."""
    prefix = "  " * indent
    title = structure.get("title", "untitled")
    print(f"{prefix}- {title}")
    for child in structure.get("children", []):
        _fallback_print_tree(child, indent + 1)

# Use fallback functions if pageindex isn't available
if not PAGEINDEX_AVAILABLE:
    structure_to_list = _fallback_structure_to_list
    remove_fields = _fallback_remove_fields
    print_tree = _fallback_print_tree

try:
    from contextpilot import RAGPipeline, RetrieverConfig, OptimizerConfig
    from contextpilot.retriever import PageIndexRetriever, PAGEINDEX_AVAILABLE as CP_PAGEINDEX_AVAILABLE
    from contextpilot.context_index import build_context_index
    from contextpilot.context_ordering import InterContextScheduler
    CONTEXTPILOT_AVAILABLE = True
except ImportError as e:
    CONTEXTPILOT_AVAILABLE = False
    print(f"Warning: contextpilot import error: {e}")


# Default test queries for benchmark
DEFAULT_QUERIES = [
    {"question": "What is the main topic of this document?", "qid": "q1"},
    {"question": "What are the key findings or conclusions?", "qid": "q2"},
    {"question": "What methodology or approach was used?", "qid": "q3"},
    {"question": "Who are the main authors or contributors?", "qid": "q4"},
    {"question": "What are the recommendations or next steps?", "qid": "q5"},
]

# Financial document queries
FINANCIAL_QUERIES = [
    {"question": "What was the total revenue for the period?", "qid": "f1"},
    {"question": "What are the main risk factors mentioned?", "qid": "f2"},
    {"question": "What is the company's growth strategy?", "qid": "f3"},
    {"question": "What are the key performance indicators?", "qid": "f4"},
    {"question": "What are the major expenses and costs?", "qid": "f5"},
]


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(label: str, value: str):
    """Print a formatted result line."""
    print(f"  {label:<35} {value}")


class PageIndexBenchmark:
    """Benchmark class for PageIndex + ContextPilot integration."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        openai_api_key: Optional[str] = None,
        cache_dir: str = "./benchmark_cache",
        verbose: bool = True,
        use_gpu: bool = True
    ):
        self.model = model
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.use_gpu = use_gpu
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def build_tree_structure(self, pdf_path: str, force_rebuild: bool = False) -> Dict[str, Any]:
        """Build or load PageIndex tree structure for a PDF."""
        pdf_name = Path(pdf_path).stem
        cache_path = os.path.join(self.cache_dir, f"{pdf_name}_structure.json")
        
        # Try to load from cache
        if os.path.exists(cache_path) and not force_rebuild:
            self._log(f"📂 Loading cached tree: {cache_path}")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        # Try to load from PageIndex test results
        pageindex_path = Path(__file__).parent.parent.parent / "PageIndex" / "tests" / "results"
        pageindex_tree = pageindex_path / f"{pdf_name}_structure.json"
        if pageindex_tree.exists():
            self._log(f"📂 Loading tree from PageIndex: {pageindex_tree}")
            with open(pageindex_tree, 'r') as f:
                return json.load(f)
        
        # Try building with local pageindex
        if not PAGEINDEX_AVAILABLE:
            raise ImportError(
                f"No cached tree found and PageIndex not available.\n"
                f"Either provide a pre-built tree structure or install pageindex.\n"
                f"Looked for: {cache_path} and {pageindex_tree}"
            )
        
        self._log(f"🌲 Building PageIndex tree for: {pdf_path}")
        start_time = time.time()
        
        result = page_index(
            pdf_path,
            model=self.model,
            if_add_node_id='yes',
            if_add_node_summary='yes',
            if_add_node_text='yes'
        )
        
        build_time = time.time() - start_time
        self._log(f"  ⏱️  Tree build time: {build_time:.2f}s")
        
        # Cache result
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)
        self._log(f"  💾 Cached to: {cache_path}")
        
        return result
    
    def load_tree_structure(self, tree_path: str) -> Dict[str, Any]:
        """Load a pre-built tree structure directly."""
        self._log(f"📂 Loading tree structure: {tree_path}")
        with open(tree_path, 'r') as f:
            return json.load(f)
    
    async def _call_llm(self, prompt: str, temperature: float = 0) -> str:
        """Call LLM for reasoning."""
        client = AsyncOpenAI(api_key=self.openai_api_key)
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    async def tree_search(
        self,
        query: str,
        tree_structure: Dict[str, Any],
        top_k: int = 5
    ) -> tuple[List[str], str, float]:
        """
        Perform tree search and return node IDs, reasoning, and time taken.
        """
        structure = tree_structure.get('structure', tree_structure)
        tree_for_search = remove_fields(structure, fields=['text'])
        
        search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_for_search, indent=2)}

Please reply in the following JSON format:
{{
    "thinking": "<Your thinking process on which nodes are relevant to the question>",
    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}}
Directly return the final JSON structure. Do not output anything else.
"""
        
        start_time = time.time()
        result = await self._call_llm(search_prompt)
        search_time = time.time() - start_time
        
        try:
            # Handle markdown code blocks
            result_cleaned = result.strip()
            if result_cleaned.startswith('```'):
                # Remove ```json or ``` prefix
                lines = result_cleaned.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines[-1].strip() == '```':
                    lines = lines[:-1]
                result_cleaned = '\n'.join(lines)
            
            result_json = json.loads(result_cleaned)
            node_ids = result_json.get('node_list', [])[:top_k]
            reasoning = result_json.get('thinking', '')
            return node_ids, reasoning, search_time
        except json.JSONDecodeError as e:
            return [], "Failed to parse response", search_time
    
    def get_node_texts(
        self,
        tree_structure: Dict[str, Any],
        node_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get text content for specified node IDs."""
        structure = tree_structure.get('structure', tree_structure)
        nodes = structure_to_list(structure)
        
        node_map = {node.get('node_id'): node for node in nodes if node.get('node_id')}
        
        results = []
        for node_id in node_ids:
            if node_id in node_map:
                node = node_map[node_id]
                results.append({
                    'node_id': node_id,
                    'title': node.get('title', ''),
                    'text': node.get('text', ''),
                    'summary': node.get('summary', ''),
                })
        
        return results
    
    def apply_contextpilot_optimization(
        self,
        contexts: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Apply ContextPilot optimization to retrieved contexts.
        Returns optimized contexts and time taken.
        """
        if not contexts:
            return contexts, 0.0
        
        # Convert to token lists for ContextIndex
        # We'll use text length as a proxy for token count
        context_tokens = []
        for ctx in contexts:
            text = ctx.get('text', '') or ctx.get('summary', '') or ''
            # Simple tokenization by characters (in practice, use proper tokenizer)
            tokens = list(range(len(text)))
            context_tokens.append(tokens)
        
        start_time = time.time()
        
        # Build context index
        clustering_result = build_context_index(
            contexts=context_tokens,
            use_gpu=self.use_gpu,
            alpha=0.001
        )
        
        # Schedule contexts
        scheduler = InterContextScheduler()
        scheduled = scheduler.schedule_contexts(clustering_result)
        
        optimization_time = time.time() - start_time
        
        # Reorder contexts based on scheduling
        # scheduled is a tuple: (reordered_contexts, original_contexts, indices, groups)
        if scheduled and len(scheduled) > 2:
            ordered_indices = scheduled[2]  # Third element contains the indices
            optimized_contexts = [contexts[i] for i in ordered_indices if i < len(contexts)]
        else:
            optimized_contexts = contexts
        
        return optimized_contexts, optimization_time
    
    async def generate_answer(
        self,
        query: str,
        contexts: List[Dict[str, Any]]
    ) -> tuple[str, float]:
        """Generate answer based on query and contexts."""
        context_text = "\n\n".join([
            f"[{ctx.get('title', 'Section')}]\n{ctx.get('text', '')}"
            for ctx in contexts
        ])
        
        answer_prompt = f"""
Answer the question based on the context provided.

Question: {query}

Context:
{context_text}

Provide a clear, concise answer based only on the context provided.
If the answer cannot be found in the context, say so.
"""
        
        start_time = time.time()
        answer = await self._call_llm(answer_prompt)
        gen_time = time.time() - start_time
        
        return answer, gen_time
    
    def build_shared_context_index(
        self,
        all_contexts: List[Dict[str, Any]]
    ) -> tuple[Any, float, Dict[str, int]]:
        """
        Build a shared ContextPilot index for all unique contexts.
        Returns (clustering_result, time_taken, node_id_to_index_map).
        """
        if not all_contexts:
            return None, 0.0, {}
        
        # Deduplicate contexts by node_id
        unique_contexts = {}
        for ctx in all_contexts:
            node_id = ctx.get('node_id')
            if node_id and node_id not in unique_contexts:
                unique_contexts[node_id] = ctx
        
        unique_list = list(unique_contexts.values())
        node_id_to_idx = {ctx['node_id']: i for i, ctx in enumerate(unique_list)}
        
        self._log(f"  📊 Total retrieved: {len(all_contexts)}, Unique nodes: {len(unique_list)}")
        
        # Convert to token lists for ContextIndex
        context_tokens = []
        for ctx in unique_list:
            text = ctx.get('text', '') or ctx.get('summary', '') or ''
            tokens = list(range(len(text)))
            context_tokens.append(tokens)
        
        start_time = time.time()
        
        # Build context index once
        clustering_result = build_context_index(
            contexts=context_tokens,
            use_gpu=self.use_gpu,
            alpha=0.001
        )
        
        optimization_time = time.time() - start_time
        
        return clustering_result, optimization_time, node_id_to_idx
    
    def reorder_contexts_with_index(
        self,
        contexts: List[Dict[str, Any]],
        clustering_result: Any,
        node_id_to_idx: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Reorder contexts based on pre-built clustering result."""
        if not contexts or clustering_result is None:
            return contexts
        
        # Get the indices in the shared index
        indices = [node_id_to_idx.get(ctx.get('node_id')) for ctx in contexts]
        indices = [i for i in indices if i is not None]
        
        if len(indices) < 2:
            return contexts
        
        # Use scheduler to get optimal ordering
        scheduler = InterContextScheduler()
        scheduled = scheduler.schedule_contexts(clustering_result)
        
        # scheduled is tuple: (reordered_contexts, original_contexts, indices, groups)
        if scheduled and len(scheduled) > 2:
            global_order = scheduled[2]
            # Filter to only indices in our local context set
            local_order = [i for i in global_order if i in indices]
            idx_to_ctx = {node_id_to_idx.get(ctx.get('node_id')): ctx for ctx in contexts}
            reordered = [idx_to_ctx[i] for i in local_order if i in idx_to_ctx]
            # Add any missing contexts at the end
            seen = set(local_order)
            for ctx in contexts:
                if node_id_to_idx.get(ctx.get('node_id')) not in seen:
                    reordered.append(ctx)
            return reordered if reordered else contexts
        
        return contexts
    
    async def run_query_benchmark(
        self,
        tree_structure: Dict[str, Any],
        queries: List[Dict[str, Any]],
        use_contextpilot: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Run benchmark on a set of queries with SHARED context index."""
        results = []
        total_search_time = 0
        total_optimization_time = 0
        total_gen_time = 0
        
        # Phase 1: Execute all tree searches first
        self._log("  Phase 1: Tree search for all queries...")
        search_results = []
        all_contexts = []
        
        for query in queries:
            query_text = query.get('question', query.get('text', ''))
            qid = query.get('qid', 'unknown')
            
            node_ids, reasoning, search_time = await self.tree_search(
                query_text, tree_structure, top_k
            )
            total_search_time += search_time
            
            contexts = self.get_node_texts(tree_structure, node_ids)
            all_contexts.extend(contexts)
            
            search_results.append({
                'qid': qid,
                'query_text': query_text,
                'node_ids': node_ids,
                'reasoning': reasoning,
                'contexts': contexts,
                'search_time': search_time,
            })
        
        # Phase 2: Build shared index (only once!)
        clustering_result = None
        node_id_to_idx = {}
        
        if use_contextpilot and len(all_contexts) > 1:
            self._log("  Phase 2: Building shared ContextPilot index...")
            clustering_result, total_optimization_time, node_id_to_idx = \
                self.build_shared_context_index(all_contexts)
        
        # Phase 3: Generate answers with optimized context ordering
        self._log("  Phase 3: Generating answers...")
        for sr in search_results:
            contexts = sr['contexts']
            
            # Reorder using shared index
            if use_contextpilot and clustering_result is not None:
                contexts = self.reorder_contexts_with_index(
                    contexts, clustering_result, node_id_to_idx
                )
            
            # Generate answer
            answer, gen_time = await self.generate_answer(sr['query_text'], contexts)
            total_gen_time += gen_time
            
            results.append({
                'qid': sr['qid'],
                'question': sr['query_text'],
                'node_ids': sr['node_ids'],
                'reasoning': sr['reasoning'],
                'answer': answer,
                'search_time': sr['search_time'],
                'generation_time': gen_time,
                'num_contexts': len(contexts),
            })
            
            self._log(f"    ✓ {sr['qid']}: {sr['search_time']:.2f}s search, {gen_time:.2f}s gen")
        
        return {
            'results': results,
            'num_queries': len(queries),
            'total_search_time': total_search_time,
            'total_optimization_time': total_optimization_time,
            'total_generation_time': total_gen_time,
            'avg_search_time': total_search_time / len(queries) if queries else 0,
            'avg_optimization_time': total_optimization_time / len(queries) if queries else 0,
            'avg_generation_time': total_gen_time / len(queries) if queries else 0,
            'unique_contexts': len(node_id_to_idx) if node_id_to_idx else 0,
            'total_contexts': len(all_contexts),
        }
    
    async def compare_with_without_contextpilot(
        self,
        tree_structure: Dict[str, Any],
        queries: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Compare performance with and without ContextPilot optimization."""
        print_header("Running without ContextPilot")
        baseline = await self.run_query_benchmark(
            tree_structure, queries, use_contextpilot=False, top_k=top_k
        )
        
        print_header("Running with ContextPilot")
        optimized = await self.run_query_benchmark(
            tree_structure, queries, use_contextpilot=True, top_k=top_k
        )
        
        # Calculate overlap stats
        overlap_ratio = 0
        if optimized.get('total_contexts') and optimized.get('unique_contexts'):
            overlap_ratio = 1 - (optimized['unique_contexts'] / optimized['total_contexts'])
        
        return {
            'baseline': baseline,
            'optimized': optimized,
            'optimization_overhead': optimized['total_optimization_time'],
            'context_overlap_ratio': overlap_ratio,
        }


def run_benchmark(args):
    """Main benchmark runner."""
    if not OPENAI_AVAILABLE:
        print("Error: OpenAI package is required. Install with: pip install openai")
        return
    
    # Check if we have a pre-built tree or need to build one
    using_prebuilt_tree = args.tree is not None
    
    if not using_prebuilt_tree and not PAGEINDEX_AVAILABLE:
        print("Error: PageIndex package is required for building trees from PDF.")
        print("Either install pageindex with: pip install pageindex")
        print("Or use --tree to specify a pre-built tree structure JSON file")
        return
    
    print_header("PageIndex + ContextPilot Benchmark")
    
    # Initialize benchmark
    benchmark = PageIndexBenchmark(
        model=args.model,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
        use_gpu=not args.no_gpu
    )
    
    # Determine tree source
    if using_prebuilt_tree:
        # Load from pre-built tree file
        tree_path = args.tree
        if not os.path.exists(tree_path):
            print(f"Error: Tree file not found: {tree_path}")
            return
        
        print_result("Using pre-built tree", tree_path)
        tree_structure = benchmark.load_tree_structure(tree_path)
        pdf_path = tree_structure.get('metadata', {}).get('source_pdf', tree_path)
    else:
        # Determine PDF path
        if args.pdf:
            pdf_path = args.pdf
        else:
            # Use default test PDF from PageIndex
            pageindex_path = Path(__file__).parent.parent.parent / "PageIndex"
            pdf_path = str(pageindex_path / "tests" / "pdfs" / "q1-fy25-earnings.pdf")
            
            if not os.path.exists(pdf_path):
                print(f"Default PDF not found: {pdf_path}")
                print("Please specify a PDF with --pdf argument or tree with --tree argument")
                return
        
        print_result("PDF Path", pdf_path)
        
        # Build or load tree structure
        tree_structure = benchmark.build_tree_structure(pdf_path, force_rebuild=args.rebuild)
    
    print_result("Model", args.model)
    print_result("Top-K", str(args.top_k))
    
    # Count nodes
    structure = tree_structure.get('structure', tree_structure)
    nodes = structure_to_list(structure)
    print_result("Total Nodes", str(len(nodes)))
    
    # Determine queries based on document type
    source_path = str(pdf_path).lower() if pdf_path else ""
    # Handle both dict and list tree structures
    if isinstance(structure, dict):
        tree_title = structure.get('title', '').lower()
    elif isinstance(structure, list) and len(structure) > 0 and isinstance(structure[0], dict):
        tree_title = structure[0].get('title', '').lower()
    else:
        tree_title = ""
    source_hint = source_path + " " + tree_title
    
    if "financial" in source_hint or "earnings" in source_hint or "annual" in source_hint:
        queries = FINANCIAL_QUERIES
    else:
        queries = DEFAULT_QUERIES
    
    if args.query:
        queries = [{"question": args.query, "qid": "custom"}]
    
    print_result("Num Queries", str(len(queries)))
    
    # Run benchmark
    if args.compare:
        results = asyncio.run(benchmark.compare_with_without_contextpilot(
            tree_structure, queries, top_k=args.top_k
        ))
        
        print_header("Comparison Results")
        
        # Context overlap stats
        if results['optimized'].get('total_contexts'):
            print_result("Total Retrieved Contexts", 
                         str(results['optimized']['total_contexts']))
            print_result("Unique Contexts", 
                         str(results['optimized']['unique_contexts']))
            print_result("Context Overlap Ratio", 
                         f"{results['context_overlap_ratio']*100:.1f}%")
        
        print_result("Baseline Total Time", 
                     f"{results['baseline']['total_search_time'] + results['baseline']['total_generation_time']:.2f}s")
        print_result("Optimized Total Time",
                     f"{results['optimized']['total_search_time'] + results['optimized']['total_generation_time'] + results['optimized']['total_optimization_time']:.2f}s")
        print_result("Index Build Time (once)",
                     f"{results['optimization_overhead']:.2f}s")
        
    else:
        print_header("Running Queries")
        results = asyncio.run(benchmark.run_query_benchmark(
            tree_structure, queries, use_contextpilot=not args.no_contextpilot, top_k=args.top_k
        ))
        
        print_header("Results Summary")
        print_result("Avg Search Time", f"{results['avg_search_time']:.2f}s")
        print_result("Avg Optimization Time", f"{results['avg_optimization_time']:.2f}s")
        print_result("Avg Generation Time", f"{results['avg_generation_time']:.2f}s")
        print_result("Total Time", 
                     f"{results['total_search_time'] + results['total_optimization_time'] + results['total_generation_time']:.2f}s")
    
    # Save results if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print_result("Results saved to", args.output)
    
    print("\n✅ Benchmark complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PageIndex + ContextPilot integration"
    )
    parser.add_argument(
        "--pdf", "-p",
        type=str,
        help="Path to PDF document to benchmark"
    )
    parser.add_argument(
        "--tree", "-t",
        type=str,
        help="Path to pre-built tree structure JSON file (use instead of --pdf)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of nodes to retrieve per query (default: 5)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Custom query to test"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with and without ContextPilot optimization"
    )
    parser.add_argument(
        "--no-contextpilot",
        action="store_true",
        help="Run without ContextPilot optimization"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Use CPU instead of GPU for ContextPilot optimization"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild tree structure (ignore cache)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./benchmark_cache",
        help="Directory to cache tree structures"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save benchmark results as JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        args.verbose = False
    
    run_benchmark(args)


if __name__ == "__main__":
    main()
