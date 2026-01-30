"""
PageIndex + ContextPilot Example

This example demonstrates how to use PageIndex's reasoning-based tree search
retrieval with ContextPilot's context optimization for improved RAG performance.

PageIndex Features:
- No Vector DB needed: Uses document structure and LLM reasoning for retrieval
- No Chunking needed: Documents are organized into natural sections
- Human-like Retrieval: Simulates how human experts navigate documents

ContextPilot Features:  
- Optimal context ordering for LLM inference
- Prefix sharing optimization for batch processing
- Multi-turn conversation support

Requirements:
    pip install pageindex openai
    pip install -e /path/to/ContextPilot
    export OPENAI_API_KEY=your-api-key
"""

import os
import json
import asyncio
from pathlib import Path

# Import PageIndex
try:
    from pageindex import page_index
    from pageindex.utils import structure_to_list, remove_fields, print_tree
    PAGEINDEX_AVAILABLE = True
except ImportError:
    PAGEINDEX_AVAILABLE = False
    print("PageIndex not installed. Install with: pip install pageindex")

# Import ContextPilot
try:
    from contextpilot import RAGPipeline, RetrieverConfig, OptimizerConfig
    from contextpilot.retriever import PageIndexRetriever
    from contextpilot.context_index import build_context_index
    from contextpilot.context_ordering import InterContextScheduler
    CONTEXTPILOT_AVAILABLE = True
except ImportError:
    CONTEXTPILOT_AVAILABLE = False
    print("ContextPilot not installed. Install from the repository.")

# OpenAI for LLM calls
try:
    import openai
    from openai import AsyncOpenAI
except ImportError:
    print("OpenAI not installed. Install with: pip install openai")


# ============================================================================
# Example 1: Basic PageIndex + ContextPilot Usage
# ============================================================================

async def example_basic_usage():
    """
    Basic example showing PageIndex tree search with ContextPilot optimization.
    """
    print("\n" + "="*70)
    print("Example 1: Basic PageIndex + ContextPilot Usage")
    print("="*70)
    
    # Setup
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize PageIndex retriever
    retriever = PageIndexRetriever(
        model="gpt-4o",
        openai_api_key=api_key,
        tree_cache_dir="./pageindex_cache",
        verbose=True
    )
    
    # Option 1: Build tree from PDF (takes time)
    # retriever.index_documents(["path/to/your/document.pdf"])
    
    # Option 2: Load pre-built tree structure (faster)
    # Find a test PDF in PageIndex
    pageindex_path = Path(__file__).parent.parent.parent / "PageIndex"
    tree_path = pageindex_path / "tests" / "results" / "q1-fy25-earnings_structure.json"
    
    if tree_path.exists():
        retriever.load_tree_structures([str(tree_path)])
        
        # Search with reasoning-based tree search
        results = retriever.search_queries(
            query_data=[{"question": "What was the total revenue?"}],
            top_k=3
        )
        
        print("\n📊 Search Results:")
        for r in results:
            print(f"  Query: {r['text']}")
            print(f"  Retrieved node IDs: {r['top_k_doc_id']}")
        
        # Get corpus for ContextPilot optimization
        corpus = retriever.get_corpus()
        print(f"\n📚 Corpus size: {len(corpus)} nodes")
        
    else:
        print(f"Tree structure not found: {tree_path}")
        print("Run: python -m pageindex path/to/pdf to generate tree")


# ============================================================================
# Example 2: Using RAGPipeline with PageIndex
# ============================================================================

async def example_rag_pipeline():
    """
    Example using the unified RAGPipeline API with PageIndex retriever.
    """
    print("\n" + "="*70)
    print("Example 2: Using RAGPipeline with PageIndex Retriever")
    print("="*70)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Find test data
    pageindex_path = Path(__file__).parent.parent.parent / "PageIndex"
    tree_path = pageindex_path / "tests" / "results" / "q1-fy25-earnings_structure.json"
    
    if not tree_path.exists():
        print(f"Tree structure not found: {tree_path}")
        return
    
    # Create pipeline with PageIndex retriever
    pipeline = RAGPipeline(
        retriever=RetrieverConfig(
            retriever_type="pageindex",
            pageindex_model="gpt-4o",
            pageindex_openai_api_key=api_key,
            pageindex_tree_paths=[str(tree_path)],
            top_k=5
        ),
        optimizer=OptimizerConfig(
            enabled=True,
            use_gpu=False,  # Set True if GPU available
            alpha=0.001
        ),
        use_contextpilot=True
    )
    
    # Setup pipeline
    pipeline.setup()
    
    # Run queries
    queries = [
        {"question": "What is the total revenue?"},
        {"question": "What are the main expenses?"},
    ]
    
    results = pipeline.run(
        queries=queries,
        generate_responses=False  # Just retrieve, don't generate
    )
    
    print("\n📊 Pipeline Results:")
    for i, r in enumerate(results):
        print(f"  Query {i+1}: {len(r.get('retrieved_docs', []))} documents retrieved")


# ============================================================================
# Example 3: Manual Tree Search with ContextPilot Optimization
# ============================================================================

async def example_manual_optimization():
    """
    Example showing manual tree search and applying ContextPilot optimization.
    """
    print("\n" + "="*70)
    print("Example 3: Manual Tree Search with ContextPilot Optimization")
    print("="*70)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Load a pre-built tree structure
    pageindex_path = Path(__file__).parent.parent.parent / "PageIndex"
    tree_path = pageindex_path / "tests" / "results" / "q1-fy25-earnings_structure.json"
    
    if not tree_path.exists():
        print(f"Tree structure not found: {tree_path}")
        return
    
    with open(tree_path) as f:
        tree_data = json.load(f)
    
    structure = tree_data.get('structure', tree_data)
    nodes = structure_to_list(structure)
    print(f"📚 Loaded tree with {len(nodes)} nodes")
    
    # Create LLM client
    client = AsyncOpenAI(api_key=api_key)
    
    async def tree_search(query: str, top_k: int = 5):
        """Perform tree search using LLM reasoning."""
        tree_for_search = remove_fields(structure, fields=['text'])
        
        search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_for_search, indent=2)}

Reply in JSON format:
{{
    "thinking": "<Your reasoning>",
    "node_list": ["node_id_1", "node_id_2", ...]
}}
"""
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": search_prompt}],
            temperature=0
        )
        
        result = json.loads(response.choices[0].message.content)
        return result['node_list'][:top_k], result.get('thinking', '')
    
    # Perform tree search
    query = "What were the key financial highlights?"
    node_ids, reasoning = await tree_search(query)
    
    print(f"\n🔍 Query: {query}")
    print(f"🧠 Reasoning: {reasoning[:200]}...")
    print(f"📋 Retrieved nodes: {node_ids}")
    
    # Get node content
    node_map = {n['node_id']: n for n in nodes if 'node_id' in n}
    retrieved_nodes = [node_map[nid] for nid in node_ids if nid in node_map]
    
    # Apply ContextPilot optimization
    if len(retrieved_nodes) > 1:
        # Convert to token lists (using summary or text length as proxy)
        context_tokens = [
            list(range(len(n.get('text', '') or n.get('summary', '') or ''))) 
            for n in retrieved_nodes
        ]
        
        # Build context index
        clustering_result = build_context_index(
            contexts=context_tokens,
            use_gpu=False,
            alpha=0.001
        )
        
        # Schedule contexts
        scheduler = InterContextScheduler()
        scheduled = scheduler.schedule_contexts(clustering_result)
        
        # scheduled is a tuple: (reordered_contexts, original_contexts, indices, groups)
        if scheduled and len(scheduled) > 2:
            print(f"\n✨ ContextPilot optimization:")
            print(f"  Original order: {list(range(len(retrieved_nodes)))}")
            print(f"  Optimized order: {list(scheduled[2])}")
    
    # Generate answer
    context_text = "\n\n".join([
        f"[{n.get('title', 'Section')}]\n{n.get('text', '')[:500]}..."
        for n in retrieved_nodes
    ])
    
    answer_prompt = f"""
Answer based on the context:

Question: {query}

Context:
{context_text}

Provide a clear, concise answer.
"""
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0
    )
    
    answer = response.choices[0].message.content
    print(f"\n💡 Answer: {answer}")


# ============================================================================
# Example 4: Multi-Document Search with PageIndex
# ============================================================================

async def example_multi_document():
    """
    Example showing PageIndex search across multiple documents.
    """
    print("\n" + "="*70)
    print("Example 4: Multi-Document Search with PageIndex")
    print("="*70)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Find test tree structures
    pageindex_path = Path(__file__).parent.parent.parent / "PageIndex"
    results_dir = pageindex_path / "tests" / "results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Load multiple tree structures
    tree_paths = list(results_dir.glob("*_structure.json"))[:3]  # Use first 3
    
    if not tree_paths:
        print("No tree structures found")
        return
    
    retriever = PageIndexRetriever(
        model="gpt-4o",
        openai_api_key=api_key,
        verbose=True
    )
    
    retriever.load_tree_structures([str(p) for p in tree_paths])
    
    print(f"\n📚 Loaded {len(retriever.documents)} documents:")
    for doc_name in retriever.documents.keys():
        print(f"  - {doc_name}")
    
    # Search across all documents
    queries = [
        {"question": "What are the main conclusions?"},
        {"question": "What methodology was used?"},
    ]
    
    results = retriever.search_queries(query_data=queries, top_k=3)
    
    print("\n🔍 Search Results:")
    for r in results:
        print(f"\n  Query: {r['text']}")
        corpus_map = retriever.get_corpus_map()
        for chunk_id in r['top_k_doc_id']:
            if chunk_id in corpus_map:
                item = corpus_map[chunk_id]
                print(f"    - [{item['doc_name']}] {item['title'][:50]}...")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all examples."""
    if not PAGEINDEX_AVAILABLE:
        print("\n⚠️  PageIndex is required. Install with: pip install pageindex")
        return
    
    if not CONTEXTPILOT_AVAILABLE:
        print("\n⚠️  ContextPilot is required. Install from the repository.")
        return
    
    print("\n" + "="*70)
    print("PageIndex + ContextPilot Integration Examples")
    print("="*70)
    print("\nThese examples demonstrate how to combine PageIndex's reasoning-based")
    print("tree search with ContextPilot's context optimization for better RAG.")
    
    # Run examples
    try:
        await example_basic_usage()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    try:
        await example_rag_pipeline()
    except Exception as e:
        print(f"Example 2 error: {e}")
    
    try:
        await example_manual_optimization()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    try:
        await example_multi_document()
    except Exception as e:
        print(f"Example 4 error: {e}")
    
    print("\n" + "="*70)
    print("Examples Complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
