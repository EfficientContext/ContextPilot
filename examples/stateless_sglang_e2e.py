#!/usr/bin/env python3
"""
Complete End-to-End Example: ContextPilot Stateless + Inference Engine

This example shows the FULL workflow:
1. Retrieve contexts (documents) for queries
2. Use ContextPilot to reorder contexts:
   - Inter-context reordering: similar contexts scheduled together
   - Intra-context reordering: shared doc IDs moved to front as common prefix
3. Build prompts with REORDERED contexts (use reordered_contexts, not original!)
4. Send to inference engine (prefix sharing maximized via KV-cache)
5. Get responses back in original order

KEY INSIGHT:
  ContextPilot doesn't just reorder queries - it also reorders the doc IDs
  WITHIN each context so that shared documents appear first as a prefix.
  This allows the inference engine to cache and reuse the prefix computation.

SETUP:
1. Start an inference engine (SGLang or vLLM):
   python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --port 30000
   # or: python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 30000 --enable-prefix-caching

2. Start ContextPilot server (stateless mode):
   python -m contextpilot.server.http_server --port 8765 --stateless

3. Run this script:
   python examples/stateless_sglang_e2e.py
"""

import requests
import json
from typing import List, Dict, Any, Optional


# ============================================================================
# Configuration
# ============================================================================

CONTEXTPILOT_URL = "http://localhost:8765"
INFERENCE_URL = "http://localhost:30000"


# ============================================================================
# Document Store (Simulated - replace with your actual retriever)
# ============================================================================

DOCUMENT_STORE = {
    1: "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    2: "Deep learning uses neural networks with multiple layers to model complex patterns.",
    3: "Natural language processing (NLP) focuses on the interaction between computers and human language.",
    4: "Transformers are a neural network architecture that uses self-attention mechanisms.",
    5: "Large language models (LLMs) are trained on vast amounts of text data to generate human-like text.",
    6: "Retrieval-augmented generation (RAG) combines retrieval with generation for more accurate responses.",
    7: "Vector databases store embeddings for efficient similarity search.",
    8: "Prompt engineering involves crafting inputs to get desired outputs from language models.",
    9: "Fine-tuning adapts pre-trained models to specific tasks or domains.",
    10: "Inference optimization techniques like KV-cache reuse improve LLM serving efficiency.",
}


def get_documents(doc_ids: List[int]) -> List[str]:
    """Retrieve documents by their IDs."""
    return [DOCUMENT_STORE.get(doc_id, f"[Document {doc_id} not found]") for doc_id in doc_ids]


# ============================================================================
# Prompt Builder
# ============================================================================

def build_rag_prompt(question: str, context_docs: List[str]) -> str:
    """Build a RAG prompt with retrieved context documents."""
    context_text = "\n\n".join([f"[Doc {i+1}]: {doc}" for i, doc in enumerate(context_docs)])
    
    prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context_text}

Question: {question}

Answer:"""
    return prompt


# ============================================================================
# ContextPilot Scheduling
# ============================================================================

def schedule_contexts(contexts: List[List[int]], alpha: float = 0.001) -> Optional[Dict]:
    """Call ContextPilot to get optimal reordering."""
    try:
        response = requests.post(
            f"{CONTEXTPILOT_URL}/reorder",
            json={
                "contexts": contexts,
                "alpha": alpha,
                "use_gpu": False,
                "linkage_method": "average"
            },
            timeout=30.0
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"ContextPilot error: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"ContextPilot connection error: {e}")
        return None


# ============================================================================
# Inference Engine
# ============================================================================

def llm_generate(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """Send a prompt to the inference engine and get the response."""
    try:
        response = requests.post(
            f"{INFERENCE_URL}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60.0
        )
        if response.status_code == 200:
            return response.json()["choices"][0].get("text", "")
        else:
            return f"[Error: {response.status_code}]"
    except requests.exceptions.RequestException as e:
        return f"[Connection error: {e}]"


def llm_generate_batch(prompts: List[str], max_tokens: int = 256) -> List[str]:
    """Send multiple prompts to the inference engine sequentially."""
    return [llm_generate(prompt, max_tokens) for prompt in prompts]


# ============================================================================
# Complete E2E Workflow
# ============================================================================

def run_rag_with_contextpilot(
    queries: List[str],
    query_doc_ids: List[List[int]],
    use_contextpilot: bool = True
) -> List[Dict[str, Any]]:
    """
    Complete RAG workflow with ContextPilot optimization.
    
    Args:
        queries: List of user questions
        query_doc_ids: For each query, list of retrieved document IDs
        use_contextpilot: Whether to use ContextPilot for scheduling
    
    Returns:
        List of results with query, answer, and metadata
    """
    n = len(queries)
    print(f"\n{'='*60}")
    print(f"Processing {n} queries {'WITH' if use_contextpilot else 'WITHOUT'} ContextPilot")
    print(f"{'='*60}")
    
    # Step 1: Get optimal reordering from ContextPilot
    if use_contextpilot:
        print("\n📊 Step 1: Getting optimal reordering from ContextPilot...")
        schedule_result = schedule_contexts(query_doc_ids)
        
        if schedule_result:
            scheduled_order = schedule_result['original_indices']
            # IMPORTANT: Use reordered_contexts for building prompts!
            # These have BOTH:
            #   1. Contexts reordered (similar ones adjacent)
            #   2. IDs within each context reordered (shared IDs as prefix)
            reordered_contexts = schedule_result['reordered_contexts']
            num_groups = schedule_result['num_groups']
            print(f"   ✓ Optimal order: {scheduled_order}")
            print(f"   ✓ Grouped into {num_groups} execution groups")
            print(f"   ✓ Document IDs reordered within each context for prefix sharing")
        else:
            print("   ⚠ ContextPilot unavailable, using original order")
            scheduled_order = list(range(n))
            reordered_contexts = query_doc_ids  # fallback to original
    else:
        scheduled_order = list(range(n))
        reordered_contexts = query_doc_ids  # no reordering
        print("\n📊 Step 1: Using original order (no optimization)")
    
    # Step 2: Build prompts using the REORDERED contexts
    # reordered_contexts[i] has document IDs reordered for prefix sharing
    print("\n📝 Step 2: Building prompts with reordered document IDs...")
    reordered_queries = [queries[i] for i in scheduled_order]
    
    prompts = []
    for i, (query, reordered_doc_ids) in enumerate(zip(reordered_queries, reordered_contexts)):
        docs = get_documents(reordered_doc_ids)  # Use the REORDERED IDs
        prompt = build_rag_prompt(query, docs)
        prompts.append(prompt)
        orig_idx = scheduled_order[i]
        original_ids = query_doc_ids[orig_idx]
        print(f"   [{i}] Query {orig_idx}: {query[:40]}...")
        print(f"       Original doc IDs:  {original_ids}")
        print(f"       Reordered doc IDs: {list(reordered_doc_ids)}")
    
    # Step 3: Send to inference engine
    print(f"\n🚀 Step 3: Sending {len(prompts)} prompts to inference engine...")

    # Option A: Sequential (for demonstration)
    responses = []
    for i, prompt in enumerate(prompts):
        print(f"   Generating response {i+1}/{len(prompts)}...", end=" ")
        response = llm_generate(prompt)
        responses.append(response)
        print("✓")

    # Option B: Batch (uncomment to use)
    # responses = llm_generate_batch(prompts)
    
    # Step 4: Reorder responses back to original order
    print("\n🔄 Step 4: Reordering results to original query order...")
    
    # Create reverse mapping: position in scheduled_order -> original index
    reverse_mapping = {scheduled_order[i]: i for i in range(n)}
    
    results = []
    for orig_idx in range(n):
        scheduled_pos = reverse_mapping[orig_idx]
        results.append({
            'query': queries[orig_idx],
            'doc_ids': query_doc_ids[orig_idx],
            'answer': responses[scheduled_pos],
            'original_index': orig_idx,
            'scheduled_position': scheduled_pos,
        })
    
    print("   ✓ Results reordered to match original query order")
    
    return results


# ============================================================================
# Demo
# ============================================================================

def main():
    print("="*60)
    print("ContextPilot End-to-End RAG Example")
    print("="*60)
    
    # Check server availability
    print("\n🔍 Checking servers...")
    
    try:
        r = requests.get(f"{CONTEXTPILOT_URL}/health", timeout=2)
        print(f"   ContextPilot: ✓ ({r.json().get('mode', 'unknown')} mode)")
        contextpilot_available = True
    except:
        print(f"   ContextPilot: ✗ Not available at {CONTEXTPILOT_URL}")
        contextpilot_available = False
    
    try:
        r = requests.get(f"{INFERENCE_URL}/health", timeout=2)
        print(f"   Inference engine: ✓ Ready")
        engine_available = True
    except:
        print(f"   Inference engine: ✗ Not available at {INFERENCE_URL}")
        engine_available = False
    
    # Example queries and their retrieved documents
    # Notice: queries 0, 1, 3 share documents 1, 5 (prefix sharing opportunity!)
    queries = [
        "What is machine learning and how does it relate to LLMs?",
        "How do large language models work?",
        "What is natural language processing?",
        "Explain RAG and its benefits for LLMs.",
        "What are vector databases used for?",
    ]
    
    # Retrieved document IDs for each query (simulating retrieval results)
    query_doc_ids = [
        [1, 5, 10],      # Query 0: ML, LLM, inference
        [1, 5, 4],       # Query 1: ML, LLM, transformers (shares 1,5 with Q0!)
        [3, 8, 9],       # Query 2: NLP, prompt eng, fine-tuning
        [1, 5, 6],       # Query 3: ML, LLM, RAG (shares 1,5 with Q0,Q1!)
        [7, 6, 10],      # Query 4: vector DB, RAG, inference
    ]
    
    print("\n📋 Queries and their retrieved documents:")
    for i, (q, docs) in enumerate(zip(queries, query_doc_ids)):
        print(f"   [{i}] {q[:50]}...")
        print(f"       → docs: {docs}")
    
    if not engine_available:
        print("\n⚠ Inference engine not available. Showing scheduling only...")
        
        if contextpilot_available:
            result = schedule_contexts(query_doc_ids)
            if result:
                print(f"\n📊 ContextPilot Schedule:")
                print(f"   Optimal order: {result['original_indices']}")
                print(f"   Groups: {result['num_groups']}")
                
                print("\n💡 With this order, the inference engine can reuse KV-cache prefixes:")
                order = result['original_indices']
                for i, idx in enumerate(order):
                    print(f"   Position {i}: Query {idx} (docs {query_doc_ids[idx]})")
        return
    
    # Run with ContextPilot optimization
    results = run_rag_with_contextpilot(
        queries=queries,
        query_doc_ids=query_doc_ids,
        use_contextpilot=engine_available and contextpilot_available
    )
    
    # Display results
    print("\n" + "="*60)
    print("📊 RESULTS (in original query order)")
    print("="*60)
    
    for r in results:
        print(f"\n[Query {r['original_index']}] {r['query']}")
        print(f"   Docs: {r['doc_ids']}")
        print(f"   Scheduled position: {r['scheduled_position']}")
        answer = r['answer'][:200] + "..." if len(r['answer']) > 200 else r['answer']
        print(f"   Answer: {answer}")


if __name__ == "__main__":
    main()
