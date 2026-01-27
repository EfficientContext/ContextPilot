"""
ContextPilot Pipeline - Usage Examples

This file demonstrates the most important use cases for the ContextPilot pipeline.
"""

from contextpilot.pipeline import RAGPipeline, InferenceConfig

# ============================================================================
# Example 1: Simple ContextPilot Pipeline with BM25 (Retrieve + Optimize)
# ============================================================================

def example_simple_contextpilot():
    """Basic ContextPilot usage with BM25: retrieval and optimization only."""
    print("\n" + "="*60)
    print("Example 1: Simple ContextPilot Pipeline (BM25)")
    print("="*60)
    
    # Create pipeline with minimal configuration
    pipeline = RAGPipeline(
        retriever="bm25",
        corpus_path="corpus.jsonl",
        es_index_name="my_bm25_index",
    )
    
    # Run on queries (retrieve + optimize)
    results = pipeline.run(queries=[
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is deep learning?"
    ])
    
    # Save optimized batch for later use
    pipeline.save_results(results, "output_contextpilot.jsonl")
    
    print(f"\n✅ Processed {results['metadata']['num_queries']} queries")
    print(f"📊 Created {len(results['optimized_batch'])} optimized groups")


# ============================================================================
# Example 2: FAISS Retriever with ContextPilot
# ============================================================================

def example_faiss_contextpilot():
    """Use FAISS semantic search with ContextPilot optimization."""
    print("\n" + "="*60)
    print("Example 2: FAISS + ContextPilot")
    print("="*60)
    
    pipeline = RAGPipeline(
        retriever="faiss",
        corpus_path="corpus.jsonl",
        index_path="corpus_index.faiss",
        embedding_model="Alibaba-NLP/gte-Qwen2-7B-instruct",
        embedding_base_url="http://localhost:30000",
        use_contextpilot=True
    )
    
    # Run retrieval + optimization
    results = pipeline.run(
        queries=["Explain quantum computing", "What is neural networks?"],
        top_k=10
    )
    
    print(f"\n✅ Processed {results['metadata']['num_queries']} queries")
    print(f"📊 Created {len(results['optimized_batch'])} optimized groups")
    
    # Save optimized results
    pipeline.save_results(results, "faiss_optimized_batch.jsonl")


# ============================================================================
# Example 3: End-to-End RAG with LLM Generation
# ============================================================================

def example_with_generation():
    """Complete RAG pipeline: retrieve, optimize, and generate responses.
    
    Prompts are automatically generated with:
    - Retrieved documents (in optimized order for cache sharing)
    - Document importance ranking
    - Formatted instructions for the LLM
    """
    print("\n" + "="*60)
    print("Example 3: ContextPilot + LLM Generation")
    print("="*60)
    
    pipeline = RAGPipeline(
        retriever="bm25",
        corpus_path="corpus.jsonl",
        inference=InferenceConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            backend="sglang",
            base_url="http://localhost:30000",
            max_tokens=20
        )
    )
    
    # Run with generation enabled
    # Prompts are automatically generated with full RAG context
    # By default, all requests are sent at once for maximum throughput

    queries = [{"qid": 1, "text": "What is AI?"}, {"qid": 2, "text": "What is ML?"}]
    results = pipeline.run(
        queries=queries,
        generate_responses=True
    )
    
    # Access generated responses
    if "generation_results" in results:
        print(f"\n✅ Generated {len(results['generation_results'])} responses")
        for i, result in enumerate(results['generation_results'][:3]):
            if result['success']:
                print(f"\nQuery {i+1} response: {result['generated_text'][:100]}...")
        
        # Check generation stats
        stats = results['metadata']['generation_stats']
        print(f"\n📊 Generation Stats:")
        print(f"  - Total time: {stats['total_time']:.2f}s")
        print(f"  - Successful: {stats['successful_requests']}/{stats['total_requests']}")


# ============================================================================
# Example 4: Step-by-Step Pipeline Control
# ============================================================================

def example_stepwise():
    """Run pipeline steps separately for fine-grained control.
    
    This shows how the pipeline:
    1. Retrieves documents
    2. Optimizes context ordering for cache efficiency
    3. Generates prompts with full RAG context (documents + ranking)
    4. Calls LLM inference API
    """
    print("\n" + "="*60)
    print("Example 4: Step-by-Step Pipeline")
    print("="*60)
    
    pipeline = RAGPipeline(
        retriever="bm25",
        corpus_path="corpus.jsonl",
        inference=InferenceConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            base_url="http://localhost:30000"
        )
    )
    
    # Step 1: Retrieval only
    print("\n📚 Step 1: Retrieving documents...")
    retrieval_results = pipeline.retrieve(
        queries=["What is AI?", "What is ML?"],
        top_k=20
    )
    print(f"Retrieved {len(retrieval_results)} results")
    
    # Step 2: Optimization
    print("\n🔧 Step 2: Optimizing context ordering...")
    optimized = pipeline.optimize(retrieval_results)
    print(f"Created {len(optimized['groups'])} optimized groups")
    
    # Step 3: Generation (prompts auto-generated with document context)
    print("\n💬 Step 3: Generating responses...")
    print("Note: Prompts are automatically generated with:")
    print("  - Retrieved documents (in optimized order)")
    print("  - Document importance ranking")
    print("  - Formatted instructions")
    generation_results = pipeline.generate(optimized)
    print(f"Generated {generation_results['metadata']['successful_requests']} responses")
    
    # Save results at each step if needed
    pipeline.save_results(
        {"optimized_batch": optimized["groups"], "metadata": optimized["metadata"]},
        "optimized_batch.jsonl"
    )


# ============================================================================
# Run examples
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         ContextPilot Pipeline Usage Examples                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # Uncomment the examples you want to run:
    
    # example_simple_contextpilot()
    # example_faiss_contextpilot()
    # example_with_generation()
    # example_stepwise()
    
    print("\n" + "="*60)
    print("Uncomment the examples in the script to run them")
    print("="*60)
