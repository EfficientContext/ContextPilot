"""
Example: Using mem0 Memory with ContextPilot

This example demonstrates how to integrate mem0's Memory system with
ContextPilot for personalized RAG with context reordering and deduplication.

mem0 stores user conversation memories, making it ideal for:
- Personalized chatbots that remember user preferences
- Multi-turn conversations with historical context
- Agent memory systems that accumulate knowledge

ContextPilot optimizes mem0 memories by:
- Reordering memories for optimal KV cache efficiency
- Deduplicating overlapping memories across conversation turns
- Scheduling memory retrieval for maximum prefix sharing
"""

# Option 1: Using the high-level RAGPipeline API
# =============================================

def pipeline_example():
    """Example using RAGPipeline with mem0 retriever."""
    from contextpilot import RAGPipeline, RetrieverConfig, MEM0_AVAILABLE
    
    if not MEM0_AVAILABLE:
        print("⚠️  mem0 not installed. Install with: pip install mem0ai")
        return
    
    # Create a pipeline with mem0 retriever
    pipeline = RAGPipeline(
        retriever=RetrieverConfig(
            retriever_type="mem0",
            top_k=10,
            # mem0 configuration
            mem0_user_id="user_alice",  # Scope memories to this user
            mem0_config={
                "llm": {
                    "provider": "openai",
                    "config": {"model": "gpt-4o-mini"}
                },
                "embedder": {
                    "provider": "openai",
                    "config": {"model": "text-embedding-3-small"}
                }
            },
        ),
        use_contextpilot=True,  # Enable context optimization
    )
    
    # Run queries - memories are automatically retrieved and optimized
    results = pipeline.run(
        queries=[
            "What are my dietary preferences?",
            "What projects am I working on?",
            "What did we discuss last time?",
        ]
    )
    
    print("Pipeline Results:")
    for result in results:
        print(f"  Query: {result.get('text', '')[:50]}...")
        print(f"  Retrieved memories: {len(result.get('top_k_doc_id', []))}")
        print()


# Option 2: Using Mem0Retriever directly
# ======================================

def retriever_example():
    """Example using Mem0Retriever directly for more control."""
    from contextpilot.retriever import Mem0Retriever, MEM0_AVAILABLE
    from contextpilot.pipeline import MultiTurnManager
    
    if not MEM0_AVAILABLE:
        print("⚠️  mem0 not installed. Install with: pip install mem0ai")
        return
    
    # Initialize the mem0 retriever
    # By default, use_integer_ids=True maps mem0 UUIDs to integers for
    # better compatibility with ContextPilot's reordering/deduplication
    retriever = Mem0Retriever(
        config={
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-4o-mini"}
            }
        },
        use_integer_ids=True,  # Default: map UUIDs to integers
    )
    
    user_id = "user_bob"
    
    # Add some memories for the user
    print("Adding memories...")
    retriever.add_memory(
        "I prefer vegetarian food and am allergic to peanuts",
        user_id=user_id
    )
    retriever.add_memory(
        "I'm working on a machine learning project about NLP",
        user_id=user_id
    )
    retriever.add_memory(
        [
            {"role": "user", "content": "Can you recommend a restaurant?"},
            {"role": "assistant", "content": "Based on your vegetarian preference, I recommend Green Garden."}
        ],
        user_id=user_id
    )
    
    # Search memories
    print("\nSearching memories...")
    results = retriever.search_queries(
        query_data=[
            {"qid": 0, "text": "What food preferences do I have?"},
            {"qid": 1, "text": "What am I working on?"},
        ],
        user_id=user_id,
        top_k=5
    )
    
    for result in results:
        print(f"\nQuery: {result['text']}")
        print(f"Retrieved {len(result['top_k_doc_id'])} memories")
        # Document IDs are now integers (e.g., [0, 1, 2]) instead of UUIDs
        print(f"Doc IDs (integers): {result['top_k_doc_id']}")
        for mem in result.get('memories', [])[:3]:
            print(f"  - {mem.get('memory', '')[:80]}...")
    
    # Access ID mapping if needed
    print("\nID Mapping (integer -> mem0 UUID):")
    for int_id, uuid in list(retriever.get_id_mapping().items())[:3]:
        print(f"  {int_id} -> {uuid}")
    
    # Use with ContextPilot's multi-turn deduplication
    print("\n\nMulti-turn deduplication example:")
    multi_turn = MultiTurnManager()
    corpus_map = retriever.get_corpus_map()
    
    # First turn
    first_result = results[0]
    context1, novel1, stats1 = multi_turn.deduplicate_context(
        conversation_id="conv_123",
        retrieved_doc_ids=first_result['top_k_doc_id'],
        corpus_map=corpus_map,
        enable_deduplication=True
    )
    print(f"Turn 1: {stats1['num_retrieved']} retrieved, {stats1['num_novel']} novel")
    
    # Second turn (with overlapping memories)
    second_result = results[1]
    context2, novel2, stats2 = multi_turn.deduplicate_context(
        conversation_id="conv_123",
        retrieved_doc_ids=second_result['top_k_doc_id'],
        corpus_map=corpus_map,
        enable_deduplication=True
    )
    print(f"Turn 2: {stats2['num_retrieved']} retrieved, {stats2['num_novel']} novel, {stats2['num_deduplicated']} deduplicated")


# Option 3: Using an existing mem0 Memory instance
# ================================================

def existing_memory_example():
    """Example using an existing mem0 Memory instance."""
    try:
        from mem0 import Memory
    except ImportError:
        print("⚠️  mem0 not installed. Install with: pip install mem0ai")
        return
    
    from contextpilot.retriever import Mem0Retriever
    
    # Create your own mem0 Memory with custom configuration
    memory = Memory.from_config({
        "llm": {
            "provider": "openai",
            "config": {"model": "gpt-4o-mini"}
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"}
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "my_memories",
                "host": "localhost",
                "port": 6333,
            }
        }
    })
    
    # Pass the existing memory to ContextPilot
    retriever = Mem0Retriever(memory=memory)
    
    # Now use the retriever with ContextPilot
    print("Using existing mem0 Memory with ContextPilot")
    print(f"Retriever ready: {retriever is not None}")


# Option 4: Using mem0 Cloud API
# ==============================

def cloud_api_example():
    """Example using mem0's cloud API with ContextPilot."""
    from contextpilot.retriever import Mem0Retriever, MEM0_AVAILABLE
    
    if not MEM0_AVAILABLE:
        print("⚠️  mem0 not installed. Install with: pip install mem0ai")
        return
    
    import os
    api_key = os.environ.get("MEM0_API_KEY")
    
    if not api_key:
        print("⚠️  MEM0_API_KEY environment variable not set")
        return
    
    # Use mem0 cloud API
    retriever = Mem0Retriever(
        use_client=True,
        api_key=api_key
    )
    
    # Use the same way as local memory
    results = retriever.search_queries(
        query_data=[{"qid": 0, "text": "My preferences"}],
        user_id="cloud_user_1",
        top_k=10
    )
    
    print(f"Retrieved {len(results[0]['top_k_doc_id'])} memories from cloud")


# Option 5: Indexing a corpus into mem0
# =====================================

def index_corpus_example():
    """Example of indexing an existing corpus into mem0."""
    from contextpilot import RAGPipeline, RetrieverConfig, MEM0_AVAILABLE
    
    if not MEM0_AVAILABLE:
        print("⚠️  mem0 not installed. Install with: pip install mem0ai")
        return
    
    # Sample corpus data
    corpus = [
        {"chunk_id": 0, "text": "Python is a programming language.", "title": "Python Intro"},
        {"chunk_id": 1, "text": "Machine learning uses statistical methods.", "title": "ML Basics"},
        {"chunk_id": 2, "text": "Neural networks are inspired by the brain.", "title": "Neural Networks"},
    ]
    
    # Create pipeline with corpus - will be indexed into mem0
    pipeline = RAGPipeline(
        retriever=RetrieverConfig(
            retriever_type="mem0",
            top_k=5,
            corpus_data=corpus,  # Index this corpus
            mem0_user_id="corpus_user",
        ),
    )
    
    # Query the indexed corpus
    results = pipeline.run(queries=["What is Python?"])
    print(f"Found {len(results)} results")


if __name__ == "__main__":
    print("=" * 60)
    print("mem0 + ContextPilot Integration Examples")
    print("=" * 60)
    
    print("\n1. Retriever Example:")
    print("-" * 40)
    retriever_example()
    
    # Uncomment to run other examples:
    # print("\n2. Pipeline Example:")
    # print("-" * 40)
    # pipeline_example()
    
    # print("\n3. Existing Memory Example:")
    # print("-" * 40)
    # existing_memory_example()
    
    # print("\n4. Cloud API Example:")
    # print("-" * 40)
    # cloud_api_example()
    
    # print("\n5. Index Corpus Example:")
    # print("-" * 40)
    # index_corpus_example()
