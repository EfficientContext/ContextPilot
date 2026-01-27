"""
Concurrent Multi-Turn RAG with ContextPilot

Natural production workflow:
1. First Turn: Batch process multiple users' first queries → ContextPilot context optimization
2. Follow-Up Turns: As each response completes, user sends next query → Per-conversation deduplication

Requires running inference server:
  python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Instruct-2507 --port 30000
"""

import concurrent.futures
from contextpilot.pipeline import RAGPipeline, InferenceConfig


def simulate_user_conversation(pipeline, user_id, queries):
    """
    Simulate a single user's multi-turn conversation.
    Each turn: retrieve → deduplicate → generate → user receives response → next query
    """
    print(f"\n{'─' * 80}")
    print(f"User: {user_id} - Starting conversation")
    print(f"{'─' * 80}")
    
    for turn_num, query in enumerate(queries, start=1):
        print(f"\n  [{user_id}] Turn {turn_num}: \"{query[:60]}...\"")
        
        # Process turn with deduplication and generation
        result = pipeline.process_conversation_turn(
            conversation_id=user_id,
            query=query,
            top_k=5,
            enable_deduplication=True,
            generate_response=True,
            max_tokens=128
        )
        
        # Show deduplication stats
        stats = result['deduplication_stats']
        print(f"  [{user_id}] Retrieved: {stats['num_retrieved']} | "
              f"Novel: {stats['num_novel']} | "
              f"Deduplicated: {stats['num_deduplicated']} "
              f"({stats['deduplication_rate']:.0%})")
        
        # Show response
        if result.get('response'):
            print(f"  [{user_id}] Response: {result['response'][:100]}...")
    
    print(f"\n  [{user_id}] ✅ Conversation complete!")
    return user_id


def main():
    """
    Natural multi-turn workflow:
    1. Batch first turns with context optimization
    2. As responses complete, users send follow-ups with deduplication (concurrent)
    """
    print("=" * 80)
    print("Concurrent Multi-Turn RAG with ContextPilot")
    print("=" * 80)
    
    # Initialize pipeline with inference
    pipeline = RAGPipeline(
        retriever="bm25",
        corpus_path="corpus.jsonl",
        use_contextpilot=True,
        inference=InferenceConfig(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            base_url="http://localhost:30000",
            max_tokens=128
        )
    )
    
    # =========================================================================
    # STEP 1: First-Turn Batch (Multiple Concurrent Users)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: First-Turn Batch with Context Optimization")
    print("=" * 80)
    
    first_queries = [
        "What is machine learning?",
        "Explain neural networks", 
        "What is deep learning?",
        "How does backpropagation work?",
        "What are convolutional neural networks?"
    ]
    user_ids = ["user1", "user2", "user3", "user4", "user5"]
    
    print(f"\n🎯 Batch processing {len(first_queries)} first-turn queries with generation...")
    
    # Batch process with context optimization and generation
    first_results = pipeline.run(
        queries=first_queries,
        top_k=5,
        generate_responses=True
    )
    
    print(f"\n✅ Generated {len(first_results['generation_results'])} responses")
    print(f"   Optimization groups: {len(first_results['optimized_batch'])}")
    
    # Show optimization groups
    print("\n📊 Context Optimization Groups:")
    for group in first_results['optimized_batch']:
        print(f"  Group {group['group_id']}: {group['group_size']} users")
    
    # Show generated responses
    print("\n💬 First-Turn Responses:")
    for i, gen_result in enumerate(first_results['generation_results']):
        if gen_result.get('success'):
            print(f"  {user_ids[i]}: {gen_result['generated_text'][:80]}...")
    
    # =========================================================================
    # STEP 2: Follow-Up Turns (Natural Asynchronous Flow)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Follow-Up Turns with Deduplication (Async)")
    print("=" * 80)
    print("\n🎯 Users process conversations asynchronously (like production)...")
    
    # Define complete conversation for each user
    user_conversations = {
        "user1": [
            "What is machine learning?",
            "How does supervised learning work?",
            "What are some ML algorithms?"
        ],
        "user2": [
            "Explain neural networks",
            "What are activation functions?",
            "Explain backprop in neural networks"
        ],
        "user3": [
            "What is deep learning?",
            "How is DL different from ML?",
            "What are deep neural networks?"
        ],
        "user4": [
            "How does backpropagation work?",
            "What is gradient descent?",
            "How to train neural networks?"
        ],
        "user5": [
            "What are convolutional neural networks?",
            "What are CNNs used for?",
            "Explain pooling layers"
        ]
    }
    
    # Process all users' conversations concurrently using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(simulate_user_conversation, pipeline, user_id, queries)
            for user_id, queries in user_conversations.items()
        ]
        
        # Wait for all to complete
        completed_users = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    print(f"\n✅ All {len(completed_users)} users completed their conversations!")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n📊 Per-User Statistics:")
    for user_id in user_ids:
        stats = pipeline.get_conversation_stats(user_id)
        print(f"\n  {user_id}:")
        print(f"    Turns: {stats['turn_count']}")
        print(f"    Retrieved: {stats['total_retrieved']} docs")
        print(f"    Deduplicated: {stats['total_deduplicated']} docs ({stats['deduplication_rate']:.1%})")
    
    print("\n💡 Natural Workflow:")
    print("  ✅ Step 1: Batch first turns → Context optimization + Generation")
    print("  ✅ Step 2: User gets response → Sends follow-up → Deduplication + Generation")
    print("  ✅ Repeat: Each turn processed independently as requests arrive")


if __name__ == "__main__":
    main()
