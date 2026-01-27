"""
Example: Multi-turn RAG conversations with context deduplication.

Demonstrates how to use ContextPilot's multi-turn conversation feature to:
1. Maintain conversation state across multiple turns
2. Deduplicate redundant document contexts
3. Track deduplication statistics
"""

from contextpilot.pipeline import RAGPipeline, InferenceConfig


def example_multi_turn_conversation():
    """
    Example of a multi-turn conversation with context deduplication.
    
    Shows how documents are deduplicated across conversation turns
    to reduce redundant prefill computation.
    """
    print("\n" + "="*70)
    print("Example: Multi-turn Conversation with Context Deduplication")
    print("="*70)
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        retriever="bm25",
        corpus_path="/home/jysc/Demnok/datasets/multihopRAG/mulhoprag_corpus.jsonl",
        use_contextpilot=True  # Enable ContextPilot optimization
    )
    
    # Conversation ID - maintains state across turns
    conversation_id = "example_conversation_1"
    
    # Turn 1: Initial query
    print("\n📝 Turn 1: Initial query")
    print("-" * 70)
    
    result1 = pipeline.process_conversation_turn(
        conversation_id=conversation_id,
        query="What are the latest tech deals?",
        top_k=5,
        enable_deduplication=True  # Enable context deduplication
    )
    
    print(f"Query: {result1['query']}")
    print(f"Retrieved documents: {result1['retrieved_docs']}")
    print(f"Novel documents: {result1['novel_docs']}")
    print(f"Deduplicated documents: {result1['deduplicated_docs']}")
    print(f"\nDeduplication stats:")
    print(f"  - Retrieved: {result1['deduplication_stats']['num_retrieved']}")
    print(f"  - Novel: {result1['deduplication_stats']['num_novel']}")
    print(f"  - Deduplicated: {result1['deduplication_stats']['num_deduplicated']}")
    print(f"  - Rate: {result1['deduplication_stats']['deduplication_rate']:.1%}")
    
    # Turn 2: Follow-up query (will have overlapping documents)
    print("\n📝 Turn 2: Follow-up query")
    print("-" * 70)
    
    result2 = pipeline.process_conversation_turn(
        conversation_id=conversation_id,
        query="Which deals are best for laptops?",
        top_k=5,
        enable_deduplication=True
    )
    
    print(f"Query: {result2['query']}")
    print(f"Retrieved documents: {result2['retrieved_docs']}")
    print(f"Novel documents: {result2['novel_docs']}")
    print(f"Deduplicated documents: {result2['deduplicated_docs']}")
    print(f"\nDeduplication stats:")
    print(f"  - Retrieved: {result2['deduplication_stats']['num_retrieved']}")
    print(f"  - Novel: {result2['deduplication_stats']['num_novel']}")
    print(f"  - Deduplicated: {result2['deduplication_stats']['num_deduplicated']}")
    print(f"  - Rate: {result2['deduplication_stats']['deduplication_rate']:.1%}")
    
    # Turn 3: Another follow-up
    print("\n📝 Turn 3: Another follow-up")
    print("-" * 70)
    
    result3 = pipeline.process_conversation_turn(
        conversation_id=conversation_id,
        query="What about Black Friday deals?",
        top_k=5,
        enable_deduplication=True
    )
    
    print(f"Query: {result3['query']}")
    print(f"Retrieved documents: {result3['retrieved_docs']}")
    print(f"Novel documents: {result3['novel_docs']}")
    print(f"Deduplicated documents: {result3['deduplicated_docs']}")
    print(f"\nDeduplication stats:")
    print(f"  - Retrieved: {result3['deduplication_stats']['num_retrieved']}")
    print(f"  - Novel: {result3['deduplication_stats']['num_novel']}")
    print(f"  - Deduplicated: {result3['deduplication_stats']['num_deduplicated']}")
    print(f"  - Rate: {result3['deduplication_stats']['deduplication_rate']:.1%}")
    
    # Overall conversation statistics
    print("\n📊 Overall Conversation Statistics")
    print("-" * 70)
    
    conv_stats = pipeline.get_conversation_stats(conversation_id)
    print(f"Conversation ID: {conv_stats['conversation_id']}")
    print(f"Total turns: {conv_stats['turn_count']}")
    print(f"Total documents retrieved: {conv_stats['total_retrieved']}")
    print(f"Total novel documents: {conv_stats['total_novel']}")
    print(f"Total deduplicated: {conv_stats['total_deduplicated']}")
    print(f"Overall deduplication rate: {conv_stats['deduplication_rate']:.1%}")
    
    # Show context snippet for last turn
    print("\n📄 Context for Turn 3 (with location hints):")
    print("-" * 70)
    print(result3['context'][:800] + "...")


def example_multi_turn_with_generation():
    """
    Example with LLM generation enabled.
    
    Shows how to generate responses while maintaining conversation context
    and applying deduplication.
    """
    print("\n" + "="*70)
    print("Example: Multi-turn with LLM Generation")
    print("="*70)
    
    # Initialize pipeline with inference config
    pipeline = RAGPipeline(
        retriever="bm25",
        corpus_path="/home/jysc/Demnok/datasets/multihopRAG/mulhoprag_corpus.jsonl",
        use_contextpilot=True,
        inference=InferenceConfig(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            backend="sglang",
            base_url="http://localhost:30000",
            max_tokens=100
        )
    )
    
    conversation_id = "example_conversation_2"
    
    # Turn 1
    print("\n📝 Turn 1")
    result1 = pipeline.process_conversation_turn(
        conversation_id=conversation_id,
        query="What are gaming laptop deals?",
        top_k=3,
        enable_deduplication=True,
        generate_response=True
    )
    
    print(f"Query: {result1['query']}")
    print(f"Deduplication: {result1['deduplication_stats']['num_deduplicated']} of {result1['deduplication_stats']['num_retrieved']}")
    if result1.get('response'):
        print(f"Response: {result1['response'][:200]}...")
    
    # Turn 2
    print("\n📝 Turn 2")
    result2 = pipeline.process_conversation_turn(
        conversation_id=conversation_id,
        query="Which one has the best graphics card?",
        top_k=3,
        enable_deduplication=True,
        generate_response=True
    )
    
    print(f"Query: {result2['query']}")
    print(f"Deduplication: {result2['deduplication_stats']['num_deduplicated']} of {result2['deduplication_stats']['num_retrieved']}")
    if result2.get('response'):
        print(f"Response: {result2['response'][:200]}...")
    
    # Show overall stats
    all_stats = pipeline.get_conversation_stats()
    print(f"\n📊 All Conversations: {all_stats['total_conversations']} conversations, "
          f"{all_stats['total_turns']} turns, "
          f"{all_stats['average_deduplication_rate']:.1%} deduplication rate")


def example_baseline_comparison():
    """
    Compare with and without deduplication (baseline).
    
    Shows the difference in number of documents processed when
    deduplication is enabled vs disabled.
    """
    print("\n" + "="*70)
    print("Example: Baseline vs. Deduplication Comparison")
    print("="*70)
    
    pipeline = RAGPipeline(
        retriever="bm25",
        corpus_path="/home/jysc/Demnok/datasets/multihopRAG/mulhoprag_corpus.jsonl",
        use_contextpilot=True
    )
    
    queries = [
        "What are laptop deals?",
        "Which laptops have best specs?",
        "Any gaming laptop deals?"
    ]
    
    # Run with deduplication
    print("\n🔄 WITH Context Deduplication:")
    print("-" * 70)
    
    conv_id_dedup = "comparison_with_dedup"
    total_novel_dedup = 0
    total_retrieved_dedup = 0
    
    for i, query in enumerate(queries, 1):
        result = pipeline.process_conversation_turn(
            conversation_id=conv_id_dedup,
            query=query,
            top_k=5,
            enable_deduplication=True
        )
        print(f"Turn {i}: {result['deduplication_stats']['num_novel']} novel / "
              f"{result['deduplication_stats']['num_retrieved']} retrieved "
              f"({result['deduplication_stats']['deduplication_rate']:.0%} dedup)")
        total_novel_dedup += result['deduplication_stats']['num_novel']
        total_retrieved_dedup += result['deduplication_stats']['num_retrieved']
    
    print(f"\nTotal: {total_novel_dedup} novel documents processed / {total_retrieved_dedup} retrieved")
    
    # Run without deduplication (baseline)
    print("\n❌ WITHOUT Context Deduplication (Baseline):")
    print("-" * 70)
    
    conv_id_baseline = "comparison_baseline"
    total_novel_baseline = 0
    total_retrieved_baseline = 0
    
    for i, query in enumerate(queries, 1):
        result = pipeline.process_conversation_turn(
            conversation_id=conv_id_baseline,
            query=query,
            top_k=5,
            enable_deduplication=False  # Baseline mode
        )
        print(f"Turn {i}: {result['deduplication_stats']['num_novel']} novel / "
              f"{result['deduplication_stats']['num_retrieved']} retrieved "
              f"({result['deduplication_stats']['deduplication_rate']:.0%} dedup)")
        total_novel_baseline += result['deduplication_stats']['num_novel']
        total_retrieved_baseline += result['deduplication_stats']['num_retrieved']
    
    print(f"\nTotal: {total_novel_baseline} novel documents processed / {total_retrieved_baseline} retrieved")
    
    # Show savings
    print("\n💡 Savings:")
    print("-" * 70)
    docs_saved = total_novel_baseline - total_novel_dedup
    saving_rate = docs_saved / total_novel_baseline if total_novel_baseline > 0 else 0
    print(f"Avoided processing {docs_saved} redundant documents ({saving_rate:.1%} reduction)")


if __name__ == "__main__":
    # Run examples
    example_multi_turn_conversation()
    
    # Uncomment to run with LLM generation (requires running server)
    # example_multi_turn_with_generation()
    
    # Uncomment to see baseline comparison
    # example_baseline_comparison()
