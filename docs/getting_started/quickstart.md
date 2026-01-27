# Quick Start

Get up and running with ContextPilot in 5 minutes.

## Prerequisites

1. ContextPilot installed ([Installation Guide](installation.md))
2. A corpus file (`corpus.jsonl`) with documents

## Step 1: Start Your Inference Engine

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 30000
```

Wait for the server to be ready (you'll see "Server is ready").

## Step 2: Prepare Your Data

Create a `corpus.jsonl` file with your documents:

```json
{"doc_id": 1, "text": "Machine learning is a subset of artificial intelligence..."}
{"doc_id": 2, "text": "Neural networks are computing systems inspired by biological neural networks..."}
{"doc_id": 3, "text": "Deep learning is part of a broader family of machine learning methods..."}
```

## Step 3: Run ContextPilot

```python
from contextpilot.pipeline import RAGPipeline, InferenceConfig

# Create pipeline with all components
pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_contextpilot=True,  # Enable ContextPilot optimization
    inference=InferenceConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:30000",
        max_tokens=256,
        temperature=0.0
    )
)

# Run the complete pipeline
results = pipeline.run(
    queries=[
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?"
    ],
    top_k=20,                    # Retrieve top 20 documents per query
    generate_responses=True      # Enable LLM generation
)

# Print results
print(f"Processed {results['metadata']['num_queries']} queries")
print(f"Created {results['metadata']['num_groups']} optimized groups")

for i, gen_result in enumerate(results["generation_results"]):
    if gen_result["success"]:
        print(f"\nQuery {i+1}: {gen_result['generated_text'][:200]}...")
```

## What Just Happened?

1. **Retrieval**: BM25 found the top-20 most relevant documents for each query
2. **Optimization**: ContextPilot clustered queries with overlapping documents for maximum prefix sharing
3. **Generation**: Queries were sent to SGLang in the optimal order

## Next Steps

- [Offline Usage](../guides/offline_usage.md) - More batch processing examples
- [Online Usage](../guides/online_usage.md) - Live index server modes
- [Multi-Turn](../guides/multi_turn.md) - Conversation handling
