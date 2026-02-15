# Offline Usage

Offline mode is best for **batch processing** where you process all queries at once without needing live cache management.

## How ContextPilot Optimizes Batches

ContextPilot performs **two levels of optimization** to maximize KV-cache prefix sharing:

1. **Inter-Context Reordering**: Queries with similar retrieved documents are scheduled together
2. **Intra-Context Reordering**: Document IDs within each query are reordered so shared documents appear first as a common prefix

For example, if Query A retrieves docs `[5, 1, 8, 2]` and Query B retrieves `[2, 9, 1, 5]`, after optimization:
- Query A: `[1, 2, 5, 8]` (shared IDs first)
- Query B: `[1, 2, 5, 9]` (same prefix `[1, 2, 5]`!)

This creates identical prefixes that the inference engine can cache and reuse.

## Prerequisites

1. **Start your inference engine:**
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 30000
```

2. **Prepare your data:**
   - `corpus.jsonl`: Documents with `doc_id` and `text` fields
   - Queries: List of strings or query objects

---

## Example 1: End-to-End Pipeline

Retrieve, optimize, and generate in one call:

```python
from contextpilot.pipeline import RAGPipeline, InferenceConfig

pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_contextpilot=True,
    inference=InferenceConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:30000",
        max_tokens=256,
        temperature=0.0
    )
)

results = pipeline.run(
    queries=[
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?"
    ],
    top_k=20,
    generate_responses=True
)

print(f"Processed {results['metadata']['num_queries']} queries")
print(f"Created {results['metadata']['num_groups']} optimized groups")

for i, gen_result in enumerate(results["generation_results"]):
    if gen_result["success"]:
        print(f"\nQuery {i+1}: {gen_result['generated_text'][:200]}...")
```

---

## Example 2: Retrieval + Optimization Only

Prepare optimized batches without generation (for later inference):

```python
from contextpilot.pipeline import RAGPipeline

pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_contextpilot=True
)

results = pipeline.run(
    queries=["What is AI?", "What is ML?"],
    top_k=20,
    generate_responses=False
)

# Save for later use
pipeline.save_results(results, "optimized_batch.jsonl")
print(f"Saved {len(results['optimized_batch'])} groups")
```

---

## Example 3: Step-by-Step Control

Fine-grained control over each pipeline stage:

```python
from contextpilot.pipeline import RAGPipeline, InferenceConfig

pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    inference=InferenceConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:30000"
    )
)

queries = ["What is machine learning?", "Explain neural networks"]

# Step 1: Retrieve documents
retrieval_results = pipeline.retrieve(queries=queries, top_k=20)
print(f"Retrieved documents for {len(retrieval_results)} queries")

# Step 2: Optimize context ordering
optimized = pipeline.optimize(retrieval_results)
print(f"Created {len(optimized['groups'])} optimized groups")

# Step 3: Generate responses
generation_results = pipeline.generate(optimized)
print(f"Generated {generation_results['metadata']['successful_requests']} responses")

# Inspect groups
for group in optimized['groups']:
    print(f"Group {group['group_id']}: {group['group_size']} queries, score={group['group_score']:.3f}")
```

---

## Example 4: Compare With/Without ContextPilot

```python
from contextpilot.pipeline import RAGPipeline, InferenceConfig

queries = ["What is AI?", "What is ML?", "What is DL?"]

# With ContextPilot optimization
pipeline_optimized = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_contextpilot=True,
    inference=InferenceConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:30000"
    )
)
results_optimized = pipeline_optimized.run(queries=queries, generate_responses=True)

# Without ContextPilot (standard RAG)
pipeline_standard = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_contextpilot=False,
    inference=InferenceConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:30000"
    )
)
results_standard = pipeline_standard.run(queries=queries, generate_responses=True)

# Compare timings
print(f"ContextPilot: {results_optimized['metadata']['total_time']:.2f}s")
print(f"Standard: {results_standard['metadata']['total_time']:.2f}s")
```

---

## Example 5: Using FAISS Retriever

For semantic search with embeddings:

```bash
# First, start an embedding server:
python -m sglang.launch_server \
    --model-path Alibaba-NLP/gte-Qwen2-7B-instruct \
    --is-embedding \
    --port 30001
```

```python
from contextpilot.pipeline import RAGPipeline, InferenceConfig

pipeline = RAGPipeline(
    retriever="faiss",
    corpus_path="corpus.jsonl",
    index_path="faiss_index.faiss",  # Created if doesn't exist
    embedding_model="Alibaba-NLP/gte-Qwen2-7B-instruct",
    embedding_base_url="http://localhost:30001",
    use_contextpilot=True,
    inference=InferenceConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:30000"
    )
)

results = pipeline.run(
    queries=["Explain quantum computing"],
    generate_responses=True
)
```

---

## Next Steps

- [Online Usage](online_usage.md) - Live index server modes
- [Multi-Turn](multi_turn.md) - Conversation handling
- [API Reference](../reference/api.md) - Full API documentation
