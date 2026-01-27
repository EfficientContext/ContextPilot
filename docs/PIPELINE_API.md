## ContextPilot Pipeline API

A high-level, user-friendly abstraction for building RAG pipelines with or without ContextPilot optimization.

### Features

- **Simple API**: Build complete RAG pipelines with just a few lines of code
- **Flexible Configuration**: Use simple strings or detailed config objects
- **Multiple Retrievers**: Support for BM25, FAISS, and custom retrievers
- **Optional Optimization**: Easy toggle between ContextPilot and standard RAG
- **Modular Design**: Run full pipeline or individual steps (retrieve → optimize → generate)

---

### Quick Start

#### Basic Usage

```python
from contextpilot.pipeline import RAGPipeline

# Create pipeline
pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    model="Qwen/Qwen2.5-7B-Instruct"
)

# Run on queries
results = pipeline.run(queries=[
    "What is artificial intelligence?",
    "What is machine learning?"
])

# Save results
pipeline.save_results(results, "output.jsonl")
```

#### Without ContextPilot Optimization

```python
# Standard RAG (no optimization)
pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_contextpilot=False
)

results = pipeline.run(queries=["What is AI?"])
```

---

### Configuration

#### Simple Configuration

Use simple strings and keyword arguments for quick setup:

```python
pipeline = RAGPipeline(
    retriever="bm25",              # or "faiss"
    corpus_path="corpus.jsonl",
    model="Qwen/Qwen2.5-7B-Instruct",
    use_contextpilot=True,             # Enable ContextPilot
    use_gpu=True,                  # Use GPU for distance computation
    linkage_method="average",      # Clustering method
    top_k=20                       # Number of docs to retrieve
)
```

#### Advanced Configuration

Use configuration objects for full control:

```python
from contextpilot.pipeline import (
    RAGPipeline,
    RetrieverConfig,
    OptimizerConfig,
    InferenceConfig
)

pipeline = RAGPipeline(
    retriever=RetrieverConfig(
        retriever_type="bm25",
        top_k=20,
        corpus_path="corpus.jsonl",
        es_host="http://localhost:9200",
        es_index_name="my_index"
    ),
    optimizer=OptimizerConfig(
        enabled=True,
        use_gpu=True,
        linkage_method="average",
        alpha=0.001
    ),
    inference=InferenceConfig(
        model_name="Qwen/Qwen2.5-32B-Instruct",
        backend="sglang",
        temperature=0.0,
        max_tokens=512
    )
)
```

---

### Retriever Options

#### BM25 (Elasticsearch)

```python
pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    es_host="http://localhost:9200",
    es_index_name="bm25_index"
)
```

#### FAISS (Semantic Search)

```python
pipeline = RAGPipeline(
    retriever="faiss",
    corpus_path="corpus.jsonl",
    index_path="faiss_index.faiss",
    embedding_model="Alibaba-NLP/gte-Qwen2-7B-instruct",
    embedding_base_url="http://localhost:30000"
)
```

#### Custom Retriever

```python
class MyRetriever:
    """Custom retriever - just implement the retrieve() method."""
    
    def retrieve(self, queries, top_k=20):
        # Your retrieval logic
        return [
            {
                "qid": q.get("qid", i),
                "text": q["text"],
                "top_k_doc_id": [...]  # Retrieved doc IDs
            }
            for i, q in enumerate(queries)
        ]

pipeline = RAGPipeline(
    retriever=MyRetriever(),
    corpus_path="corpus.jsonl"
)
```

---

### Step-by-Step Pipeline

Run pipeline steps individually for more control:

```python
pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl"
)

# Step 1: Retrieval
retrieval_results = pipeline.retrieve(
    queries=["What is AI?"],
    top_k=20
)

# Step 2: Optimization (ContextPilot)
optimized = pipeline.optimize(retrieval_results)

# Step 3: Save for inference
pipeline.save_results(
    {"optimized_batch": optimized["groups"]},
    "batch.jsonl"
)

# Step 4: Use with existing inference scripts
# python examples/batch_inference/sglang_inference.py \
#   --batch_path batch.jsonl \
#   --corpus_path corpus.jsonl
```

---

### Query Formats

The pipeline accepts queries in multiple formats:

```python
# Single string
pipeline.run(queries="What is AI?")

# List of strings
pipeline.run(queries=[
    "What is AI?",
    "What is ML?"
])

# List of dictionaries (with metadata)
pipeline.run(queries=[
    {
        "qid": 1,
        "text": "What is AI?",
        "answer": ["Artificial Intelligence"]
    },
    {
        "qid": 2,
        "text": "What is ML?",
        "answer": ["Machine Learning"]
    }
])
```

---

### Output Format

The pipeline returns optimized batches grouped for maximum cache efficiency:

```python
{
    "optimized_batch": [
        {
            "group_id": 0,
            "group_size": 3,
            "group_score": 0.85,
            "items": [
                {
                    "qid": 1,
                    "text": "What is AI?",
                    "answer": ["..."],
                    "top_k_doc_id": [5, 12, 3, ...],      # Reordered
                    "orig_top_k_doc_id": [12, 5, 3, ...]  # Original order
                },
                ...
            ]
        },
        ...
    ],
    "metadata": {
        "optimized": true,
        "num_groups": 5,
        "num_queries": 15,
        "total_time": 2.34,
        "linkage_method": "average",
        "use_gpu": true
    }
}
```

---

### Full Example

```python
from contextpilot.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="data/corpus.jsonl",
    model="Qwen/Qwen2.5-7B-Instruct",
    use_contextpilot=True,
    use_gpu=True,
    top_k=20
)

# Prepare queries
queries = [
    {"qid": 1, "text": "What is artificial intelligence?"},
    {"qid": 2, "text": "What is machine learning?"},
    {"qid": 3, "text": "What is deep learning?"},
]

# Run pipeline
results = pipeline.run(queries)

# Save optimized batch
pipeline.save_results(results, "output/optimized_batch.jsonl")

# Print statistics
print(f"Queries processed: {results['metadata']['num_queries']}")
print(f"Groups created: {results['metadata']['num_groups']}")
print(f"Total time: {results['metadata']['total_time']:.2f}s")
```

---

### Comparison: ContextPilot vs Standard RAG

```python
from contextpilot.pipeline import RAGPipeline

queries = ["What is AI?", "What is ML?"]

# With ContextPilot
contextpilot = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_contextpilot=True
)
contextpilot_results = contextpilot.run(queries)

# Without ContextPilot
standard = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_contextpilot=False
)
standard_results = standard.run(queries)

# Compare
print(f"ContextPilot groups: {len(contextpilot_results['optimized_batch'])}")
print(f"Standard groups: {len(standard_results['optimized_batch'])}")
```

---

### Integration with Existing Code

The pipeline is designed to integrate seamlessly with existing ContextPilot workflows:

```python
# 1. Use pipeline for retrieval + optimization
pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_contextpilot=True
)

results = pipeline.run(queries)
pipeline.save_results(results, "batch.jsonl")

# 2. Use existing inference scripts
# $ python examples/batch_inference/sglang_inference.py \
#     --batch_path batch.jsonl \
#     --corpus_path corpus.jsonl \
#     --model Qwen/Qwen2.5-7B-Instruct
```

---

### API Reference

#### RAGPipeline

Main pipeline class that orchestrates retrieval, optimization, and inference.

**Constructor Parameters:**
- `retriever`: Retriever type ("bm25", "faiss") or config object or custom instance
- `optimizer`: OptimizerConfig or bool to enable/disable ContextPilot
- `inference`: InferenceConfig or model name string
- `corpus_path`: Path to corpus JSONL file
- `corpus_data`: List of corpus documents (alternative to corpus_path)
- `model`: Model name (shorthand for inference config)
- `use_contextpilot`: Enable/disable ContextPilot (default: True)
- `**kwargs`: Additional configuration options

**Methods:**
- `setup()`: Initialize pipeline components
- `retrieve(queries, top_k)`: Retrieve documents for queries
- `optimize(retrieval_results)`: Apply ContextPilot optimization
- `run(queries, top_k, return_intermediate)`: Run complete pipeline
- `save_results(results, output_path, format)`: Save results to file

#### Configuration Classes

- **RetrieverConfig**: Configure retrieval settings
- **OptimizerConfig**: Configure ContextPilot optimization
- **InferenceConfig**: Configure LLM inference
- **PipelineConfig**: Overall pipeline configuration

See `contextpilot/pipeline/components.py` for detailed configuration options.

---

### Examples

See `examples/pipeline_examples.py` for comprehensive usage examples:

1. Simple ContextPilot pipeline with BM25
2. FAISS + ContextPilot
3. End-to-End with LLM Generation
4. Step-by-step pipeline control

---

### Notes

- The pipeline automatically handles index creation for BM25 and FAISS
- GPU acceleration is used by default for distance computation (if available)
- Retrieval results are cached within a pipeline instance
- Prompts are automatically generated with full RAG context (documents + importance ranking)

---

### See Also

- [Main README](../README.md)
- [Quick Start Guide](QUICK_START.md)
- [Batch Workflow Guide](BATCH_WORKFLOW.md)
- [Examples Directory](../examples/)
