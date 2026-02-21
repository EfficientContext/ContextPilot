# ContextPilot Batch Preparation

This directory contains a CLI tool for preparing optimized batches from retrieval results.

## Overview

ContextPilot reorders retrieved contexts to maximize prefix sharing, which significantly improves KV cache efficiency during batch inference.

## Files

- **`prepare_batch.py`** - Optimize and group retrieval results for batch inference

## Quick Start

### Prepare Batch

```bash
python prepare_batch.py \
    --context_path retrieval_results.jsonl \
    --output_path optimized_batch.jsonl
```

**Options:**
- `--context_path`: Path to retrieval results (JSONL format)
- `--output_path`: Where to save the prepared batch
- `--use_gpu`: Use GPU for distance computation (recommended for >128 contexts)
- `--linkage_method`: Clustering method - `average`, `complete`, or `single` (default: `average`)
- `--alpha`: Weight for position differences in distance calculation (default: 0.001)

### Run Inference with RAGPipeline

After preparing the batch, use `RAGPipeline` for inference:

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

# Run retrieval + optimization + generation in one call
results = pipeline.run(
    queries=["What is AI?", "What is ML?"],
    generate_responses=True
)

# Access responses
for result in results["generation_results"]:
    print(result["generated_text"])
```

## Input Data Format

The input JSONL file should have this format:

```json
{"qid": 0, "text": "What is the capital of France?", "answer": ["Paris"], "top_k_doc_id": [123, 456, 789]}
```

**Required fields:**
- `qid`: Unique query identifier
- `text`: Query text
- `answer`: List of acceptable answers
- `top_k_doc_id`: List of retrieved document IDs

## Output Format

The prepared batch contains grouped queries optimized for inference:

```json
{
    "group_id": 0,
    "group_size": 5,
    "group_score": 0.85,
    "items": [
        {
            "qid": 1,
            "question": "Query text",
            "answer": ["Answer"],
            "top_k_doc_id": [101, 102, 103],
            "orig_top_k_doc_id": [103, 101, 102]
        }
    ]
}
```

## What It Does

1. **Intra-context reordering**: Reorders documents within each query for better cache sharing
2. **Inter-context clustering**: Groups similar queries using hierarchical clustering
3. **Tree-based scheduling**: Organizes execution order to maximize KV cache hits

## See Also

- [API Reference](../../docs/reference/api.md) - Full endpoint and class reference
- [Offline Usage](../../docs/guides/offline_usage.md) - Complete workflow guide
