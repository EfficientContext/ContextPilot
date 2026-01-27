# ContextPilot Examples

Practical code examples for using ContextPilot. See [docs/USAGE.md](../docs/USAGE.md) for comprehensive documentation.

## Quick Reference

| Example | Description | Prerequisites |
|---------|-------------|---------------|
| `pipeline_examples.py` | Basic Pipeline API usage | Corpus file |
| `multi_turn_example.py` | Multi-turn conversation with deduplication | Corpus file |
| `http_server_example.py` | Index server integration | ContextPilot server running |
| `stateless_batch_example.py` | Stateless batch scheduling | ContextPilot server (stateless mode) |

## Running Examples

### 1. Basic Pipeline (Offline Mode)

```bash
# Start inference server
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 30000

# Run pipeline example
python examples/pipeline_examples.py
```

### 2. Multi-Turn Conversations

```bash
python examples/multi_turn_example.py
```

### 3. Index Server (Stateless Mode)

```bash
# Terminal 1: Start server
python -m contextpilot.server.http_server --port 8765 --stateless

# Terminal 2: Run example
python examples/stateless_batch_example.py
```

### 4. Index Server (Stateful Mode)

```bash
# Terminal 1: Start SGLang
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 30000

# Terminal 2: Start ContextPilot server
python -m contextpilot.server.http_server --port 8765 --max-tokens 1000000 --infer-api-url http://localhost:30000

# Terminal 3: Run example
python examples/http_server_example.py
```

## Directory Structure

```
examples/
├── pipeline_examples.py          # Basic Pipeline API
├── multi_turn_example.py         # Multi-turn with deduplication
├── http_server_example.py        # Stateful index server
├── stateless_batch_example.py    # Stateless batch scheduling
├── batch_inference/              # Low-level batch processing
│   ├── prepare_batch.py          # Batch optimization
│   ├── sglang_inference.py       # SGLang inference
│   └── analyze_results.py        # Performance analysis
└── construct_rag_data/           # Data retrieval utilities
    ├── multihopRAG_bm25.py       # BM25 retrieval
    └── multihopRAG_faiss.py      # FAISS retrieval
```

## Data Formats

**Queries (JSONL):**
```json
{"qid": 0, "text": "What is machine learning?"}
```

**Corpus (JSONL):**
```json
{"doc_id": 0, "text": "Machine learning is a subset of AI..."}
```

## More Information

See [docs/USAGE.md](../docs/USAGE.md) for complete documentation.
