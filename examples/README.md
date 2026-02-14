# ContextPilot Examples

Practical code examples for using ContextPilot.

## Quick Reference

| Example | Description | Prerequisites |
|---------|-------------|---------------|
| `simple_reorder_example.py` | Minimal "hello world" — call `/schedule` with 4 contexts | ContextPilot server (stateless) |
| `pipeline_examples.py` | Pipeline API usage (BM25, FAISS, generation) | Corpus file |
| `http_server_example.py` | Stateful index server (build, proxy, eviction) | ContextPilot + SGLang |
| `stateless_batch_example.py` | Stateless batch scheduling (3 approaches) | ContextPilot server (stateless) |
| `stateless_sglang_e2e.py` | Stateless scheduling → SGLang inference e2e | ContextPilot + SGLang |
| `pageindex_e2e_example.py` | PageIndex tree search + ContextPilot optimization | PageIndex package |
| `mem0_bench_simple.py` | A/B benchmark: Baseline vs ContextPilot (cache hit rate) | mem0ai + OpenAI key + SGLang |
| `mem0_locomo_example.py` | LoCoMo multi-turn benchmark with TTFT & F1 scoring | mem0ai + OpenAI key + SGLang |

## Running Examples

### 1. Minimal Example (Stateless)

```bash
# Terminal 1: Start server in stateless mode
python -m contextpilot.server.http_server --port 8765 --stateless

# Terminal 2: Run
python examples/simple_reorder_example.py
```

### 2. Pipeline API (Offline)

```bash
python examples/pipeline_examples.py
```

### 3. Stateful Server (Live Mode)

```bash
# Terminal 1: Start SGLang with ContextPilot eviction callback
CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B --port 30000 --schedule-policy lpm

# Terminal 2: Start ContextPilot server
python -m contextpilot.server.http_server --port 8765 --infer-api-url http://localhost:30000

# Terminal 3: Run example
python examples/http_server_example.py
```

### 4. Stateless Scheduling → SGLang e2e

```bash
# Terminal 1: Start SGLang
python -m sglang.launch_server --model-path Qwen/Qwen3-4B --port 30000

# Terminal 2: Start ContextPilot (stateless — no CONTEXTPILOT_INDEX_URL needed on SGLang)
python -m contextpilot.server.http_server --port 8765 --stateless --infer-api-url http://localhost:30000

# Terminal 3: Run
python examples/stateless_sglang_e2e.py
```

### 5. mem0 A/B Benchmark

```bash
pip install mem0ai
export OPENAI_API_KEY="your-key"

# Start SGLang + ContextPilot (live mode), then:
python examples/mem0_bench_simple.py
```

## Directory Structure

```
examples/
├── simple_reorder_example.py     # Minimal hello world
├── pipeline_examples.py          # Pipeline API (BM25, FAISS, generation)
├── http_server_example.py        # Stateful index server
├── stateless_batch_example.py    # Stateless batch scheduling
├── stateless_sglang_e2e.py       # Stateless → SGLang e2e
├── pageindex_e2e_example.py      # PageIndex + ContextPilot
├── mem0_bench_simple.py          # A/B cache hit benchmark
├── mem0_locomo_example.py        # LoCoMo multi-turn benchmark
├── batch_inference/              # Batch processing utilities
│   └── prepare_batch.py          # Batch optimization
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
