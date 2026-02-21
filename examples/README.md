# ContextPilot Examples

Practical code examples for using ContextPilot.

## Quick Reference

| Example | Description | Prerequisites |
|---------|-------------|---------------|
| `simple_reorder_example.py` | Minimal "hello world" — call `/reorder` with 4 contexts | ContextPilot server (stateless) |
| `pipeline_examples.py` | Pipeline API usage (BM25, FAISS, generation) | Corpus file |
| `http_server_example.py` | Stateful index server (build, proxy, eviction) | ContextPilot + inference engine |
| `vllm_patch_e2e_check.py` | Supervisor check: 2-request reorder + vLLM eviction sync | ContextPilot + patched vLLM |
| `stateless_batch_example.py` | Stateless batch reordering (3 approaches) | ContextPilot server (stateless) |
| `stateless_sglang_e2e.py` | Stateless reordering → inference e2e | ContextPilot + inference engine |
| `pageindex_e2e_example.py` | PageIndex tree → ContextPilot scheduling with prefix sharing | ContextPilot (demo: none; full: PageIndex + OpenAI) |
| `mem0_bench_simple.py` | A/B benchmark: Baseline vs ContextPilot (cache hit rate) | mem0ai + OpenAI key + inference engine |
| `mem0_locomo_example.py` | LoCoMo multi-turn benchmark with TTFT & F1 scoring | mem0ai + OpenAI key + inference engine |

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
# Terminal 1: Start inference engine with ContextPilot eviction callback
# SGLang:
CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B --port 30000 --schedule-policy lpm
# or vLLM:
CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --port 30000 --enable-prefix-caching

# Terminal 2: Start ContextPilot server
python -m contextpilot.server.http_server --port 8765 --infer-api-url http://localhost:30000

# Terminal 3: Run example
python examples/http_server_example.py
```

### 4. Stateless Reordering → Inference e2e

```bash
# Terminal 1: Start inference engine
python -m sglang.launch_server --model-path Qwen/Qwen3-4B --port 30000
# or: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B --port 30000 --enable-prefix-caching

# Terminal 2: Start ContextPilot (stateless)
python -m contextpilot.server.http_server --port 8765 --stateless --infer-api-url http://localhost:30000

# Terminal 3: Run
python examples/stateless_sglang_e2e.py
```

### 5. vLLM Patch E2E Check (Supervisor Validation)

```bash
# Terminal 1: Start ContextPilot
python -m contextpilot.server.http_server --port 8765

# Terminal 2: Start patched vLLM (prefix caching enabled)
CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --port 8000 --enable-prefix-caching

# Terminal 3: Run verifier
python examples/vllm_patch_e2e_check.py
```

The script exits with non-zero on failure and checks:
- two-request reorder improves shared prefix,
- no early/weird eviction before stress,
- eviction is observed only after vLLM cache pressure.

### 6. PageIndex + ContextPilot (Demo)

```bash
# No API key needed — uses bundled Disney earnings tree (41 nodes)
python examples/pageindex_e2e_example.py
```

To generate a tree from your own PDF, use [PageIndex](https://github.com/yinsicheng/PageIndex):

```bash
pip install pageindex
# See https://github.com/yinsicheng/PageIndex for usage
```

For the full LLM pipeline (tree search + answer generation via `PageIndexRetriever`):

```bash
pip install openai
export OPENAI_API_KEY="your-key"
python examples/pageindex_e2e_example.py --tree path/to/my_tree.json -q "What was DTC revenue?"
```

### 7. mem0 A/B Benchmark

```bash
pip install mem0ai
export OPENAI_API_KEY="your-key"

# Start inference engine + ContextPilot (live mode), then:
python examples/mem0_bench_simple.py
```

## Directory Structure

```
examples/
├── simple_reorder_example.py     # Minimal hello world
├── pipeline_examples.py          # Pipeline API (BM25, FAISS, generation)
├── http_server_example.py        # Stateful index server
├── vllm_patch_e2e_check.py       # vLLM patch e2e verifier
├── stateless_batch_example.py    # Stateless batch reordering
├── stateless_sglang_e2e.py       # Stateless → inference e2e
├── pageindex_e2e_example.py      # PageIndex + ContextPilot
├── mem0_bench_simple.py          # A/B cache hit benchmark
├── mem0_locomo_example.py        # LoCoMo multi-turn benchmark
├── offline/                      # Batch processing utilities
│   └── prepare_batch.py          # Batch optimization
├── construct_rag_data/           # Data retrieval utilities
│   ├── multihopRAG_bm25.py       # BM25 retrieval
│   └── multihopRAG_faiss.py      # FAISS retrieval
└── data/                         # Sample data
    └── disney_q1_fy25_tree.json  # 41-node PageIndex tree (Disney Q1 FY25)
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
