# mem0 + ContextPilot LoCoMo Benchmark

This example measures TTFT and answer accuracy (token-F1, LLM judge) with and without ContextPilot context reordering, using mem0 as the memory backend and SGLang for inference.

* [Mem0](https://github.com/mem0ai/mem0) is an intelligent memory layer that facilitates memory storage and retrieval for agents.
* [Locomo](https://github.com/snap-research/locomo) is a long conversation benchmark used to test memory retrieval. 

![mem0_locomo_diagram](../images/mem0_locomo.png)

## Setup

```bash
pip install -r requirements.txt
pip install mem0ai openai tqdm
pip install "sglang[all]==0.5.6"
bash patches/sglang/apply_patch.sh
```

## Start servers

```bash
python -m contextpilot.server.http_server --port 8765
```

In a separate terminal:

```bash
export RAGBOOST_INDEX_URL=http://localhost:8765
python -m sglang.launch_server --model <model> --port 30000
```

## Run

```bash
export OPENAI_API_KEY=<your API key>
python examples/mem0_locomo_example.py
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `SGLANG_URL` | `http://localhost:30000` | SGLang endpoint |
| `CONTEXTPILOT_URL` | `http://localhost:8765` | ContextPilot server endpoint |
| `JUDGE_MODEL` | `gpt-4.1-2025-04-14` | OpenAI model for LLM judge |
| `LOCOMO_CONV_INDEX` | `0` | Which LoCoMo conversation to use |
| `LOCOMO_MAX_QA` | `150` | Max QA pairs to evaluate |
| `LOCOMO_MAX_TOKENS` | `32` | Max generation tokens |
| `LOCOMO_NUM_TURNS` | `150` | Multi-turn conversation length |
| `LOCOMO_TOP_K_LIST` | `20,100` | Comma-separated top-k values to benchmark |

## Results

LoCoMo conv 0, 100 memories, 150 turns:

| k | mode | ttft | judge |
|---|---|---|---|
| 20 | baseline | 0.0397s | 0.440 |
| 20 | reorder | 0.0334s | 0.457 |
| 100 | baseline | 0.1002s | 0.437 |
| 100 | reorder | 0.0823s | 0.452 |

## General usage

### Store and retrieve memories

```python
from contextpilot.retriever import Mem0Retriever

retriever = Mem0Retriever(config={
    "llm": {"provider": "openai", "config": {"model": "gpt-4.1-mini-2025-04-14"}},
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
})

retriever.add_memory(
    [{"role": "user", "content": "I'm allergic to peanuts"},
     {"role": "assistant", "content": "Noted."}],
    user_id="user123",
)

results = retriever.search_queries(
    query_data=[{"text": "dietary restrictions?"}],
    user_id="user123", top_k=20,
)
corpus_map = retriever.get_corpus_map()
```

### Reorder with the library

```python
from contextpilot.context_index import build_context_index
from contextpilot.context_ordering import InterContextScheduler

contexts = [r["top_k_doc_id"] for r in results]
index_result = build_context_index(contexts)
reordered, _, order, _ = InterContextScheduler().schedule_contexts(index_result)
```

### Reorder via the server (enables KV-cache tracking)

```python
import requests

requests.post("http://localhost:8765/reset")
resp = requests.post("http://localhost:8765/build", json={
    "contexts": contexts,
    "use_gpu": False,
    "linkage_method": "average",
    "alpha": 0.005,
}).json()

reordered = resp["reordered_contexts"]  # reordered doc ID lists
```

### Multi-turn

Just call `/build` each turn — ContextPilot auto-detects whether the index exists and uses incremental mode accordingly:

```python
for turn, query in enumerate(queries):
    results = retriever.search_queries(
        query_data=[{"text": query}], user_id="user123", top_k=20)
    resp = requests.post("http://localhost:8765/build", json={
        "contexts": [results[0]["top_k_doc_id"]],
        "use_gpu": False,
        "linkage_method": "average",
        "alpha": 0.005,
        "incremental": turn > 0,
    }).json()
    reordered_ids = resp["reordered_contexts"][0]
```
