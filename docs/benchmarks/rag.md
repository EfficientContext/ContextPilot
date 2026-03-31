# RAG Benchmark Results

## Qwen3-32B on 4×A6000

Single-node academic RAG with a 32B model on consumer GPUs.

| Benchmark | Method | Prefill TP (tok/s) | Cache Hit | F1 (%) |
|-----------|--------|--------------------|-----------|--------|
| MultihopRAG | SGLang | 7,290 | 4.64% | 60.42 |
|              | **SGLang + ContextPilot** | **14,214** | **33.97%** | **64.39** |
| NarrativeQA | SGLang | 7,921 | 5.91% | 28.41 |
|              | **SGLang + ContextPilot** | **12,117** | **20.82%** | **29.64** |

## DeepSeek-R1-671B on 16×H20

Production-scale 671B MoE inference on a multi-node GPU cluster.

| Benchmark | Method | Prefill TP (tok/s) | Cache Hit | F1 (%) |
|-----------|--------|--------------------|-----------|--------|
| MultihopRAG | SGLang | 9,636 | 5.12% | 64.15 |
|            | **SGLang + ContextPilot** | **17,498** | **60.37%** | **64.68** |
| NarrativeQA | SGLang | 8,687 | 6.08% | 40.20 |
|            | **SGLang + ContextPilot** | **13,201** | **38.24%** | **41.08** |

For methodology and full results, see the [paper](https://arxiv.org/abs/2511.03475).
