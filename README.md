<div align="center">
  <img src="assets/about.png" alt="ContextPilot Logo" width="800"/>

  <h1><strong>ContextPilot: Efficient Long Context Inference with Context Reuse</strong></h1>

  [![arXiv](https://img.shields.io/badge/arXiv-2511.03475-b31b1b.svg)](https://arxiv.org/abs/2511.03475)
  [![Python](https://img.shields.io/badge/python-≥3.10-blue)](https://www.python.org/)
  [![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

</div>

--------------------------------------------------------------------------------

| [**Documentation**](docs/README.md) | [**Examples**](examples/) | [**Benchmarks**](docs/reference/benchmarks.md) |

## News

- [2026/01] ContextPilot has been accepted to MLSys 2026 🎉! See you in Bellevue, WA, USA.
- [2026/01] Code is released! 

## About

ContextPilot is a fast optimization system on context engineering layer for agentic workloads:
1. **High Throughput**: Boosting prefill throughput with intelligent context reuse.
2. **Accuracy Preserved**: Reasoning accuracy is fully preserved and even enhanced!
3. **Strong Compatibility**: Strong compatibility with existing popular RAG libraries (PageIndex), Agentic memory layer (Mem0), KV cache optimization engine (LMCache), and Inference engines (vLLM and SGLang). Both single-node and multi-node deployment!
4. **Widely Tested**: Tested with a wide range of RAG and Agentic AI applications.

## Target Workloads

1. **Trending Topic QA with Retrieval** — Search and generation for breaking news and hot topics beyond model knowledge
2. **Closed-Domain Long-Context QA** — Retrieval-augmented QA over specialized corpora (novels, financial reports, legal documents)
3. **Multi-Turn Conversations with Long-Term Memory** — Persistent context across sessions (e.g. [Mem0](https://github.com/mem0ai/mem0))

## Benchmark and Performance

### System Performance

![Benchmark Results](assets/deepseek_r1_results.png)

ContextPilot on DeepSeek-R1 maintains accuracy compared to SGLang, achieving 64.68% vs 64.15% F1 on MultihopRAG and 41.08% vs 40.20% F1 on NarrativeQA.

### Accuracy on MT-RAG Benchmark

| Method | Qwen3-4B | Llama3.1-8B | Qwen3-30B-A3B |
|--------|----------|-------------|-----------|
| LMCache | 62.56 | **68.46** | 75.12 |
| CacheBlend | 50.33 | 56.52 | X |
| RadixCache | 62.56 | **68.46** | 75.12 |
| **ContextPilot** | **64.27** | 68.12 | **75.81** |

ContextPilot delivers **4-13x** improvements in cache hit rates and **1.5-3.5x** reductions in prefill latency for large-batch RAG workloads, while maintaining or improving accuracy.

**Furthermore**, ContextPilot has been tested to reduce input token costs by around **36%** with GPT-5.2.

See [Benchmarks](docs/reference/benchmarks.md) in the documentation for GPU vs CPU performance analysis and detailed benchmark methodology.

## Getting Started

### Installation

**Requirements:** Python >= 3.10

```bash
pip install contextpilot
```

Or install from source:
```bash
git clone https://github.com/SecretSettler/ContextPilot.git
cd ContextPilot
pip install -e .
```

Install an inference engine (SGLang recommended):
```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang" --prerelease=allow
```

More [detailed installation instructions](docs/getting_started/installation.md) are available in the docs, including Docker setup and FAISS configuration.

### PageIndex Integration (NEW!)

ContextPilot now supports [PageIndex](https://github.com/VectifyAI/PageIndex), a **reasoning-based, vectorless RAG** system. PageIndex uses LLM reasoning over hierarchical document tree structures instead of vector similarity search:

```python
from contextpilot.retriever import PageIndexRetriever
from contextpilot import RAGPipeline, RetrieverConfig, OptimizerConfig

# Option 1: Use PageIndexRetriever directly
retriever = PageIndexRetriever(model="gpt-4o")
retriever.load_tree_structures(["document_structure.json"])
results = retriever.search_queries(query_data=[{"question": "What is the revenue?"}])

# Option 2: Use unified RAGPipeline
pipeline = RAGPipeline(
    retriever=RetrieverConfig(
        retriever_type="pageindex",
        pageindex_model="gpt-4o",
        pageindex_tree_paths=["document_structure.json"],
        top_k=5
    ),
    optimizer=OptimizerConfig(enabled=True),
    use_contextpilot=True
)
pipeline.setup()
```

See [examples/pageindex_example.py](examples/pageindex_example.py) for detailed usage.

## Documentation

Check out the ContextPilot [documentation](docs/README.md) for comprehensive guides.

## Examples

Go hands-on with our [examples](examples/), demonstrating how to address different use cases with ContextPilot.

## Contributing

We welcome and value all contributions! Please feel free to submit issues and pull requests.

## Citation

If you use the code or data of ContextPilot, please declare the reference with the following:

```bibtex
@misc{jiang2025contextpilot,
      title={ContextPilot: Efficient Retrieval-Augmented Generation with Accuracy-Preserving Context Reuse}, 
      author={Yinsicheng Jiang and Yeqi Huang and Liang Cheng and Cheng Deng and Xuan Sun and Luo Mai},
      year={2025},
      eprint={2511.03475},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.03475}, 
}
```
