# Performance Benchmarking

This guide explains how to benchmark ContextPilot performance on your machine.

## Quick Start

Run the benchmark script from the repository root:

```bash
python scripts/benchmark.py
```

## Benchmark Options

### Basic Usage

```bash
# Run all benchmarks (recommended for first run)
python scripts/benchmark.py

# Quick benchmark with smaller context sizes
python scripts/benchmark.py --quick

# Include GPU benchmarks
python scripts/benchmark.py --gpu

# CPU only (skip GPU detection)
python scripts/benchmark.py --cpu-only
```

### Custom Context Sizes

Test with specific context counts:

```bash
# Small scale test
python scripts/benchmark.py --sizes 25 50 100

# Large scale test
python scripts/benchmark.py --sizes 100 500 1000 2000

# Single size
python scripts/benchmark.py --sizes 500
```

## What Gets Benchmarked

### 1. Scaling Analysis

Tests how performance scales with increasing context counts:

```
======================================================================
  SCALING ANALYSIS
======================================================================
  Contexts     Time (s)     Per ctx (ms)    Throughput  
----------------------------------------------------------------------
  50           0.328        6.56            152.4       
  100          0.445        4.45            224.5       
  200          0.528        2.64            378.8       
  400          1.234        3.09            324.1       
```

### 2. CPU Clustering

Benchmarks the core clustering algorithm on CPU:

```
======================================================================
  CPU CLUSTERING (200 contexts)
======================================================================
  Contexts:            200
  Total time:          0.528s
  Per context:         2.64ms
  Throughput:          378.8 contexts/s
```

### 3. GPU Clustering (Optional)

When using `--gpu`, benchmarks GPU-accelerated distance computation:

```
======================================================================
  GPU CLUSTERING (200 contexts)
======================================================================
  Contexts:            200
  Total time:          0.156s
  Per context:         0.78ms
  Throughput:          1282.1 contexts/s
  GPU:                 NVIDIA A100-SXM4-80GB
```

### 4. CPU vs GPU Comparison

Direct comparison when GPU is available:

```
======================================================================
  CPU vs GPU COMPARISON (200 contexts)
======================================================================
  CPU time:            0.528s
  GPU time:            0.156s
  Speedup:             3.38x
  GPU:                 NVIDIA A100-SXM4-80GB
```

### 5. Linkage Methods

Compares different hierarchical clustering linkage methods:

```
======================================================================
  LINKAGE METHOD COMPARISON (50 contexts)
======================================================================
  single:              0.312s
  complete:            0.318s
  average:             0.328s
```

### 6. Full Pipeline

End-to-end benchmark including clustering and scheduling:

```
======================================================================
  FULL PIPELINE (200 contexts)
======================================================================
  Contexts:            200
  Clustering:          0.528s
  Scheduling:          0.001s
  Total:               0.529s
  Throughput:          378.1 contexts/s
```

## Using pytest (Alternative)

If you prefer pytest-style benchmarks:

```bash
# Run benchmark tests with output
pytest tests/test_performance.py -v -s -k benchmark

# Specific benchmark
pytest tests/test_performance.py::TestBenchmarkClustering::test_benchmark_scaling -v -s
```

**Note:** The `-s` flag is required to see benchmark output with pytest.

## Interpreting Results

### Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Throughput** | Contexts processed per second | >100 contexts/s |
| **Per context** | Time per context in milliseconds | <10ms |
| **GPU Speedup** | GPU time / CPU time | >2x |

### Expected Performance

Performance varies by hardware. Typical ranges:

| Hardware | 100 contexts | 500 contexts |
|----------|--------------|--------------|
| Modern CPU (8+ cores) | 0.3-0.5s | 2-5s |
| GPU (RTX 3090+) | 0.1-0.2s | 0.5-1s |

### Scaling Behavior

- **Distance computation**: O(n²) where n = number of contexts
- **Clustering**: O(n² log n) due to hierarchical clustering
- **Scheduling**: O(n log n)

For very large context sets (>1000), GPU acceleration is recommended.

## Troubleshooting

### "GPU not available"

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

If False, ensure:
- NVIDIA drivers are installed
- PyTorch is installed with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Import Errors

```bash
# Make sure ContextPilot is installed
pip install -e .
```

### Slow Performance

1. **Check CPU utilization**: ContextPilot uses multiprocessing for CPU distance computation
2. **Try GPU**: Use `--gpu` flag if GPU is available
3. **Reduce batch size**: For memory-constrained systems

## Programmatic Benchmarking

You can also benchmark programmatically:

```python
import time
from contextpilot.context_index import ContextIndex
from contextpilot.context_ordering import InterContextScheduler

# Prepare contexts (lists of token IDs)
contexts = [
    list(range(i, i + 30)) + list(range(1000 + i, 1000 + i + 15))
    for i in range(200)
]

# Benchmark clustering
index = ContextIndex(
    linkage_method="average",
    use_gpu=True,  # or False for CPU
    alpha=0.005
)

start = time.time()
result = index.fit_transform(contexts)
clustering_time = time.time() - start

# Benchmark scheduling
scheduler = InterContextScheduler()

start = time.time()
scheduled = scheduler.schedule_contexts(result)
scheduling_time = time.time() - start

print(f"Clustering: {clustering_time:.3f}s")
print(f"Scheduling: {scheduling_time:.3f}s")
print(f"Total: {clustering_time + scheduling_time:.3f}s")
```
