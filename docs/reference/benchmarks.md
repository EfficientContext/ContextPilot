# Benchmarks

Performance benchmarks for ContextPilot.

## GPU vs CPU Performance

ContextPilot supports both GPU and CPU for distance computation in context index construction.

### Test Configuration

- **GPU**: NVIDIA A6000
- **CPU**: AMD EPYC 7313P 16-Core
- **Metric**: Time to compute pairwise distances for context clustering

### Results

| Contexts | GPU Time (s) | CPU Time (s) | Speedup |
|----------|--------------|--------------|---------|
| 64 | 0.22 ± 0.30 | 0.20 ± 0.00 | 0.89x |
| 128 | 0.02 ± 0.00 | 0.28 ± 0.00 | **17.40x** |
| 512 | 0.05 ± 0.00 | 1.02 ± 0.01 | **20.51x** |
| 4,000 | 0.89 ± 0.05 | 52.02 ± 0.06 | **58.43x** |
| 8,000 | 3.19 ± 0.22 | 211.45 ± 1.12 | **66.27x** |
| 12,000 | 6.77 ± 0.45 | 490.91 ± 7.98 | **72.48x** |
| 100,000 | 687.64 ± 0.02 | N/A | N/A |

### Key Findings

- GPU performance advantage **scales with problem size**
- At 64 contexts, CPU is slightly faster (0.89x) due to GPU overhead
- Crossover point: **~100-128 contexts**
- At 12k contexts: **72x speedup** with GPU

---

## Deployment Recommendations

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| < 128 contexts | **CPU** | GPU overhead exceeds computation benefit |
| ≥ 128 contexts | **GPU** | 17-72x speedup for batch processing |
| Production workloads | **GPU** | Critical for high-throughput requirements |
| Development/testing | **CPU** | Simpler setup, no GPU dependency |

---

## End-to-End Performance

When integrated with SGLang or vLLM:

| Metric | Improvement |
|--------|-------------|
| Cache hit rate | **4-13x** |
| Prefill latency | **1.5-3.5x** reduction |
| Accuracy | Maintained or improved |

See the [main README](../../README.md) for accuracy benchmarks on MT-RAG.

---

## Running Your Own Benchmarks

```bash
# GPU vs CPU distance computation
python tests/test_gpu_distance_performance.py

# Full benchmark suite (scaling, clustering, scheduling)
python scripts/benchmark.py

# Quick benchmark with smaller sizes
python scripts/benchmark.py --quick

# Include GPU benchmarks
python scripts/benchmark.py --gpu

# Custom context sizes
python scripts/benchmark.py --sizes 100 500 1000 2000
```
