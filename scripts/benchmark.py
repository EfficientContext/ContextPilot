#!/usr/bin/env python3
"""
ContextPilot Performance Benchmark

Run this script to benchmark clustering and indexing performance on your machine.

Usage:
    python benchmark.py              # Run all benchmarks
    python benchmark.py --quick      # Quick benchmark (smaller sizes)
    python benchmark.py --gpu        # Include GPU benchmarks
    python benchmark.py --sizes 50 100 200  # Custom context sizes
"""

import argparse
import time
import sys


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(label: str, value: str):
    """Print a formatted result line."""
    print(f"  {label:<20} {value}")


def generate_contexts(n: int):
    """Generate n synthetic contexts (lists of token IDs)."""
    contexts = []
    for i in range(n):
        # Create contexts with some overlap for realistic clustering
        base = list(range(i % 50, i % 50 + 30))
        unique = list(range(1000 + i * 5, 1000 + i * 5 + 15))
        contexts.append(base + unique)
    return contexts


def benchmark_clustering_cpu(contexts, linkage_method="average"):
    """Benchmark CPU clustering."""
    from contextpilot.context_index import ContextIndex
    
    index = ContextIndex(
        linkage_method=linkage_method,
        use_gpu=False,
        alpha=0.005
    )
    
    start = time.time()
    result = index.fit_transform(contexts)
    elapsed = time.time() - start
    
    return elapsed, result


def benchmark_clustering_gpu(contexts, linkage_method="average"):
    """Benchmark GPU clustering."""
    import torch
    from contextpilot.context_index import ContextIndex
    
    if not torch.cuda.is_available():
        return None, None
    
    # Warm-up
    index = ContextIndex(linkage_method=linkage_method, use_gpu=True, alpha=0.005)
    _ = index.fit_transform(contexts[:min(20, len(contexts))])
    
    index = ContextIndex(
        linkage_method=linkage_method,
        use_gpu=True,
        alpha=0.005
    )
    
    torch.cuda.synchronize()
    start = time.time()
    result = index.fit_transform(contexts)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed, result


def benchmark_scheduling(clustering_result):
    """Benchmark context scheduling."""
    from contextpilot.context_ordering import InterContextScheduler
    
    scheduler = InterContextScheduler()
    
    start = time.time()
    result = scheduler.schedule_contexts(clustering_result)
    elapsed = time.time() - start
    
    return elapsed, result


def run_scaling_benchmark(sizes, use_gpu=False):
    """Run scaling benchmark across different context sizes."""
    print_header("SCALING ANALYSIS")
    print(f"  {'Contexts':<12} {'Time (s)':<12} {'Per ctx (ms)':<15} {'Throughput':<12}")
    print("-" * 70)
    
    results = []
    for n in sizes:
        contexts = generate_contexts(n)
        
        elapsed, _ = benchmark_clustering_cpu(contexts)
        per_ctx = elapsed / n * 1000
        throughput = n / elapsed
        
        print(f"  {n:<12} {elapsed:<12.3f} {per_ctx:<15.2f} {throughput:<12.1f}")
        results.append((n, elapsed, per_ctx, throughput))
    
    return results


def run_cpu_benchmark(n_contexts):
    """Run CPU clustering benchmark."""
    print_header(f"CPU CLUSTERING ({n_contexts} contexts)")
    
    contexts = generate_contexts(n_contexts)
    
    elapsed, result = benchmark_clustering_cpu(contexts)
    
    print_result("Contexts:", str(n_contexts))
    print_result("Total time:", f"{elapsed:.3f}s")
    print_result("Per context:", f"{elapsed / n_contexts * 1000:.2f}ms")
    print_result("Throughput:", f"{n_contexts / elapsed:.1f} contexts/s")
    
    return elapsed


def run_gpu_benchmark(n_contexts):
    """Run GPU clustering benchmark."""
    import torch
    
    if not torch.cuda.is_available():
        print_header(f"GPU CLUSTERING ({n_contexts} contexts)")
        print("  GPU not available - skipping")
        return None
    
    print_header(f"GPU CLUSTERING ({n_contexts} contexts)")
    
    contexts = generate_contexts(n_contexts)
    
    elapsed, result = benchmark_clustering_gpu(contexts)
    
    print_result("Contexts:", str(n_contexts))
    print_result("Total time:", f"{elapsed:.3f}s")
    print_result("Per context:", f"{elapsed / n_contexts * 1000:.2f}ms")
    print_result("Throughput:", f"{n_contexts / elapsed:.1f} contexts/s")
    print_result("GPU:", torch.cuda.get_device_name(0))
    
    return elapsed


def run_cpu_vs_gpu_benchmark(n_contexts):
    """Compare CPU vs GPU performance."""
    import torch
    
    print_header(f"CPU vs GPU COMPARISON ({n_contexts} contexts)")
    
    contexts = generate_contexts(n_contexts)
    
    # CPU
    cpu_time, _ = benchmark_clustering_cpu(contexts)
    print_result("CPU time:", f"{cpu_time:.3f}s")
    
    # GPU
    if torch.cuda.is_available():
        gpu_time, _ = benchmark_clustering_gpu(contexts)
        print_result("GPU time:", f"{gpu_time:.3f}s")
        print_result("Speedup:", f"{cpu_time / gpu_time:.2f}x")
        print_result("GPU:", torch.cuda.get_device_name(0))
    else:
        print_result("GPU:", "Not available")


def run_linkage_benchmark(n_contexts):
    """Compare different linkage methods."""
    print_header(f"LINKAGE METHOD COMPARISON ({n_contexts} contexts)")
    
    contexts = generate_contexts(n_contexts)
    
    for method in ["single", "complete", "average"]:
        elapsed, _ = benchmark_clustering_cpu(contexts, linkage_method=method)
        print_result(f"{method}:", f"{elapsed:.3f}s")


def run_full_pipeline_benchmark(n_contexts):
    """Benchmark full pipeline (clustering + scheduling)."""
    print_header(f"FULL PIPELINE ({n_contexts} contexts)")
    
    contexts = generate_contexts(n_contexts)
    
    # Clustering
    clustering_time, clustering_result = benchmark_clustering_cpu(contexts)
    
    # Scheduling
    scheduling_time, _ = benchmark_scheduling(clustering_result)
    
    total_time = clustering_time + scheduling_time
    
    print_result("Contexts:", str(n_contexts))
    print_result("Clustering:", f"{clustering_time:.3f}s")
    print_result("Scheduling:", f"{scheduling_time:.3f}s")
    print_result("Total:", f"{total_time:.3f}s")
    print_result("Throughput:", f"{n_contexts / total_time:.1f} contexts/s")


def main():
    parser = argparse.ArgumentParser(
        description="ContextPilot Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark.py              # Run all benchmarks
    python benchmark.py --quick      # Quick benchmark
    python benchmark.py --gpu        # Include GPU benchmarks
    python benchmark.py --sizes 50 100 200 400  # Custom sizes
        """
    )
    parser.add_argument("--quick", action="store_true", 
                        help="Run quick benchmark with smaller sizes")
    parser.add_argument("--gpu", action="store_true",
                        help="Include GPU benchmarks")
    parser.add_argument("--sizes", type=int, nargs="+",
                        help="Custom context sizes for scaling benchmark")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Skip GPU benchmarks")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  ContextPilot Performance Benchmark")
    print("=" * 70)
    
    # Determine sizes
    if args.sizes:
        sizes = args.sizes
    elif args.quick:
        sizes = [25, 50, 100]
    else:
        sizes = [50, 100, 200, 400]
    
    default_size = sizes[-1] if len(sizes) > 0 else 100
    
    try:
        # Import check
        print("\nChecking imports...")
        from contextpilot.context_index import ContextIndex
        from contextpilot.context_ordering import InterContextScheduler
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        # Run benchmarks
        run_scaling_benchmark(sizes)
        
        run_cpu_benchmark(default_size)
        
        if args.gpu and not args.cpu_only:
            run_gpu_benchmark(default_size)
            run_cpu_vs_gpu_benchmark(default_size)
        
        run_linkage_benchmark(min(50, default_size))
        
        run_full_pipeline_benchmark(default_size)
        
        print("\n" + "=" * 70)
        print("  Benchmark Complete!")
        print("=" * 70 + "\n")
        
    except ImportError as e:
        print(f"\nError: Could not import ContextPilot modules: {e}")
        print("Make sure ContextPilot is installed: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
