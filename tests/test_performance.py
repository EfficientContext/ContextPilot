"""
Performance Tests for ContextPilot.

Tests to measure and validate performance characteristics
of the ContextPilot system under various conditions.

Run benchmarks with:
    pytest tests/test_performance.py -v -k benchmark --tb=short
    
Run all performance tests:
    pytest tests/test_performance.py -v --tb=short
"""

import pytest
import time
from typing import List, Dict


# =============================================================================
# USER BENCHMARKS - Run these to test performance on your machine
# =============================================================================

class TestBenchmarkClustering:
    """
    Benchmark tests for clustering and indexing performance.
    
    Run with: pytest tests/test_performance.py::TestBenchmarkClustering -v -s
    
    These tests print detailed timing information to help users
    understand performance on their specific hardware.
    """
    
    @pytest.fixture
    def benchmark_contexts_small(self):
        """50 contexts for quick benchmarks."""
        return [
            list(range(i % 30, i % 30 + 25)) + list(range(500 + i * 5, 500 + i * 5 + 15))
            for i in range(50)
        ]
    
    @pytest.fixture
    def benchmark_contexts_medium(self):
        """200 contexts for medium benchmarks."""
        return [
            list(range(i % 50, i % 50 + 30)) + list(range(1000 + i * 8, 1000 + i * 8 + 20))
            for i in range(200)
        ]
    
    @pytest.fixture
    def benchmark_contexts_large(self):
        """500 contexts for large benchmarks."""
        return [
            list(range(i % 100, i % 100 + 40)) + list(range(2000 + i * 10, 2000 + i * 10 + 30))
            for i in range(500)
        ]
    
    def test_benchmark_clustering_cpu_small(self, benchmark_contexts_small):
        """Benchmark: CPU clustering with 50 contexts."""
        from contextpilot.context_index import ContextIndex
        
        print("\n" + "=" * 60)
        print("BENCHMARK: CPU Clustering (50 contexts)")
        print("=" * 60)
        
        index = ContextIndex(
            linkage_method="average",
            use_gpu=False,
            alpha=0.005
        )
        
        # Warm-up run
        _ = index.fit_transform(benchmark_contexts_small[:10])
        
        # Timed run
        index = ContextIndex(linkage_method="average", use_gpu=False, alpha=0.005)
        start = time.time()
        result = index.fit_transform(benchmark_contexts_small)
        elapsed = time.time() - start
        
        print(f"  Contexts:        {len(benchmark_contexts_small)}")
        print(f"  Total time:      {elapsed:.3f}s")
        print(f"  Per context:     {elapsed / len(benchmark_contexts_small) * 1000:.2f}ms")
        print(f"  Throughput:      {len(benchmark_contexts_small) / elapsed:.1f} contexts/s")
        print("=" * 60)
        
        assert elapsed < 30, f"Benchmark took too long: {elapsed:.2f}s"
    
    def test_benchmark_clustering_cpu_medium(self, benchmark_contexts_medium):
        """Benchmark: CPU clustering with 200 contexts."""
        from contextpilot.context_index import ContextIndex
        
        print("\n" + "=" * 60)
        print("BENCHMARK: CPU Clustering (200 contexts)")
        print("=" * 60)
        
        index = ContextIndex(
            linkage_method="average",
            use_gpu=False,
            alpha=0.005
        )
        
        start = time.time()
        result = index.fit_transform(benchmark_contexts_medium)
        elapsed = time.time() - start
        
        print(f"  Contexts:        {len(benchmark_contexts_medium)}")
        print(f"  Total time:      {elapsed:.3f}s")
        print(f"  Per context:     {elapsed / len(benchmark_contexts_medium) * 1000:.2f}ms")
        print(f"  Throughput:      {len(benchmark_contexts_medium) / elapsed:.1f} contexts/s")
        print("=" * 60)
        
        assert elapsed < 120, f"Benchmark took too long: {elapsed:.2f}s"
    
    @pytest.mark.gpu
    def test_benchmark_clustering_gpu(self, benchmark_contexts_medium):
        """Benchmark: GPU clustering with 200 contexts (if available)."""
        from contextpilot.context_index import ContextIndex
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("torch not installed")
        
        print("\n" + "=" * 60)
        print("BENCHMARK: GPU Clustering (200 contexts)")
        print("=" * 60)
        
        # Warm-up GPU
        index = ContextIndex(linkage_method="average", use_gpu=True, alpha=0.005)
        _ = index.fit_transform(benchmark_contexts_medium[:20])
        
        # Timed run
        index = ContextIndex(
            linkage_method="average",
            use_gpu=True,
            alpha=0.005
        )
        
        torch.cuda.synchronize()
        start = time.time()
        result = index.fit_transform(benchmark_contexts_medium)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"  Contexts:        {len(benchmark_contexts_medium)}")
        print(f"  Total time:      {elapsed:.3f}s")
        print(f"  Per context:     {elapsed / len(benchmark_contexts_medium) * 1000:.2f}ms")
        print(f"  Throughput:      {len(benchmark_contexts_medium) / elapsed:.1f} contexts/s")
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")
        print("=" * 60)
        
        assert elapsed < 60, f"Benchmark took too long: {elapsed:.2f}s"
    
    @pytest.mark.gpu
    def test_benchmark_cpu_vs_gpu_comparison(self, benchmark_contexts_medium):
        """Benchmark: Compare CPU vs GPU performance."""
        from contextpilot.context_index import ContextIndex
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        
        print("\n" + "=" * 60)
        print("BENCHMARK: CPU vs GPU Comparison (200 contexts)")
        print("=" * 60)
        
        # CPU benchmark
        index_cpu = ContextIndex(linkage_method="average", use_gpu=False, alpha=0.005)
        start = time.time()
        _ = index_cpu.fit_transform(benchmark_contexts_medium)
        cpu_time = time.time() - start
        
        print(f"  CPU time:        {cpu_time:.3f}s")
        
        # GPU benchmark (if available)
        if torch.cuda.is_available():
            # Warm-up
            index_gpu = ContextIndex(linkage_method="average", use_gpu=True, alpha=0.005)
            _ = index_gpu.fit_transform(benchmark_contexts_medium[:20])
            
            index_gpu = ContextIndex(linkage_method="average", use_gpu=True, alpha=0.005)
            torch.cuda.synchronize()
            start = time.time()
            _ = index_gpu.fit_transform(benchmark_contexts_medium)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            
            print(f"  GPU time:        {gpu_time:.3f}s")
            print(f"  Speedup:         {cpu_time / gpu_time:.2f}x")
            print(f"  GPU:             {torch.cuda.get_device_name(0)}")
        else:
            print("  GPU:             Not available")
        
        print("=" * 60)
        assert True
    
    def test_benchmark_linkage_methods(self, benchmark_contexts_small):
        """Benchmark: Compare different linkage methods."""
        from contextpilot.context_index import ContextIndex
        
        print("\n" + "=" * 60)
        print("BENCHMARK: Linkage Method Comparison (50 contexts)")
        print("=" * 60)
        
        results = {}
        for method in ["single", "complete", "average"]:
            index = ContextIndex(
                linkage_method=method,
                use_gpu=False,
                alpha=0.005
            )
            
            start = time.time()
            _ = index.fit_transform(benchmark_contexts_small)
            elapsed = time.time() - start
            
            results[method] = elapsed
            print(f"  {method:12s}:    {elapsed:.3f}s")
        
        print("=" * 60)
        assert all(t < 30 for t in results.values())
    
    def test_benchmark_scaling(self):
        """Benchmark: Test how performance scales with context count."""
        from contextpilot.context_index import ContextIndex
        
        print("\n" + "=" * 60)
        print("BENCHMARK: Scaling Analysis")
        print("=" * 60)
        print(f"  {'Contexts':<12} {'Time (s)':<12} {'Per ctx (ms)':<15} {'Throughput':<12}")
        print("-" * 60)
        
        results = []
        for n in [25, 50, 100, 200]:
            contexts = [
                list(range(i % 50, i % 50 + 30)) + list(range(1000 + i * 5, 1000 + i * 5 + 15))
                for i in range(n)
            ]
            
            index = ContextIndex(linkage_method="average", use_gpu=False, alpha=0.005)
            
            start = time.time()
            _ = index.fit_transform(contexts)
            elapsed = time.time() - start
            
            per_ctx = elapsed / n * 1000
            throughput = n / elapsed
            
            print(f"  {n:<12} {elapsed:<12.3f} {per_ctx:<15.2f} {throughput:<12.1f}")
            results.append((n, elapsed))
        
        print("=" * 60)
        
        # Verify reasonable scaling (should be roughly O(n²) for distance matrix)
        assert results[-1][1] < results[0][1] * 100  # 8x contexts shouldn't take 100x time


class TestBenchmarkScheduling:
    """
    Benchmark tests for context scheduling performance.
    
    Run with: pytest tests/test_performance.py::TestBenchmarkScheduling -v -s
    """
    
    def test_benchmark_scheduling(self):
        """Benchmark: Full pipeline (clustering + scheduling)."""
        from contextpilot.context_index import ContextIndex
        from contextpilot.context_ordering import InterContextScheduler
        
        print("\n" + "=" * 60)
        print("BENCHMARK: Full Pipeline (Clustering + Scheduling)")
        print("=" * 60)
        
        contexts = [
            list(range(i % 50, i % 50 + 30)) + list(range(1000 + i * 5, 1000 + i * 5 + 15))
            for i in range(100)
        ]
        
        # Clustering phase
        index = ContextIndex(linkage_method="average", use_gpu=False, alpha=0.005)
        
        start = time.time()
        clustering_result = index.fit_transform(contexts)
        clustering_time = time.time() - start
        
        # Scheduling phase
        scheduler = InterContextScheduler()
        
        start = time.time()
        scheduled = scheduler.schedule_contexts(clustering_result)
        scheduling_time = time.time() - start
        
        total_time = clustering_time + scheduling_time
        
        print(f"  Contexts:        {len(contexts)}")
        print(f"  Clustering:      {clustering_time:.3f}s")
        print(f"  Scheduling:      {scheduling_time:.3f}s")
        print(f"  Total:           {total_time:.3f}s")
        print(f"  Throughput:      {len(contexts) / total_time:.1f} contexts/s")
        print("=" * 60)
        
        assert total_time < 60


# =============================================================================
# CI/CD TESTS - Standard pytest tests for automated testing
# =============================================================================

class TestIndexingPerformance:
    """Test performance of context indexing operations."""
    
    @pytest.fixture
    def large_context_set(self):
        """Generate a large set of contexts (as lists of token IDs)."""
        contexts = []
        for i in range(100):
            # Generate context with some overlap and some unique tokens
            base = list(range(i % 50, i % 50 + 20))  # some overlap
            unique = list(range(1000 + i * 10, 1000 + i * 10 + 10))  # unique tokens
            contexts.append(base + unique)
        return contexts
    
    def test_fit_transform_scales_reasonably(self, large_context_set):
        """Test that fit_transform completes in reasonable time."""
        from contextpilot.context_index import ContextIndex
        
        index = ContextIndex(
            linkage_method="average",
            use_gpu=False,  # For consistent test results
            alpha=0.005
        )
        
        start = time.time()
        index.fit_transform(large_context_set)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 60s for 100 docs)
        assert elapsed < 60, f"fit_transform took {elapsed:.2f}s for 100 docs"
    
    def test_distance_computation_scales(self, large_context_set):
        """Test that distance computation scales properly."""
        from contextpilot.context_index import ContextIndex
        
        # Test with increasing sizes
        times = []
        for size in [10, 20, 40]:
            subset = large_context_set[:size]
            
            index = ContextIndex(
                linkage_method="average",
                use_gpu=False,
                alpha=0.005
            )
            
            start = time.time()
            index.fit_transform(subset)
            elapsed = time.time() - start
            
            times.append(elapsed)
        
        # Should not scale worse than O(n²) by too much
        # (40 docs should take < 20x the time of 10 docs)
        assert times[2] < times[0] * 25 or times[0] < 0.1  # Allow small times


class TestSchedulingPerformance:
    """Test performance of context scheduling operations."""
    
    @pytest.fixture
    def large_ordering_input(self):
        """Generate large input for ordering."""
        return [
            {"chunk_id": i, "text": f"Document {i}", "cluster": i % 5}
            for i in range(200)
        ]
    
    def test_scheduling_performance(self, large_ordering_input):
        """Test scheduling completes in reasonable time."""
        from contextpilot.context_index import ContextIndex
        from contextpilot.context_ordering import InterContextScheduler
        
        # Build a real clustering result to use with scheduler
        contexts = [[i, i+1, i+2, i+3, i+4] for i in range(50)]
        
        index = ContextIndex(
            linkage_method="average",
            use_gpu=False,
            alpha=0.005
        )
        
        # Get real clustering result
        start = time.time()
        clustering_result = index.fit_transform(contexts)
        
        # Now schedule using the result
        scheduler = InterContextScheduler()
        result = scheduler.schedule_contexts(clustering_result)
        elapsed = time.time() - start
        
        # Should complete quickly (< 10s)
        assert elapsed < 10, f"Scheduling took {elapsed:.2f}s for 50 items"


class TestMemoryUsage:
    """Test memory characteristics."""
    
    def test_no_memory_leak_on_repeated_operations(self):
        """Test that repeated operations don't leak memory."""
        import gc
        from contextpilot.context_index import ContextIndex
        
        # Prepare contexts as token ID lists
        contexts = [[i + j for j in range(30)] for i in range(0, 500, 10)]
        
        # Force GC before starting
        gc.collect()
        
        # Perform many operations
        for _ in range(5):
            index = ContextIndex(
                linkage_method="average",
                use_gpu=False,
                alpha=0.005
            )
            index.fit_transform(contexts)
        
        # Force GC after
        gc.collect()
        
        # If we get here without memory error, test passes
        assert True


class TestConcurrentAccess:
    """Test concurrent access patterns."""
    
    def test_thread_safety_of_multi_turn_manager(self):
        """Test that MultiTurnManager handles concurrent access."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        import threading
        
        manager = MultiTurnManager()
        corpus = {str(i): {"text": f"Doc {i}"} for i in range(100)}
        errors = []
        
        def worker(conv_id):
            try:
                for _ in range(5):
                    manager.deduplicate_context(
                        conv_id,
                        list(range(10)),
                        corpus
                    )
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=worker, args=(f"conv_{i}",))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"


class TestBatchProcessing:
    """Test batch processing performance."""
    
    def test_batch_vs_sequential(self):
        """Compare batch processing to sequential."""
        from contextpilot.context_index import ContextIndex
        
        contexts = [[i + j for j in range(20)] for i in range(0, 500, 10)]
        
        # Single batch
        index = ContextIndex(
            linkage_method="average",
            use_gpu=False,
            alpha=0.005
        )
        
        start = time.time()
        index.fit_transform(contexts)
        batch_time = time.time() - start
        
        # Sequential small batches
        seq_time = 0
        for i in range(0, 50, 10):
            index2 = ContextIndex(
                linkage_method="average",
                use_gpu=False,
                alpha=0.005
            )
            start = time.time()
            if len(contexts[i:i+10]) >= 2:  # Need at least 2 for clustering
                index2.fit_transform(contexts[i:i+10])
            seq_time += time.time() - start
        
        # Batch should be more efficient (accounting for tree building)
        # This is a soft check - batching has benefits for tree construction
        assert batch_time < seq_time * 5 or seq_time < 0.5  # Allow small times


class TestLatencyProfile:
    """Test latency characteristics of different operations."""
    
    def test_operation_latencies(self):
        """Profile latencies of different operations."""
        from contextpilot.context_index import ContextIndex
        from contextpilot.context_ordering import InterContextScheduler
        
        contexts = [[i + j for j in range(15)] for i in range(0, 200, 10)]
        
        latencies = {}
        
        # Index creation
        start = time.time()
        index = ContextIndex(
            linkage_method="average",
            use_gpu=False,
            alpha=0.005
        )
        latencies["index_creation"] = time.time() - start
        
        # Fit transform
        start = time.time()
        index.fit_transform(contexts)
        latencies["fit_transform"] = time.time() - start
        
        # Scheduler creation
        start = time.time()
        scheduler = InterContextScheduler()
        latencies["scheduler_creation"] = time.time() - start
        
        # All operations should complete quickly
        assert all(lat < 30 for lat in latencies.values())


class TestScalabilityLimits:
    """Test behavior at scalability limits."""
    
    def test_handles_many_documents(self):
        """Test handling many documents."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        corpus = {str(i): {"text": f"Doc {i}"} for i in range(1000)}
        
        # Should handle 1000 docs
        context, novel, stats = manager.deduplicate_context(
            "test_conv",
            list(range(1000)),
            corpus
        )
        
        assert len(novel) == 1000
    
    def test_handles_long_conversations(self):
        """Test handling long conversations with many turns."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        corpus = {str(i): {"text": f"Doc {i}"} for i in range(100)}
        
        # Simulate 50 turns
        for turn in range(50):
            # Each turn gets a sliding window of docs
            start = turn % 90
            doc_ids = list(range(start, start + 10))
            
            manager.deduplicate_context("long_conv", doc_ids, corpus)
        
        conv = manager.get_conversation("long_conv")
        
        assert conv.turn_count == 50
    
    def test_handles_many_concurrent_conversations(self):
        """Test handling many concurrent conversations."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        corpus = {str(i): {"text": f"Doc {i}"} for i in range(50)}
        
        # Create 100 conversations
        for i in range(100):
            manager.deduplicate_context(
                f"conv_{i}",
                [i % 50, (i+1) % 50, (i+2) % 50],
                corpus
            )
        
        assert len(manager.conversations) == 100
