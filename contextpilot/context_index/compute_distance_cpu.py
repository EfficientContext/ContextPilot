"""
Optimized CPU distance matrix computation using sorted data structures
and merge algorithm (similar approach to GPU version).
"""

import numpy as np
import time
from multiprocessing import Pool, cpu_count
from typing import List


def compute_distance_single(context_a: List[int], context_b: List[int], alpha: float = 0.005) -> float:
    """
    Compute distance between two contexts using our metric:
    distance = (1 - overlap/max_size) + alpha * avg_position_diff
    
    Args:
        context_a: First context (list of doc IDs)
        context_b: Second context (list of doc IDs)
        alpha: Weight for position difference term
        
    Returns:
        Distance value (lower = more similar)
    """
    if not context_a or not context_b:
        return 1.0
    
    # Build position maps
    pos_a = {doc: pos for pos, doc in enumerate(context_a)}
    pos_b = {doc: pos for pos, doc in enumerate(context_b)}
    
    # Find intersection
    intersection = set(pos_a.keys()) & set(pos_b.keys())
    intersection_size = len(intersection)
    
    if intersection_size == 0:
        return 1.0
    
    # Overlap term: 1 - |intersection| / max(|A|, |B|)
    max_size = max(len(context_a), len(context_b))
    overlap_term = 1.0 - (intersection_size / max_size)
    
    # Position term: alpha * avg(|pos_A - pos_B|)
    position_diff_sum = sum(abs(pos_a[doc] - pos_b[doc]) for doc in intersection)
    position_term = alpha * (position_diff_sum / intersection_size)
    
    return overlap_term + position_term


def compute_distances_batch(queries: List[List[int]], 
                           targets: List[List[int]], 
                           alpha: float = 0.005,
                           num_workers: int = None) -> np.ndarray:
    """
    Compute distances from multiple query contexts to multiple target contexts.
    Returns a (num_queries x num_targets) distance matrix.
    
    This is optimized for the search use case where we have:
    - queries: new contexts to search for
    - targets: existing node doc_ids in the tree
    
    Args:
        queries: List of query contexts
        targets: List of target contexts (e.g., node doc_ids)
        alpha: Weight for position difference term
        num_workers: Number of parallel workers (default: all CPU cores)
        
    Returns:
        Distance matrix of shape (len(queries), len(targets))
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Cap at 8 for this use case
    
    n_queries = len(queries)
    n_targets = len(targets)
    
    if n_queries == 0 or n_targets == 0:
        return np.zeros((n_queries, n_targets), dtype=np.float32)
    
    # For small inputs, just compute directly (avoid multiprocessing overhead)
    if n_queries * n_targets < 1000:
        distances = np.ones((n_queries, n_targets), dtype=np.float32)
        for i, query in enumerate(queries):
            for j, target in enumerate(targets):
                distances[i, j] = compute_distance_single(query, target, alpha)
        return distances
    
    # Prepare all pairs to compute
    pairs = [(i, j) for i in range(n_queries) for j in range(n_targets)]
    
    # Split into batches
    batch_size = max(100, len(pairs) // (num_workers * 4))
    batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
    
    # Worker function
    def worker(batch):
        results = []
        for i, j in batch:
            dist = compute_distance_single(queries[i], targets[j], alpha)
            results.append((i, j, dist))
        return results
    
    # Compute in parallel
    distances = np.ones((n_queries, n_targets), dtype=np.float32)
    
    with Pool(num_workers) as pool:
        for batch_results in pool.imap_unordered(worker, batches):
            for i, j, dist in batch_results:
                distances[i, j] = dist
    
    return distances


def prepare_contexts_for_cpu(contexts):
    """Prepare contexts using same strategy as GPU - sorting and flattening."""
    n = len(contexts)
    
    sorted_data = []
    lengths = np.zeros(n, dtype=np.int32)
    
    print("Preparing data structures...")
    for idx, ctx in enumerate(contexts):
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx+1:,}/{n:,} contexts...")
        
        if len(ctx) == 0:
            sorted_data.append([])
            lengths[idx] = 0
            continue
        
        # Create (chunk_id, original_position) pairs
        pairs = [(chunk_id, orig_pos) for orig_pos, chunk_id in enumerate(ctx)]
        pairs.sort(key=lambda x: x[0])
        
        sorted_data.append(pairs)
        lengths[idx] = len(pairs)
    
    # Create offset array
    offsets = np.zeros(n + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths)
    
    # Flatten into contiguous arrays
    total_elements = offsets[-1]
    chunk_ids = np.zeros(total_elements, dtype=np.int32)
    original_positions = np.zeros(total_elements, dtype=np.int32)
    
    for i, pairs in enumerate(sorted_data):
        start = offsets[i]
        end = offsets[i + 1]
        for j, (chunk_id, orig_pos) in enumerate(pairs):
            chunk_ids[start + j] = chunk_id
            original_positions[start + j] = orig_pos
    
    print(f"Total elements: {total_elements:,}")
    print(f"Memory: {(chunk_ids.nbytes + original_positions.nbytes + lengths.nbytes + offsets.nbytes) / 1e6:.1f} MB")
    
    return chunk_ids, original_positions, lengths, offsets


def compute_distance_optimized(chunk_ids, original_positions, lengths, offsets, i, j, alpha):
    """
    Optimized distance computation using merge algorithm (same as GPU).
    Much faster than dictionary/set approach.
    
    Time complexity: O(n) where n is context length
    Space complexity: O(1) - no additional data structures
    """
    len_i = lengths[i]
    len_j = lengths[j]
    
    if len_i == 0 or len_j == 0:
        return 1.0
    
    # Get pointers to chunk data
    offset_i = offsets[i]
    offset_j = offsets[j]
    
    chunks_i = chunk_ids[offset_i:offset_i + len_i]
    pos_i = original_positions[offset_i:offset_i + len_i]
    chunks_j = chunk_ids[offset_j:offset_j + len_j]
    pos_j = original_positions[offset_j:offset_j + len_j]
    
    # Merge-style intersection (O(n) instead of O(n log n) with sets)
    intersection_size = 0
    position_diff_sum = 0
    
    pi = 0
    pj = 0
    
    while pi < len_i and pj < len_j:
        chunk_i = chunks_i[pi]
        chunk_j = chunks_j[pj]
        
        if chunk_i == chunk_j:
            # Found match
            intersection_size += 1
            position_diff_sum += abs(pos_i[pi] - pos_j[pj])
            pi += 1
            pj += 1
        elif chunk_i < chunk_j:
            pi += 1
        else:
            pj += 1
    
    # Compute distance
    max_size = max(len_i, len_j)
    overlap_term = 1.0 - (intersection_size / max_size)
    
    if intersection_size == 0:
        position_term = 0.0
    else:
        avg_pos_diff = position_diff_sum / intersection_size
        position_term = alpha * avg_pos_diff
    
    return overlap_term + position_term


def compute_batch_worker(args):
    """
    Worker that processes a batch of pairs (more efficient than single pairs).
    Batch processing improves cache locality and reduces IPC overhead.
    """
    batch_indices, chunk_ids, original_positions, lengths, offsets, alpha = args
    
    results = []
    for i, j in batch_indices:
        dist = compute_distance_optimized(
            chunk_ids, original_positions, lengths, offsets, i, j, alpha
        )
        results.append((i, j, dist))
    
    return results


def compute_distance_matrix_cpu_optimized(contexts: List[List[int]], 
                                          alpha: float = 0.005,
                                          num_workers: int = None,
                                          batch_size: int = 1000) -> np.ndarray:
    """
    Optimized CPU distance matrix computation using:
    1. Sorted data structures (like GPU)
    2. Merge algorithm for intersection (O(n) vs O(n log n))
    3. NumPy arrays instead of Python dicts/sets
    4. Batch processing for better cache performance
    
    Args:
        contexts: List of context lists (each context is a list of chunk IDs)
        alpha: Weight for position term in distance calculation
        num_workers: Number of parallel workers (default: all CPU cores)
        batch_size: Number of pairs per batch (default: 1000)
    
    Returns:
        Condensed distance matrix (upper triangle) for scipy.hierarchy.linkage()
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    n = len(contexts)
    num_pairs = n * (n - 1) // 2
    
    print(f"\n{'='*70}")
    print(f"Optimized CPU Distance Matrix Computation")
    print(f"{'='*70}")
    print(f"Contexts: {n:,}")
    print(f"Pairs to compute: {num_pairs:,}")
    print(f"Workers (CPU cores): {num_workers}")
    print(f"Batch size: {batch_size:,}")
    print(f"Alpha: {alpha}")
    
    # Prepare data (same as GPU)
    print(f"\nPreparing data...")
    start = time.time()
    chunk_ids, original_positions, lengths, offsets = prepare_contexts_for_cpu(contexts)
    prep_time = time.time() - start
    print(f"✓ Prepared in {prep_time:.1f}s")
    
    # Generate batches of pair indices
    print(f"\nGenerating pair batches...")
    batches = []
    current_batch = []
    
    for i in range(n):
        for j in range(i + 1, n):
            current_batch.append((i, j))
            
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
    
    if current_batch:
        batches.append(current_batch)
    
    print(f"✓ Generated {len(batches):,} batches")
    
    # Prepare arguments for workers
    worker_args = [
        (batch, chunk_ids, original_positions, lengths, offsets, alpha)
        for batch in batches
    ]
    
    # Compute distances in parallel
    print(f"\nComputing distances with {num_workers} workers...")
    
    condensed_distances = np.zeros(num_pairs, dtype=np.float32)
    
    start_time = time.time()
    processed = 0
    
    with Pool(num_workers) as pool:
        for batch_results in pool.imap_unordered(compute_batch_worker, worker_args):
            for i, j, dist in batch_results:
                # Convert (i, j) to condensed index
                condensed_idx = n * i - i * (i + 1) // 2 + j - i - 1
                condensed_distances[condensed_idx] = dist
                
                processed += 1
                
                # Progress update
                if processed % 100000 == 0 or processed == num_pairs:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (num_pairs - processed) / rate if rate > 0 else 0
                    progress_pct = processed / num_pairs * 100
                    
                    print(f"  {processed:,}/{num_pairs:,} ({progress_pct:.1f}%) | "
                          f"Rate: {rate:,.0f} pairs/sec | "
                          f"Elapsed: {elapsed:.1f}s | "
                          f"ETA: {eta:.1f}s ({eta/60:.1f} min)")
    
    compute_time = time.time() - start_time
    total_time = compute_time + prep_time
    
    print(f"\n{'='*70}")
    print(f"COMPLETE!")
    print(f"{'='*70}")
    print(f"Prep time: {prep_time:.1f}s")
    print(f"Compute time: {compute_time:.1f}s")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Avg rate: {num_pairs/compute_time:,.0f} pairs/second")
    
    return condensed_distances