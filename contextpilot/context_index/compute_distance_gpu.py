#!/usr/bin/env python3
"""
GPU-accelerated distance matrix computation for context similarity.

This module computes pairwise distances between contexts using CUDA,
considering both overlap and positional alignment of chunks.
"""

import numpy as np
import cupy as cp
import time
from tqdm import tqdm


def get_gpu_info(device_id=0):
    """Get GPU device information."""
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    return {
        'name': props['name'].decode('utf-8'),
        'compute_capability': f"{props['major']}.{props['minor']}",
        'total_memory': props['totalGlobalMem']
    }


# CUDA kernel for distance computation
distance_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_distances(
    const int* __restrict__ chunk_ids,
    const int* __restrict__ original_pos,
    const int* __restrict__ lengths,
    const int* __restrict__ offsets,
    float* __restrict__ distances,
    const int n,
    const int batch_start,
    const int batch_end,
    const float alpha
) {
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Calculate total pairs in this batch
    int total_pairs = 0;
    for (int row = batch_start; row < batch_end; row++) {
        total_pairs += (n - row - 1);
    }
    
    if (global_idx >= total_pairs) return;
    
    // Convert linear index to (i, j) pair
    int i = batch_start;
    int remaining = global_idx;
    
    while (i < batch_end) {
        int pairs_in_row = n - i - 1;
        if (remaining < pairs_in_row) {
            break;
        }
        remaining -= pairs_in_row;
        i++;
    }
    
    int j = i + 1 + remaining;
    
    int len_i = lengths[i];
    int len_j = lengths[j];
    
    if (len_i == 0 || len_j == 0) {
        distances[global_idx] = 1.0f;
        return;
    }
    
    int offset_i = offsets[i];
    int offset_j = offsets[j];
    
    const int* chunks_i = chunk_ids + offset_i;
    const int* pos_i = original_pos + offset_i;
    const int* chunks_j = chunk_ids + offset_j;
    const int* pos_j = original_pos + offset_j;
    
    int intersection_size = 0;
    long long position_diff_sum = 0;
    
    // Merge-style intersection
    int pi = 0, pj = 0;
    
    while (pi < len_i && pj < len_j) {
        int chunk_i = chunks_i[pi];
        int chunk_j = chunks_j[pj];
        
        if (chunk_i == chunk_j) {
            intersection_size++;
            
            int orig_pos_i = pos_i[pi];
            int orig_pos_j = pos_j[pj];
            
            long long diff = (long long)orig_pos_i - (long long)orig_pos_j;
            if (diff < 0) diff = -diff;
            
            position_diff_sum += diff;
            
            pi++;
            pj++;
        } else if (chunk_i < chunk_j) {
            pi++;
        } else {
            pj++;
        }
    }
    
    // Compute distance
    int max_size = (len_i > len_j) ? len_i : len_j;
    float overlap_term = 1.0f - ((float)intersection_size / (float)max_size);
    
    float position_term = 0.0f;
    if (intersection_size > 0) {
        float avg_pos_diff = (float)position_diff_sum / (float)intersection_size;
        position_term = alpha * avg_pos_diff;
    }
    
    distances[global_idx] = overlap_term + position_term;
}
''', 'compute_distances')


def prepare_contexts_for_gpu(contexts):
    """Prepare contexts for GPU computation."""
    n = len(contexts)
    
    sorted_data = []
    lengths = np.zeros(n, dtype=np.int32)
    
    for idx, ctx in enumerate(contexts):
        if len(ctx) == 0:
            sorted_data.append([])
            lengths[idx] = 0
            continue
            
        # Create (chunk_id, original_position) pairs and sort by chunk_id
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
    
    return chunk_ids, original_positions, lengths, offsets


def compute_distance_matrix_gpu(contexts, alpha=0.001, batch_rows=5000):
    """Compute distance matrix on GPU."""
    n = len(contexts)
    num_pairs = n * (n - 1) // 2
    
    print(f"\n{'='*70}")
    print(f"GPU Distance Matrix Computation")
    print(f"{'='*70}")
    print(f"Contexts: {n:,}")
    print(f"Pairs: {num_pairs:,}")
    print(f"Alpha: {alpha}")
    
    # Prepare data
    print(f"\nPreparing data...")
    start = time.time()
    chunk_ids, original_positions, lengths, offsets = prepare_contexts_for_gpu(contexts)
    prep_time = time.time() - start
    print(f"Prepared in {prep_time:.1f}s")
    
    # Transfer to GPU
    print(f"\nTransferring to GPU...")
    start = time.time()
    d_chunk_ids = cp.asarray(chunk_ids)
    d_original_positions = cp.asarray(original_positions)
    d_lengths = cp.asarray(lengths)
    d_offsets = cp.asarray(offsets)
    transfer_time = time.time() - start
    print(f"Transferred in {transfer_time:.1f}s")
    
    # Allocate output
    d_all_distances = cp.zeros(num_pairs, dtype=cp.float32)
    
    # Compute
    print(f"\nComputing distances...")
    total_compute_time = 0
    current_output_idx = 0
    num_batches = (n + batch_rows - 1) // batch_rows
    
    threads_per_block = 256
    
    for batch_idx in tqdm(range(num_batches), desc="Batches", ncols=80):
        batch_start = batch_idx * batch_rows
        batch_end = min(batch_start + batch_rows, n)
        
        batch_pairs = sum(n - i - 1 for i in range(batch_start, batch_end))
        if batch_pairs == 0:
            continue
        
        d_batch_output = d_all_distances[current_output_idx:current_output_idx + batch_pairs]
        blocks = (batch_pairs + threads_per_block - 1) // threads_per_block
        
        start = time.time()
        distance_kernel(
            (blocks,), (threads_per_block,),
            (d_chunk_ids, d_original_positions, d_lengths, d_offsets,
             d_batch_output, n, batch_start, batch_end, np.float32(alpha))
        )
        
        if batch_idx % 10 == 9 or batch_idx == num_batches - 1:
            cp.cuda.Stream.null.synchronize()
        
        total_compute_time += time.time() - start
        current_output_idx += batch_pairs
    
    cp.cuda.Stream.null.synchronize()
    
    # Transfer back
    print(f"\nTransferring results...")
    transfer_start = time.time()
    condensed_distances = cp.asnumpy(d_all_distances)
    transfer_back_time = time.time() - transfer_start
    print(f"Transferred in {transfer_back_time:.1f}s")
    
    total_time = prep_time + transfer_time + total_compute_time + transfer_back_time
    
    print(f"\n{'='*70}")
    print(f"COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"GPU rate: {num_pairs/total_compute_time:,.0f} pairs/second")
    
    del d_chunk_ids, d_original_positions, d_lengths, d_offsets, d_all_distances
    cp.get_default_memory_pool().free_all_blocks()
    
    return condensed_distances