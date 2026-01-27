"""
Test for verifying that all members in a group share a common prefix.

This test validates the InterContextScheduler's grouping mechanism by:
1. Creating synthetic test cases with known prefix patterns
2. Building context index and scheduling contexts
3. Verifying that all members within each group share a common prefix
"""

import pytest
from typing import List, Tuple

from contextpilot.context_index import build_context_index
from contextpilot.context_ordering.inter_scheduler import InterContextScheduler


def find_common_prefix(sequences: List[List[int]]) -> List[int]:
    """
    Find the longest common prefix among a list of sequences.
    
    Args:
        sequences: List of sequences (each is a list of integers)
        
    Returns:
        The longest common prefix
    """
    if not sequences:
        return []
    
    if len(sequences) == 1:
        return sequences[0]
    
    # Find minimum length
    min_len = min(len(seq) for seq in sequences)
    
    # Find common prefix
    common = []
    for i in range(min_len):
        val = sequences[0][i]
        if all(seq[i] == val for seq in sequences):
            common.append(val)
        else:
            break
    
    return common


def verify_group_prefix_sharing(
    groups: List[Tuple[float, List[int]]],
    contexts: List[List[int]],
) -> Tuple[bool, List[dict]]:
    """
    Verify that all members in each group share a common prefix.
    
    Args:
        groups: List of (score, group_indices) tuples
        contexts: List of all contexts (reordered)
        
    Returns:
        Tuple of (all_passed, issues_list)
    """
    issues = []
    
    for group_id, (score, group_indices) in enumerate(groups):
        if len(group_indices) <= 1:
            # Single-item groups trivially share prefix with themselves
            continue
        
        # Get contexts for all members in this group
        group_contexts = [contexts[idx] for idx in group_indices]
        
        # Find common prefix
        common_prefix = find_common_prefix(group_contexts)
        
        # Check if there is a common prefix
        if len(common_prefix) == 0:
            issues.append({
                'group_id': group_id,
                'size': len(group_indices),
                'score': score,
                'query_indices': group_indices,
                'contexts': group_contexts,
                'common_prefix_length': 0
            })
    
    return len(issues) == 0, issues


def get_final_contexts_from_clustering(clustering_result, num_contexts):
    """Extract final contexts after scheduler node reordering."""
    final_contexts = []
    for i in range(num_contexts):
        for node_id, node in clustering_result.unique_nodes.items():
            if node.is_leaf and i in node.original_indices:
                final_contexts.append(node.doc_ids)
                break
    return final_contexts


class TestGroupPrefixSharing:
    """Tests for verifying group prefix sharing in context scheduling."""
    
    def test_simple_synthetic_case(self):
        """
        Test with a simple synthetic case where we know the expected grouping.
        All contexts have the same length (simulating top-k retrieval in RAG).
        """
        # Create synthetic contexts with known prefix patterns
        # All contexts have 5 documents (simulating top-5 retrieval)
        contexts = [
            [1, 2, 3, 4, 5],    # Group A
            [1, 2, 3, 6, 7],    # Group A
            [1, 2, 4, 8, 9],    # Group A
            [3, 4, 5, 6, 7],    # Group B
            [3, 4, 5, 8, 9],    # Group B
            [3, 4, 6, 10, 11],  # Group B
            [5, 6, 7, 8, 9],    # Group C
            [5, 6, 7, 12, 13],  # Group C
        ]
        
        # Build context index
        clustering_result = build_context_index(contexts, use_gpu=False, alpha=0.005)
        
        # Schedule contexts
        scheduler = InterContextScheduler()
        scheduled_reordered, scheduled_originals, index_mapping, groups = scheduler.schedule_contexts(
            clustering_result
        )
        
        # Get final contexts after scheduler node reordering
        final_contexts = get_final_contexts_from_clustering(clustering_result, len(contexts))
        
        # Verify prefix sharing
        all_passed, issues = verify_group_prefix_sharing(groups, final_contexts)
        
        assert all_passed, f"Groups without common prefix: {issues}"
    
    def test_edge_cases(self):
        """
        Test edge cases with same-length contexts but different prefix patterns.
        """
        # Create edge case contexts - all with same length (5 docs)
        contexts = [
            [1, 5, 6, 7, 8],        # Starts with 1
            [1, 2, 10, 11, 12],     # Starts with 1, 2
            [1, 2, 3, 13, 14],      # Starts with 1, 2, 3
            [2, 15, 16, 17, 18],    # Starts with 2
            [2, 3, 19, 20, 21],     # Starts with 2, 3
            [2, 3, 4, 22, 23],      # Starts with 2, 3, 4
            [1, 2, 3, 4, 24],       # Starts with 1, 2, 3, 4
        ]
        
        # Build context index
        clustering_result = build_context_index(contexts, use_gpu=False, alpha=0.005)
        
        # Schedule contexts
        scheduler = InterContextScheduler()
        scheduled_reordered, scheduled_originals, index_mapping, groups = scheduler.schedule_contexts(
            clustering_result
        )
        
        # Get final contexts after scheduler node reordering
        final_contexts = get_final_contexts_from_clustering(clustering_result, len(contexts))
        
        # Verify prefix sharing
        all_passed, issues = verify_group_prefix_sharing(groups, final_contexts)
        
        assert all_passed, f"Groups without common prefix: {issues}"
    
    @pytest.mark.slow
    def test_real_dataset_sample(self):
        """
        Test with a sample from real dataset (if available).
        """
        import json
        import os
        
        dataset_path = '/home/jysc/Demnok/datasets/multihopRAG/mulhoprag_new_queries_top15.jsonl'
        
        if not os.path.exists(dataset_path):
            pytest.skip(f"Dataset not found at {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            queries = [json.loads(line) for line in f][:50]  # Use first 50 queries
        
        contexts = [q['top_k_doc_id'] for q in queries]
        
        # Build context index
        clustering_result = build_context_index(contexts, use_gpu=False, alpha=0.005)
        
        # Schedule contexts
        scheduler = InterContextScheduler()
        scheduled_reordered, scheduled_originals, index_mapping, groups = scheduler.schedule_contexts(
            clustering_result
        )
        
        # Get final contexts after scheduler node reordering
        final_contexts = get_final_contexts_from_clustering(clustering_result, len(contexts))
        
        # Verify prefix sharing
        all_passed, issues = verify_group_prefix_sharing(groups, final_contexts)
        
        assert all_passed, f"{len(issues)} groups don't share common prefix"
