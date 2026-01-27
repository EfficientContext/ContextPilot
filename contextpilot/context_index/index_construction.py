"""
Context Index Construction Module

This module provides hierarchical clustering-based indexing for context reordering.
It builds an efficient context index structure using distance computation and
hierarchical clustering algorithms.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage
from typing import List, Optional

# Import tree node management
from .tree_nodes import ClusterNode, NodeManager

# Import context ordering from the new context_ordering module
from contextpilot.context_ordering import IntraContextOrderer

# Import the distance computation modules
try:
    from .compute_distance_gpu import compute_distance_matrix_gpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    
from .compute_distance_cpu import compute_distance_matrix_cpu_optimized


class IndexResult:
    """Container for index construction results."""
    
    def __init__(self, linkage_matrix, cluster_nodes, unique_nodes, 
                 reordered_contexts, original_contexts, stats, search_paths=None):
        self.linkage_matrix = linkage_matrix
        self.cluster_nodes = cluster_nodes
        self.unique_nodes = unique_nodes
        self.reordered_contexts = reordered_contexts
        self.original_contexts = original_contexts
        self.stats = stats
        self.search_paths = search_paths  # List of search paths (child indices)
        
        # Legacy attributes for backward compatibility
        self.reordered_prompts = reordered_contexts
        self.original_prompts = original_contexts
    
    def print_tree(self):
        """Print the unique cluster tree."""
        print("\n--- Unique Cluster Tree Nodes ---")
        for node_id in sorted(self.unique_nodes.keys()):
            node = self.unique_nodes[node_id]
            print(node)
            print(f"  Content: {node.doc_ids}")
            print(f"  Original indices: {sorted(node.original_indices)}")
            if node.search_path:
                path_str = "[" + "][".join(map(str, node.search_path)) + "]"
                print(f"  Search path (child indices from root): {path_str}")
            else:
                print(f"  Search path: (root node)")
            if not node.is_leaf:
                print(f"  Children: {node.children}")
                print(f"  Merge distance: {node.merge_distance:.4f}")
            print("-" * 40)


class ContextIndex:
    """
    Main class for building context index using hierarchical clustering.
    
    This class provides the primary interface for constructing a context index
    from a set of contexts/contexts, using hierarchical clustering with configurable
    distance computation (GPU or CPU) and similarity methods.
    """
    
    def __init__(self, 
                 linkage_method: str = "average",
                 use_gpu: bool = True,
                 alpha: float = 0.005,
                 num_workers: Optional[int] = None,
                 batch_size: int = 1000):
        """
        Initialize the ContextIndex.
        
        Args:
            linkage_method: Linkage method for hierarchical clustering ("average", "complete", "single")
            use_gpu: Whether to use GPU for distance computation (if available)
            alpha: Weight for position term in distance calculation
            num_workers: Number of parallel workers for CPU computation (default: all cores)
            batch_size: Batch size for distance computation
        """
        self.linkage_method = linkage_method
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.alpha = alpha
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        self.node_manager = NodeManager()
        self.context_orderer = IntraContextOrderer()
        
        if self.use_gpu:
            print("Using GPU for distance computation")
        else:
            if not GPU_AVAILABLE:
                print("GPU not available, using CPU")
            else:
                print("Using CPU for distance computation")
    
    def fit_transform(self, contexts: List[List[int]]) -> IndexResult:
        """
        Perform clustering and return results.
        
        Args:
            contexts: List of contexts, where each prompt is a list of chunk IDs
            
        Returns:
            IndexResult object containing clustering results
        """
        n = len(contexts)
        
        if n < 2:
            if n == 0:
                print("No contexts provided, returning empty index.")
            else:
                print(f"Only {n} context(s) provided, skipping clustering.")
            return self._handle_single_prompt(contexts)
        
        # Compute distance matrix
        print("Computing distance matrix...")
        condensed_distances = self._compute_distance_matrix(contexts)
        
        # Perform hierarchical clustering
        print("Performing hierarchical clustering...")
        linkage_matrix = linkage(condensed_distances, method=self.linkage_method)
        
        # Build tree structure
        print("Building tree structure...")
        self._build_tree(contexts, linkage_matrix)
        
        # Clean up tree
        self.node_manager.cleanup_empty_nodes()
        
        # Update search paths for all nodes
        print("Computing search paths...")
        self.node_manager.update_search_paths()
        
        # Generate reordered contexts
        print("Generating reordered contexts...")
        reordered_contexts = self.context_orderer.reorder_contexts(
            contexts, self.node_manager.unique_nodes
        )
        
        # Extract search paths after reordering
        search_paths = self.context_orderer.extract_search_paths(
            self.node_manager.unique_nodes, len(contexts)
        )
        
        # Print statistics
        stats = self.node_manager.get_node_stats()
        print(f"Clustering completed: {stats['total_nodes']} unique nodes, "
              f"{stats['leaf_nodes']} leaf nodes")
        
        return IndexResult(
            linkage_matrix=linkage_matrix,
            cluster_nodes=self.node_manager.cluster_nodes,
            unique_nodes=self.node_manager.unique_nodes,
            reordered_contexts=reordered_contexts,
            original_contexts=contexts,
            stats=stats,
            search_paths=search_paths
        )
    
    def _compute_distance_matrix(self, contexts: List[List[int]]) -> np.ndarray:
        """Compute condensed distance matrix for clustering."""
        if self.use_gpu:
            try:
                return compute_distance_matrix_gpu(
                    contexts, 
                    alpha=self.alpha,
                    batch_rows=self.batch_size
                )
            except Exception as e:
                print(f"GPU computation failed ({e}), falling back to CPU")
                return compute_distance_matrix_cpu_optimized(
                    contexts,
                    alpha=self.alpha,
                    num_workers=self.num_workers,
                    batch_size=self.batch_size
                )
        else:
            return compute_distance_matrix_cpu_optimized(
                contexts,
                alpha=self.alpha,
                num_workers=self.num_workers,
                batch_size=self.batch_size
            )
    
    def _handle_single_prompt(self, contexts: List[List[int]]) -> IndexResult:
        """Handle case with less than 2 contexts."""
        for i, prompt in enumerate(contexts):
            self.node_manager.create_leaf_node(i, prompt)
        
        # Update search paths even for single nodes
        self.node_manager.update_search_paths()
        
        # For single context, extract search paths (will be empty for root-only tree)
        search_paths = self.context_orderer.extract_search_paths(
            self.node_manager.unique_nodes, len(contexts)
        )
        
        return IndexResult(
            linkage_matrix=np.empty((0, 4)),
            cluster_nodes=self.node_manager.cluster_nodes,
            unique_nodes=self.node_manager.unique_nodes,
            reordered_contexts=contexts.copy() if hasattr(contexts, 'copy') else list(contexts),
            original_contexts=contexts,
            stats=self.node_manager.get_node_stats(),
            search_paths=search_paths
        )
    
    def _build_tree(self, contexts: List[List[int]], linkage_matrix: np.ndarray):
        """Build the clustering tree from linkage matrix."""
        n = len(contexts)
        
        # Create leaf nodes
        for i, prompt in enumerate(contexts):
            self.node_manager.create_leaf_node(i, prompt)
        
        # Process internal nodes
        for i, (idx1, idx2, distance, _) in enumerate(linkage_matrix):
            new_node_id = n + i
            self.node_manager.create_internal_node(
                new_node_id, int(idx1), int(idx2), distance
            )


# Convenience function for backward compatibility
def build_context_index(contexts: List[List[int]], 
                       linkage_method: str = "average",
                       use_gpu: bool = True,
                       alpha: float = 0.005,
                       num_workers: Optional[int] = None,
                       batch_size: int = 1000) -> IndexResult:
    """
    Convenience function for building a context index.
    
    Args:
        contexts: List of contexts, where each prompt is a list of chunk IDs
        linkage_method: Linkage method for hierarchical clustering
        use_gpu: Whether to use GPU for distance computation
        alpha: Weight for position term in distance calculation
        num_workers: Number of parallel workers for CPU computation
        batch_size: Batch size for distance computation
        
    Returns:
        IndexResult object containing clustering results
    """
    indexer = ContextIndex(
        linkage_method=linkage_method,
        use_gpu=use_gpu,
        alpha=alpha,
        num_workers=num_workers,
        batch_size=batch_size
    )
    return indexer.fit_transform(contexts)
