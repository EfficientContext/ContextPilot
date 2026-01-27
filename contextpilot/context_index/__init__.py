"""
Context Index Module

Provides hierarchical clustering-based context indexing for efficient
context reordering and similarity computation.
"""

# Import tree node management
from .tree_nodes import (
    ClusterNode,
    NodeManager
)

# Import index construction
from .index_construction import (
    ContextIndex,
    IndexResult,
    build_context_index
)

# Import distance computation
from .compute_distance_cpu import (
    compute_distance_matrix_cpu_optimized,
    prepare_contexts_for_cpu
)

try:
    from .compute_distance_gpu import (
        compute_distance_matrix_gpu,
        prepare_contexts_for_gpu,
        get_gpu_info
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    # Main classes
    'ContextIndex',
    'IndexResult',
    
    # Tree node management
    'ClusterNode',
    'NodeManager',
    
    # Convenience functions
    'build_context_index',
    
    # Distance computation
    'compute_distance_matrix_cpu_optimized',
    'prepare_contexts_for_cpu',
    
    # GPU utilities (if available)
    'GPU_AVAILABLE',
]

if GPU_AVAILABLE:
    __all__.extend([
        'compute_distance_matrix_gpu',
        'prepare_contexts_for_gpu',
        'get_gpu_info'
    ])
