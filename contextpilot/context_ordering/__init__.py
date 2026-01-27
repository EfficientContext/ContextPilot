"""
Context Ordering Module

This module provides comprehensive context reordering functionality for optimizing
batch processing of contexts/queries. It includes both intra-context ordering
(reordering elements within individual contexts) and inter-context ordering
(reordering contexts relative to each other).

The module is organized into two main components:

1. **Intra-Context Ordering**: Reorders elements within individual contexts to
   maximize prefix sharing based on hierarchical clustering tree analysis.

2. **Inter-Context Ordering**: Reorders contexts relative to each other using
   tree-based grouping and optimization to maximize cache efficiency.
"""

# Intra-context ordering (within individual contexts)
from .intra_ordering import IntraContextOrderer

# Inter-context ordering (between contexts using tree-based approach)
from .inter_scheduler import InterContextScheduler

__all__ = [
    # Intra-context ordering (within contexts)
    'IntraContextOrderer',
    
    # Inter-context ordering (between contexts)
    'InterContextScheduler',
]
