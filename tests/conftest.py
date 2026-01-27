"""
Pytest Configuration and Shared Fixtures.

This file contains shared fixtures and configuration for all tests.
"""

import pytest
import numpy as np
from typing import List, Dict


# ============================================================================
# Context Fixtures
# ============================================================================

@pytest.fixture
def simple_contexts() -> List[List[int]]:
    """Return simple test contexts."""
    return [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [1, 3, 5, 7, 9],
    ]


@pytest.fixture
def overlapping_contexts() -> List[List[int]]:
    """Return contexts with significant overlap for testing optimization."""
    return [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6, 7],
        [1, 2, 3, 8, 9],
        [1, 2, 10, 11, 12],
    ]


@pytest.fixture
def disjoint_contexts() -> List[List[int]]:
    """Return completely disjoint contexts."""
    return [
        [1, 2, 3],
        [10, 20, 30],
        [100, 200, 300],
    ]


@pytest.fixture
def clustered_contexts() -> List[List[int]]:
    """Return contexts that form natural clusters."""
    # Cluster 1: contexts sharing prefix [1, 2, 3]
    cluster1 = [
        [1, 2, 3, 100, 101],
        [1, 2, 3, 102, 103],
        [1, 2, 3, 104, 105],
    ]
    # Cluster 2: contexts sharing prefix [10, 20, 30]
    cluster2 = [
        [10, 20, 30, 200, 201],
        [10, 20, 30, 202, 203],
        [10, 20, 30, 204, 205],
    ]
    return cluster1 + cluster2


# ============================================================================
# Index Fixtures
# ============================================================================

@pytest.fixture
def built_context_index(simple_contexts):
    """Return a built context index."""
    from contextpilot.context_index import build_context_index
    return build_context_index(simple_contexts, use_gpu=False)


# ============================================================================
# Configuration
# ============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Skip GPU tests if no GPU available
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    
    if not has_gpu:
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ============================================================================
# Utility Functions
# ============================================================================

@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42
