#!/usr/bin/env python3
"""
Simplest ContextPilot Reordering Example

Usage:
1. Start server: python -m contextpilot.server.http_server --port 8765 --stateless
2. Run: python examples/simple_reorder_example.py
"""

import requests

# Each query's retrieved doc IDs (shuffled order)
contexts = [
    [5, 1, 3, 2],      # Query 0: contains {1, 2, 3}
    [10, 11, 12],      # Query 1: completely different
    [2, 3, 1, 4],      # Query 2: also contains {1, 2, 3} (overlaps with 0)
    [12, 10, 11],      # Query 3: (overlaps with 1)
]

# Call ContextPilot
response = requests.post("http://localhost:8765/reorder", json={
    "contexts": contexts
})
result = response.json()

# Get results
original_indices = result["original_indices"]        # Execution order
reordered_contexts = result["reordered_contexts"]    # Reordered doc IDs

print(f"Execution order: {original_indices}")
print(f"Number of groups: {result['num_groups']}")

print("\nReordered contexts:")
for i, ctx in enumerate(reordered_contexts):
    orig_idx = original_indices[i]
    print(f"  Position {i}: Query {orig_idx} -> {ctx}")
