#!/usr/bin/env python3
"""
Simplest ContextPilot Scheduling Example

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
response = requests.post("http://localhost:8765/schedule", json={
    "contexts": contexts
})
result = response.json()

# Get results
scheduled_order = result["original_indices"]      # Scheduled query order
scheduled_contexts = result["scheduled_contexts"]  # Reordered doc IDs

print(f"Scheduled order: {scheduled_order}")
print(f"Number of groups: {result['num_groups']}")

print("\nReordered contexts:")
for i, ctx in enumerate(scheduled_contexts):
    orig_idx = scheduled_order[i]
    print(f"  Position {i}: Query {orig_idx} -> {ctx}")
