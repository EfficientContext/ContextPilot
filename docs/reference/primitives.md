# Supported Primitives

ContextPilot transforms context blocks before inference using **optimization primitives**. Each primitive is a standalone operation that can be applied independently or composed. More primitives will be added over time.

## Reorder

Reorders context blocks so that shared blocks across requests align into a common prefix, enabling the inference engine to reuse cached KV states instead of recomputing them.

Two levels of optimization:

- **Inter-context scheduling** — groups requests with overlapping blocks to run consecutively, maximizing prefix sharing across the batch.
- **Intra-context reordering** — moves shared blocks to the front of each individual request so the engine finds the longest reusable prefix.

To preserve answer quality, ContextPilot injects an importance ranking into the prompt so the model still prioritizes blocks in their original relevance order.

**API:** `cp_instance.reorder()`, `cp_instance.optimize()`, `cp_instance.optimize_batch()` — see [API Reference](api.md).

## Deduplicate

Tracks which context blocks have already been sent in previous turns of a conversation. On subsequent turns, blocks that the model has already seen are replaced with lightweight reference hints (e.g., *"Please refer to [Doc A] from the previous context"*), so only genuinely new content is transmitted in full.

This reduces redundant token transmission by ~36% per turn in typical multi-turn workloads.

**API:** `cp_instance.deduplicate()` — see [API Reference](api.md).
