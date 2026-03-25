
# Token Proxy Integration Guide (Reuse Plugins)

This guide explains how to integrate the **ContextReorder** and **ContextDedup** plugins into the Token Proxy Middleware.

## Overview

The plugins are designed to sit between the Agent Framework (OpenClaw) and the LLM Server (SGLang). They process standard OpenAI-formatted JSON payloads to maximize KV cache hits.

## Installation

Ensure the `contextpilot` core logic is installed in your environment:
```bash
pip install -e .
```

## Sequential Usage Example

Here is how to use both plugins in a sequential pipeline (e.g., inside a FastAPI endpoint).

```python
import asyncio
from refactored_plugins.reorder import ContextReorderPlugin
from refactored_plugins.dedup import ContextDedupPlugin

# 1. Initialize Plugins
# Tip: Use the same model name as your SGLang backend for accurate tokenization
reorder_plugin = ContextReorderPlugin(model_name="Qwen/Qwen2.5-7B-Instruct")
dedup_plugin = ContextDedupPlugin()

async def proxy_middleware_endpoint(request_batch: list):
    """
    Example middleware logic for a batch of requests.
    """
    
    # --- STAGE 1: DEDUPLICATION ---
    # Process each request in the batch to remove redundant history
    compressed_batch = []
    for req in request_batch:
        optimized_req = await dedup_plugin.process(req)
        compressed_batch.append(optimized_req)
        
    # --- STAGE 2: REORDERING ---
    # Reorder the whole batch to maximize prefix sharing in the KV Cache
    final_batch = await reorder_plugin.process(compressed_batch)
    
    # --- TELEMETRY ---
    print(f"Reorder Metrics: {reorder_plugin.get_plugin_metrics()}")
    print(f"Dedup Metrics:   {dedup_plugin.get_plugin_metrics()}")
    
    return final_batch

# Mock Batch for Testing
mock_request = {
    "user_id": "user_1",
    "messages": [
        {"role": "system", "content": "Tool: Math. Tool: Code."},
        {"role": "user", "content": "Calculate 2+2"}
    ]
}

# Run the pipeline
if __name__ == "__main__":
    final = asyncio.run(proxy_middleware_endpoint([mock_request]))
    print("Optimization Complete.")
```

## Plugin Specifics

### ContextReorderPlugin
- **Input**: `List[Dict]` (Batch of OpenAI requests).
- **Output**: `List[Dict]` (Reordered batch).
- **Logic**: Clusters requests by content overlap and schedules the execution sequence to ensure adjacent requests share the longest possible prefix.

### ContextDedupPlugin
- **Input**: `Dict` (Single OpenAI request).
- **Output**: `Dict` (Modified request with history replaced by reference hints).
- **Logic**: Tracks conversation state via `user_id` and `parent_id`. Replaces previously seen messages with strings like `[Reference to Turn 1]`.
