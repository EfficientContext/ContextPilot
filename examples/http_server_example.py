"""
Example: ContextPilot HTTP Server with Inference Backend Integration

This shows how to use the ContextPilot HTTP server for online inference
with automatic KV cache management.

SETUP:
1. Start an inference engine with ContextPilot patch:
   # SGLang:
   CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m sglang.launch_server \
       --model-path Qwen/Qwen2.5-7B-Instruct --port 30000
   # or vLLM:
   CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-7B-Instruct --port 30000 --enable-prefix-caching

2. Start ContextPilot server:
   python -m contextpilot.server.http_server --port 8765 --infer-api-url http://localhost:30000

3. Run this example:
   python examples/http_server_example.py
"""

import requests
import json


BASE_URL = "http://localhost:8765"


def check_server():
    """Check if ContextPilot server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2.0)
        health = response.json()
        print(f"✓ Server status: {health['status']}")
        return health
    except Exception as e:
        print(f"✗ Server not running: {e}")
        print("\nPlease start the servers first:")
        print("  # Terminal 1: Start inference engine with ContextPilot patch")
        print("  # SGLang:")
        print("  CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m sglang.launch_server \\")
        print("      --model-path Qwen/Qwen2.5-7B-Instruct --port 30000")
        print("  # or vLLM:")
        print("  CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m vllm.entrypoints.openai.api_server \\")
        print("      --model Qwen/Qwen2.5-7B-Instruct --port 30000 --enable-prefix-caching")
        print()
        print("  # Terminal 2: Start ContextPilot server")
        print("  python -m contextpilot.server.http_server --port 8765 --infer-api-url http://localhost:30000")
        return None


def build_index():
    """
    Build the ContextPilot index with your contexts.
    
    Each context is a list of document/chunk IDs that will be used together.
    ContextPilot clusters similar contexts and returns request_ids for tracking.
    """
    # Example contexts (each is a list of document IDs for one query)
    contexts = [
        [1, 5, 10, 15, 20],     # Query 1 uses docs 1, 5, 10, 15, 20
        [2, 5, 11, 16, 21],     # Query 2 uses docs 2, 5, 11, 16, 21
        [1, 5, 12, 17, 22],     # Query 3 uses docs 1, 5, 12, 17, 22
        [3, 6, 13, 18, 23],     # Query 4 uses docs 3, 6, 13, 18, 23
        [1, 5, 10, 19, 24],     # Query 5 uses docs 1, 5, 10, 19, 24
    ]
    
    print(f"Building index with {len(contexts)} contexts...")
    
    response = requests.post(
        f"{BASE_URL}/reorder",
        json={
            "contexts": contexts,
            "initial_tokens_per_context": 0,
            "use_gpu": False,
            "alpha": 0.001,
            "linkage_method": "average"
        },
        timeout=30.0
    )
    
    result = response.json()
    print(f"✓ Index built: {len(result['request_ids'])} request IDs")
    print(f"  Reordered contexts for optimal cache sharing")
    
    return result


def make_inference_request(request_id: str, prompt: str):
    """
    Make an inference request through ContextPilot proxy.
    
    The request_id links this request to the pre-built context index,
    enabling automatic KV cache tracking and eviction coordination.
    """
    response = requests.post(
        f"{BASE_URL}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.0,
            "request_id": request_id  # Links to ContextPilot index
        },
        timeout=60.0
    )
    
    return response.json()


def get_stats():
    """Get current index statistics."""
    response = requests.get(f"{BASE_URL}/stats", timeout=5.0)
    return response.json()


def stateless_schedule():
    """
    Use stateless mode for one-off batch reordering.
    
    This doesn't maintain any index - just clusters and reorders contexts.
    Useful for offline batch processing.
    """
    contexts = [
        [1, 5, 10, 15, 20],
        [2, 5, 11, 16, 21],
        [1, 5, 12, 17, 22],
    ]
    
    print(f"Reordering {len(contexts)} contexts (stateless)...")
    
    response = requests.post(
        f"{BASE_URL}/reorder",
        json={
            "contexts": contexts,
            "alpha": 0.001,
            "linkage_method": "average"
        },
        timeout=30.0
    )
    
    result = response.json()
    print(f"✓ Reordered into {len(result['groups'])} groups")
    
    return result


def main():
    """Complete example workflow."""
    print("=" * 70)
    print("ContextPilot HTTP Server Example")
    print("=" * 70)
    print()
    
    # Check server
    health = check_server()
    if not health:
        return
    
    print()
    
    # Build index
    print("--- Building Index ---")
    build_result = build_index()
    request_ids = build_result["request_ids"]
    print()
    
    # Show reordering
    print("--- Reordered Contexts ---")
    reordered = build_result.get("reordered_contexts", [])
    for i, (rid, ctx) in enumerate(zip(request_ids[:3], reordered[:3])):
        print(f"  {rid}: {ctx}")
    if len(request_ids) > 3:
        print(f"  ... and {len(request_ids) - 3} more")
    print()
    
    # Make inference requests (if inference backend is available)
    print("--- Inference Requests ---")
    try:
        for i, rid in enumerate(request_ids[:2]):
            print(f"Request {i+1} (rid={rid[:8]}...):")
            result = make_inference_request(
                request_id=rid,
                prompt=f"Answer question {i+1} based on the provided documents."
            )
            
            if "choices" in result:
                text = result["choices"][0].get("text", "")[:100]
                print(f"  Response: {text}...")
            elif "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Result: {result}")
            print()
    except Exception as e:
        print(f"  ⚠ Inference backend not available: {e}")
        print("  (This is expected if no inference engine is running)")
    print()
    
    # Show stats
    print("--- Index Stats ---")
    try:
        stats = get_stats()
        evict_stats = stats.get("eviction_stats", {})
        print(f"  Total nodes: {evict_stats.get('total_nodes', 'N/A')}")
        print(f"  Total tokens: {evict_stats.get('total_tokens', 'N/A')}")
    except Exception as e:
        print(f"  Could not get stats: {e}")
    print()
    
    # Stateless scheduling example
    print("--- Stateless Reordering ---")
    try:
        reorder_result = stateless_schedule()
        for group in reorder_result["groups"]:
            print(f"  Group {group['group_id']}: {group['group_size']} contexts")
    except Exception as e:
        print(f"  Could not schedule: {e}")
    print()
    
    print("=" * 70)
    print("✓ Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
