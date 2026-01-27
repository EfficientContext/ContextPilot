"""
Example: Stateless Batch Scheduling with ContextPilot

This shows how to use ContextPilot in STATELESS mode - just for clustering and 
scheduling contexts WITHOUT tracking SGLang's cache state.

Use this when:
1. You want to process batches independently
2. You don't need eviction sync with SGLang
3. You just want optimal ordering for prefix sharing

SETUP:
1. Start ContextPilot server in stateless mode:
   python -m contextpilot.server.http_server --port 8765 --stateless
   
2. Send batches to /schedule endpoint
"""

import requests
import json
from contextpilot.server.http_client import ContextPilotIndexClient, schedule_batch


# ============================================================================
# OPTION 1: Using the Client Class
# ============================================================================

def example_with_client():
    """Use the client class for multiple batch operations."""
    
    # Create client
    client = ContextPilotIndexClient("http://localhost:8765")
    
    # Your RAG contexts (each context is a list of document IDs)
    contexts = [
        [1, 5, 10, 15, 20],     # Query 1 uses docs 1, 5, 10, 15, 20
        [2, 5, 11, 16, 21],     # Query 2 uses docs 2, 5, 11, 16, 21
        [1, 5, 12, 17, 22],     # Query 3 uses docs 1, 5, 12, 17, 22
        [3, 6, 13, 18, 23],     # Query 4 uses docs 3, 6, 13, 18, 23
        [1, 5, 10, 19, 24],     # Query 5 uses docs 1, 5, 10, 19, 24
    ]
    
    print("Scheduling batch with ContextPilot (stateless mode)...")
    result = client.schedule(contexts)
    
    if result:
        print(f"\n✓ Batch scheduled successfully!")
        print(f"  Mode: {result.get('mode', 'stateless')}")
        print(f"  Number of contexts: {result['num_contexts']}")
        print(f"  Number of execution groups: {result['num_groups']}")
        
        print(f"\nScheduled order (original indices): {result['original_indices']}")
        print(f"\nExecution groups:")
        for i, group in enumerate(result['groups']):
            print(f"  Group {i}: {group}")
        
        # Send to SGLang in this order
        scheduled_contexts = result['scheduled_contexts']
        print(f"\n→ Send contexts to SGLang in this order for optimal prefix sharing")
    else:
        print("Failed to schedule batch")
    
    client.close()


# ============================================================================
# OPTION 2: Using Convenience Function
# ============================================================================

def example_with_function():
    """Use the simple function for one-off batch scheduling."""
    
    contexts = [
        [1, 5, 10, 15, 20],
        [2, 5, 11, 16, 21],
        [1, 5, 12, 17, 22],
        [3, 6, 13, 18, 23],
        [1, 5, 10, 19, 24],
    ]
    
    print("Scheduling batch with convenience function...")
    result = schedule_batch(
        contexts=contexts,
        server_url="http://localhost:8765",
        alpha=0.005,
        use_gpu=False
    )
    
    if result:
        print(f"✓ Scheduled {result['num_contexts']} contexts into {result['num_groups']} groups")
        print(f"Original indices order: {result['original_indices']}")
    else:
        print("Failed to schedule batch")


# ============================================================================
# OPTION 3: Direct HTTP Request (No Client Required)
# ============================================================================

def example_direct_http():
    """Make direct HTTP request without any client library."""
    
    contexts = [
        [1, 5, 10, 15, 20],
        [2, 5, 11, 16, 21],
        [1, 5, 12, 17, 22],
        [3, 6, 13, 18, 23],
        [1, 5, 10, 19, 24],
    ]
    
    print("Scheduling batch with direct HTTP request...")
    response = requests.post(
        "http://localhost:8765/schedule",
        json={
            "contexts": contexts,
            "alpha": 0.005,
            "use_gpu": False,
            "linkage_method": "average"
        },
        timeout=30.0
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Scheduled successfully!")
        print(f"  Groups: {result['num_groups']}")
        print(f"  Order: {result['original_indices']}")
    else:
        print(f"Failed: {response.text}")


# ============================================================================
# EXAMPLE: Batch Processing Workflow
# ============================================================================

def batch_processing_workflow():
    """
    Complete workflow: Schedule batch → Send to SGLang → Process responses.
    
    This is the typical workflow when using ContextPilot in stateless mode
    without needing cache sync.
    """
    
    print("=" * 60)
    print("BATCH PROCESSING WORKFLOW (Stateless Mode)")
    print("=" * 60)
    
    # Step 1: Prepare your contexts
    # Each context is a list of document IDs that a query needs
    contexts = [
        [101, 102, 103, 104, 105],  # Query A: needs docs 101-105
        [101, 102, 106, 107, 108],  # Query B: shares 101, 102 with A
        [201, 202, 203, 204, 205],  # Query C: completely different
        [101, 102, 103, 109, 110],  # Query D: shares 101, 102, 103 with A
        [201, 202, 206, 207, 208],  # Query E: shares 201, 202 with C
    ]
    original_prompts = [
        "Question about topic A",
        "Question about topic B",
        "Question about topic C",
        "Question about topic D",
        "Question about topic E",
    ]
    
    print(f"\n1. Prepared {len(contexts)} queries with their contexts")
    
    # Step 2: Get optimal scheduling from ContextPilot
    print("\n2. Calling ContextPilot /schedule endpoint...")
    result = schedule_batch(contexts=contexts)
    
    if not result:
        print("   ✗ Scheduling failed!")
        return
    
    scheduled_order = result['original_indices']
    print(f"   ✓ Optimal order: {scheduled_order}")
    print(f"   ✓ {result['num_groups']} execution groups")
    
    # Step 3: Reorder your data according to the schedule
    print("\n3. Reordering data for SGLang...")
    reordered_contexts = [contexts[i] for i in scheduled_order]
    reordered_prompts = [original_prompts[i] for i in scheduled_order]
    
    for i, (prompt, ctx) in enumerate(zip(reordered_prompts, reordered_contexts)):
        orig_idx = scheduled_order[i]
        print(f"   Position {i}: Original query {orig_idx} - {prompt[:30]}... (docs: {ctx[:3]}...)")
    
    # Step 4: Send to SGLang in this order
    print("\n4. Send to SGLang in scheduled order...")
    print("   (This would be your actual SGLang API calls)")
    
    # Example pseudo-code for SGLang:
    # for prompt, context in zip(reordered_prompts, reordered_contexts):
    #     full_prompt = build_prompt(prompt, context)
    #     response = sglang_client.generate(full_prompt)
    #     results.append(response)
    
    # Step 5: Reorder results back to original order
    print("\n5. After getting responses, reorder back to original indices")
    print("   reverse_mapping = {scheduled_order[i]: i for i in range(len(scheduled_order))}")
    print("   original_order_results = [results[reverse_mapping[i]] for i in range(len(results))]")
    
    print("\n" + "=" * 60)
    print("DONE - Results are now in original query order")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    print("ContextPilot Stateless Batch Scheduling Example")
    print("-" * 50)
    print("\nMake sure the server is running in stateless mode:")
    print("  python -m contextpilot.server.http_server --port 8765 --stateless")
    print("-" * 50)
    
    # Check server health
    try:
        response = requests.get("http://localhost:8765/health", timeout=2)
        if response.status_code == 200:
            health = response.json()
            print(f"\n✓ Server is running (mode: {health.get('mode', 'unknown')})")
        else:
            print(f"\n✗ Server returned status {response.status_code}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to server at http://localhost:8765")
        print("  Please start the server first with: python -m contextpilot.server.http_server --port 8765 --stateless")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    
    # Run the batch processing workflow example
    batch_processing_workflow()
