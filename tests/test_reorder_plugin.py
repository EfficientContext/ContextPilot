
import pytest
import asyncio
from refactored_plugins.reorder import ContextReorderPlugin

@pytest.mark.asyncio
async def test_context_reorder_plugin_end_to_end():
    """
    E2E Test: Reorder a batch of OpenAI requests with overlapping system prompts.
    """
    # 1. Setup Mock OpenAI Requests
    # Group A: Common system prompt + Tool Set A
    # Group B: Common system prompt + Tool Set B
    system_prompt = "You are an assistant."
    tools_a = "Tools for math: add, subtract."
    tools_b = "Tools for coding: python, bash."
    
    requests = [
        {"id": "req1", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": "How's the weather?"}]}, # No tools
        {"id": "req2", "messages": [{"role": "system", "content": system_prompt}, {"role": "system", "content": tools_a}, {"role": "user", "content": "Add 5+5"}]}, # Tools A
        {"id": "req3", "messages": [{"role": "system", "content": system_prompt}, {"role": "system", "content": tools_b}, {"role": "user", "content": "Write python script"}]}, # Tools B
        {"id": "req4", "messages": [{"role": "system", "content": system_prompt}, {"role": "system", "content": tools_a}, {"role": "user", "content": "Subtract 10-2"}]}, # Tools A
    ]
    
    # 2. Initialize Plugin (using small alpha to detect overlap)
    # We use a dummy model name to avoid downloading 7B tokenizer during test (uses fallback)
    plugin = ContextReorderPlugin(model_name="test-model", alpha=0.1)
    
    # 3. Process Batch
    reordered_requests = await plugin.process(requests)
    
    # 4. Assertions
    assert len(reordered_requests) == len(requests)
    
    # In a perfect world, req2 and req4 (Tools A) should be adjacent 
    # and follow the same prefix logic we saw in the verification script.
    
    # Let's find the positions of Tools A requests
    tools_a_indices = [i for i, r in enumerate(reordered_requests) if "Add 5+5" in str(r) or "Subtract 10-2" in str(r)]
    
    # Check if they are adjacent
    is_adjacent = abs(tools_a_indices[0] - tools_a_indices[1]) == 1
    
    print("\nReordered Sequence IDs:", [r["id"] for r in reordered_requests])
    
    assert is_adjacent, f"Tools A requests were not grouped together! Indices: {tools_a_indices}"
    print("\nSUCCESS: ContextReorderPlugin grouped requests with shared tool definitions.")

if __name__ == "__main__":
    asyncio.run(test_context_reorder_plugin_end_to_end())
