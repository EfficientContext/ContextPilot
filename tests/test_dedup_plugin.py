
import pytest
import asyncio
from refactored_plugins.dedup import ContextDedupPlugin

@pytest.mark.asyncio
async def test_context_dedup_plugin_multi_turn():
    """
    Test: Deduplicate a two-turn conversation.
    Turn 1: System + Question 1
    Turn 2: System + Question 1 + Answer 1 + Question 2
    """
    plugin = ContextDedupPlugin()
    
    # --- TURN 1 ---
    system_msg = {"role": "system", "content": "You are a helpful assistant."}
    q1 = {"role": "user", "content": "What is the capital of France?"}
    
    request_t1 = {
        "user_id": "user123",
        "messages": [system_msg, q1]
    }
    
    # Process Turn 1 (should be no change, but registers the history)
    resp_t1 = await plugin.process(request_t1)
    assert len(resp_t1["messages"]) == 2
    assert resp_t1["messages"][0]["content"] == system_msg["content"]
    
    t1_id = resp_t1["current_id"]

    # --- TURN 2 (The Agent framework sends the whole history again) ---
    a1 = {"role": "assistant", "content": "The capital of France is Paris."}
    q2 = {"role": "user", "content": "And Germany?"}
    
    request_t2 = {
        "user_id": "user123",
        "parent_id": t1_id, # Link to Turn 1
        "messages": [
            system_msg, # Duplicate
            q1,         # Duplicate
            a1,         # New
            q2          # New
        ]
    }
    
    # Calculate original length
    original_total_len = sum(len(m["content"]) for m in request_t2["messages"])
    
    # Process Turn 2
    resp_t2 = await plugin.process(request_t2)
    
    # Calculate deduplicated length
    dedup_total_len = sum(len(m["content"]) for m in resp_t2["messages"])
    
    # --- ASSERTIONS ---
    assert len(resp_t2["messages"]) == 4
    
    # The first two messages should now be hints
    assert "[Reference to Turn" in resp_t2["messages"][0]["content"]
    assert "[Reference to Turn" in resp_t2["messages"][1]["content"]
    
    # The new messages should remain intact
    assert resp_t2["messages"][2]["content"] == a1["content"]
    assert resp_t2["messages"][3]["content"] == q2["content"]
    
    # Compression check
    assert dedup_total_len < original_total_len
    
    print(f"\nOriginal Length: {original_total_len} chars")
    print(f"Dedup Length:    {dedup_total_len} chars")
    print(f"Compression:     {(1 - dedup_total_len/original_total_len)*100:.2f}%")
    print("\nSUCCESS: ContextDedupPlugin compressed multi-turn history using reference hints.")

if __name__ == "__main__":
    asyncio.run(test_context_dedup_plugin_multi_turn())
