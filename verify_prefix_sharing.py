
import time
from typing import List
from contextpilot.server.live_index import ContextPilot, compute_prefix_length

def calculate_total_prefix_sharing(contexts: List[List[int]]) -> int:
    if not contexts:
        return 0
    
    total_sharing = 0
    for i in range(1, len(contexts)):
        shared = compute_prefix_length(contexts[i-1], contexts[i])
        total_sharing += shared
    return total_sharing

def run_verification():
    print("=" * 60)
    print("VERIFYING PREFIX SHARING BENEFIT")
    print("=" * 60)

    # 1. Generate overlapping synthetic contexts
    system_prompt = list(range(1, 101)) # 100
    tool_defs = list(range(101, 201))    # 100
    
    contexts = []
    for i in range(20):
        if i < 10:
            ctx = system_prompt + tool_defs + [1000 + i]
        else:
            ctx = system_prompt + [2000 + i]
        contexts.append(ctx)

    # 2. Calculate sharing BEFORE reordering
    import random
    random.seed(42)
    shuffled_contexts = list(contexts)
    random.shuffle(shuffled_contexts)
    
    sharing_before = calculate_total_prefix_sharing(shuffled_contexts)
    
    # 3. Apply ContextPilot reordering (using the full Pilot)
    pilot = ContextPilot(use_gpu=False, alpha=0.1)
    # build_and_schedule returns reordered_contexts in the scheduled order
    result = pilot.build_and_schedule(shuffled_contexts)
    reordered_contexts = result["reordered_contexts"]
    
    sharing_after = calculate_total_prefix_sharing(reordered_contexts)
    
    # 4. Debug: Check the first few reordered contexts
    print("\nDEBUG: Reordered Context Structure")
    for i in range(min(10, len(reordered_contexts))):
        ctx = reordered_contexts[i]
        has_tools = all(t in ctx for t in tool_defs[:10])
        print(f"  Context {i}: len={len(ctx)}, has_tool_defs={has_tools}")

    # 5. Results
    print(f"\nBatch Size:         {len(contexts)} contexts")
    print(f"Sharing BEFORE:     {sharing_before} tokens")
    print(f"Sharing AFTER:      {sharing_after} tokens")
    
    improvement = (sharing_after - sharing_before) / (sharing_before + 1e-9) * 100
    print(f"Improvement:        {improvement:.2f}%")
    
    if sharing_after > sharing_before:
        print("\nSUCCESS: ContextPilot increased prefix sharing!")
    elif sharing_after == sharing_before:
        print("\nNEUTRAL: No change. Is the baseline already optimal?")
    else:
        print("\nFAILURE: Reordering decreased sharing!")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
