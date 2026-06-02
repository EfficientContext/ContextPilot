import json
import re
import time
import random

def heavy_json_ops():
    """Simulates API payload handling (serialization/deserialization)."""
    # Create a reasonably complex nested structure
    payload = {
        "metadata": {"version": "1.0", "timestamp": time.time()},
        "agents": [
            {"id": i, "name": f"Agent_{i}", "history": ["observation" * 10 for _ in range(20)]}
            for i in range(50)
        ],
        "configuration": {f"key_{i}": "value" * 50 for i in range(100)}
    }
    
    # Burn CPU parsing and serializing
    for _ in range(1000): # Reduced from 10k to keep dummy run time reasonable (approx few seconds)
        s = json.dumps(payload)
        _ = json.loads(s)

def heavy_regex_ops():
    """Simulates prompt formatting and log parsing."""
    # Large block of text
    base_text = "The quick brown fox jumps over the lazy dog. " * 500
    
    # Search for patterns and manipulate strings
    patterns = [r"\b\w{5}\b", r"fox.*?dog", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"]
    
    for _ in range(50):
        # Regex search
        for p in patterns:
            re.findall(p, base_text)
        
        # String concatenation (O(n^2) behavior in some languages, but Python is optimized)
        # Still burns time for large N
        _ = "".join([base_text[i:i+10] for i in range(0, len(base_text), 2)])

def agent_turn(turn_id):
    print(f"Executing Agent Turn {turn_id}...")
    heavy_json_ops()
    heavy_regex_ops()

def main():
    start_time = time.time()
    # Simulate 50 agent turns
    for i in range(50):
        agent_turn(i)
    
    end_time = time.time()
    print(f"\nSimulation complete in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
