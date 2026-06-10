import asyncio
import json
import logging
import sys
from typing import Any, Dict, List

# Import all 4 plugins from refactored_plugins
from refactored_plugins.skill_index import SkillAwareContextPlugin
from refactored_plugins.dedup import ContextDedupPlugin
from refactored_plugins.reorder import ContextReorderPlugin
from refactored_plugins.kv_lookup import KVCacheLookupPlugin

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class MockProxy:
    """
    Simulates the core Token Proxy Middleware pipeline by chaining all Phase 1 plugins.
    """
    def __init__(self):
        # 1. SkillAwareContextPlugin (dummy tool registry)
        dummy_tool_registry = {
            "math": {
                "type": "function",
                "function": {
                    "name": "math_tool",
                    "description": "Performs mathematical calculations"
                }
            },
            "weather": {
                "type": "function",
                "function": {
                    "name": "weather_tool",
                    "description": "Gets the current weather"
                }
            }
        }
        self.skill_plugin = SkillAwareContextPlugin(tool_registry=dummy_tool_registry)
        
        # 2. ContextDedupPlugin
        self.dedup_plugin = ContextDedupPlugin()
        
        # 3. ContextReorderPlugin
        # We specify use_gpu=False for the mock test to avoid requiring torch/CUDA.
        self.reorder_plugin = ContextReorderPlugin(use_gpu=False)
        
        # 4. KVCacheLookupPlugin (dummy ZMQ endpoints)
        dummy_endpoints = ["tcp://localhost:5557", "tcp://localhost:5558"]
        self.kv_lookup_plugin = KVCacheLookupPlugin(endpoints=dummy_endpoints)

    async def process_batch(self, request_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes the exact chain of execution for a batch of requests.
        """
        # 1. SkillAwareContextPlugin (processes individual requests)
        batch_after_skill = []
        for req in request_batch:
            res = await self.skill_plugin.process(req)
            batch_after_skill.append(res)
            
        # 2. ContextDedupPlugin (processes individual requests)
        batch_after_dedup = []
        user_to_last_id = {}
        for req in batch_after_skill:
            user_id = req.get("user_id")
            if user_id in user_to_last_id:
                req["parent_id"] = user_to_last_id[user_id]
                
            res = await self.dedup_plugin.process(req)
            
            if user_id and "current_id" in res:
                user_to_last_id[user_id] = res["current_id"]
                
            batch_after_dedup.append(res)
            
        # 3. ContextReorderPlugin (processes the entire batch)
        batch_after_reorder = await self.reorder_plugin.process(batch_after_dedup)
        
        # 4. KVCacheLookupPlugin (processes individual requests)
        final_batch = []
        for req in batch_after_reorder:
            res = await self.kv_lookup_plugin.process(req)
            final_batch.append(res)
            
        return final_batch

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Aggregates telemetry from all plugins."""
        return {
            "skill_plugin": self.skill_plugin.get_plugin_metrics(),
            "dedup_plugin": self.dedup_plugin.get_plugin_metrics(),
            "reorder_plugin": self.reorder_plugin.get_plugin_metrics(),
            "kv_lookup_plugin": self.kv_lookup_plugin.get_plugin_metrics(),
        }

async def main():
    # Create a complex mock batch of 3 OpenAI requests simulating:
    # 1. Multi-turn conversation (redundant history)
    # 2. Dynamic Tool/Skill filtering
    # 3. Overlapping system prompts for Prefix Sharing
    mock_batch = [
        {
            "user_id": "user_1",
            "_required_skills": ["math"],
            "messages": [
                {"role": "system", "content": "You are an AI assistant. Answer accurately and be concise."},
                {"role": "user", "content": "Hello! I am preparing for my exams."},
                {"role": "assistant", "content": "Hello! I can help you study. What subject?"},
                {"role": "user", "content": "Calculate 15 * 32 for my math homework."}
            ]
        },
        {
            "user_id": "user_1",
            "_required_skills": ["weather"],
            "messages": [
                {"role": "system", "content": "You are an AI assistant. Answer accurately and be concise."},
                {"role": "user", "content": "Hello! I am preparing for my exams."},
                {"role": "assistant", "content": "Hello! I can help you study. What subject?"},
                {"role": "user", "content": "Actually, skip studying. What is the weather outside?"}
            ]
        },
        {
            "user_id": "user_2",
            "_required_skills": ["math", "weather", "invalid_skill"],
            "messages": [
                {"role": "system", "content": "You are an AI assistant. Answer accurately and be concise."},
                {"role": "user", "content": "I need help with math and weather!"}
            ]
        }
    ]

    print("=== Phase 1: Core Merge - Mock Proxy Pipeline ===\n")
    print("[1] Initializing Mock Proxy & 4 Plugins...")
    proxy = MockProxy()
    
    print("\n[2] Executing process_batch()...")
    optimized_batch = await proxy.process_batch(mock_batch)
    
    print("\n=== Optimized Batch Output ===")
    print(json.dumps(optimized_batch, indent=2))
    
    print("\n=== Telemetry Metrics ===")
    metrics = proxy.get_all_metrics()
    print(json.dumps(metrics, indent=2))
    
    # Force exit to cleanly terminate lingering background ZMQ tasks created by KVCacheLookupPlugin
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
