import asyncio
import os
import json
import logging
from typing import Any, Dict
from openai import AsyncOpenAI

# Set PYTHONPATH in the environment before running if needed
from refactored_plugins.skill_index import SkillAwareContextPlugin
from refactored_plugins.dedup import ContextDedupPlugin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Dummy tool registry for SkillAwareContextPlugin to filter from
DUMMY_TOOL_REGISTRY = {
    "python_repl": {
        "type": "function",
        "function": {
            "name": "python_repl",
            "description": "Executes Python code in a sandboxed environment"
        }
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web for up-to-date documentation"
        }
    },
    "file_writer": {
        "type": "function",
        "function": {
            "name": "file_writer",
            "description": "Writes code to a file"
        }
    }
}

async def evaluate_task(task_data: Dict[str, Any], 
                       skill_plugin: SkillAwareContextPlugin, 
                       dedup_plugin: ContextDedupPlugin, 
                       client: AsyncOpenAI) -> Dict[str, Any]:
    """
    Optimizes a request using Phase 1 plugins and sends it to the ELM API.
    """
    # 1. ContextDedupPlugin (strip redundant history)
    optimized_data = await dedup_plugin.process(task_data)
    
    # 2. SkillAwareContextPlugin (strip redundant tools based on _required_skills)
    optimized_data = await skill_plugin.process(optimized_data)
    
    # 3. Call ELM API via OpenAI client
    api_kwargs = {
        "model": "gpt-5.5",
        "messages": optimized_data.get("messages", [])
    }
    
    # Only pass tools if the plugin injected any
    if "tools" in optimized_data and optimized_data["tools"]:
        api_kwargs["tools"] = optimized_data["tools"]
        
    try:
        response = await client.chat.completions.create(**api_kwargs)
        message = response.choices[0].message
        if message.tool_calls:
            # If the model decides to invoke tools, format the tool calls details
            calls = []
            for tc in message.tool_calls:
                # Safely parse arguments if present
                args = {}
                if tc.function.arguments:
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = tc.function.arguments
                calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args
                })
            response_content = f"Tool Calls Triggered:\n{json.dumps(calls, indent=2)}"
        else:
            response_content = message.content
    except Exception as e:
        response_content = f"API Error: {str(e)}"
        response = None
    
    # 4. Gather Telemetry
    telemetry = {
        "skill_plugin_metrics": skill_plugin.get_plugin_metrics(),
        "dedup_plugin_metrics": dedup_plugin.get_plugin_metrics()
    }
    
    return {
        "response": response,
        "response_content": response_content,
        "telemetry": telemetry,
        "optimized_payload": optimized_data
    }

async def main():
    api_key = os.environ.get("OPENAI_API_KEY", "dummy-elm-key")
    base_url = os.environ.get("BASE_URL", "https://api.openai.com/v1")
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    skill_plugin = SkillAwareContextPlugin(tool_registry=DUMMY_TOOL_REGISTRY)
    dedup_plugin = ContextDedupPlugin()
    
    print("=== Phase 2: ELM API Evaluator ===")
    
    # To demonstrate deduplication savings, we first run a mock Turn 1 to prime the history
    turn_1 = {
        "user_id": "evaluator_1",
        "messages": [
            {"role": "system", "content": "You are an expert Python engineer taking the BigCodeBench evaluation."},
            {"role": "user", "content": "Write a script to compute the fast inverse square root."},
            {"role": "assistant", "content": "Here is the implementation: `def q_rsqrt(number): ...`"}
        ]
    }
    turn_1_res = await dedup_plugin.process(turn_1)
    parent_id = turn_1_res.get("current_id")
    
    # Mock BigCodeBench Turn 2 Task (includes redundant history from Turn 1)
    mock_task = {
        "user_id": "evaluator_1",
        "parent_id": parent_id,
        "_required_skills": ["python_repl", "file_writer"],
        "messages": [
            {"role": "system", "content": "You are an expert Python engineer taking the BigCodeBench evaluation."},
            {"role": "user", "content": "Write a script to compute the fast inverse square root."},
            {"role": "assistant", "content": "Here is the implementation: `def q_rsqrt(number): ...`"},
            {"role": "user", "content": "Now, execute this code in the python_repl to verify it handles float(0.15625) correctly."}
        ]
    }
    
    print(f"\n[1] Starting API Request Evaluation...")
    result = await evaluate_task(mock_task, skill_plugin, dedup_plugin, client)
    
    print("\n[2] Optimized Payload Sent to ELM API:")
    print(json.dumps(result["optimized_payload"], indent=2))
    
    print("\n[3] ELM API Response:")
    print(result["response_content"])
    
    print("\n[4] Cost-Savings Telemetry:")
    print(json.dumps(result["telemetry"], indent=2))

if __name__ == "__main__":
    asyncio.run(main())
