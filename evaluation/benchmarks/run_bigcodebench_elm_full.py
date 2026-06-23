import asyncio
import json
import logging
import os
import re

# pip install datasets
from datasets import load_dataset
from openai import AsyncOpenAI

# Set PYTHONPATH in the environment before running
from refactored_plugins.skill_index import SkillAwareContextPlugin
from refactored_plugins.dedup import ContextDedupPlugin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Create a registry of 10 dummy tools to trigger the Skill plugin
DUMMY_TOOL_REGISTRY = {
    f"tool_{i}": {
        "type": "function",
        "function": {
            "name": f"tool_{i}",
            "description": f"Dummy tool number {i}"
        }
    }
    for i in range(1, 11)
}

async def process_task(task, skill_plugin, dedup_plugin, client, semaphore, output_file, turn_1_id):
    """
    Processes a single BigCodeBench task through our ContextPilot plugins and ELM API.
    """
    async with semaphore:
        task_id = task.get("task_id", "unknown_task")
        # BigCodeBench prompts are usually in 'complete_prompt' or 'instruction'
        prompt = task.get("complete_prompt", task.get("instruction", "No prompt found."))
        
        # Mock heavy agent request with redundant history and bloated tools
        request = {
            "user_id": "evaluator_1",
            "parent_id": turn_1_id,
            "_required_skills": ["tool_1", "tool_3", "tool_7"],  # Require only 3 tools out of 10
            "messages": [
                {"role": "system", "content": "You are a senior python developer. Always wrap your code in ```python blocks."},
                {"role": "user", "content": "Please help me write some code."},
                {"role": "assistant", "content": "Of course! I can help you with that."},
                {"role": "user", "content": prompt}
            ],
            "tools": list(DUMMY_TOOL_REGISTRY.values())
        }
        
        # Pass through ContextPilot local plugins
        optimized_request = await dedup_plugin.process(request)
        optimized_request = await skill_plugin.process(optimized_request)
        
        # Prepare ELM API request (OpenAI-compatible)
        api_kwargs = {
            "model": "gpt-5.5",
            "messages": optimized_request.get("messages", [])
        }
        if "tools" in optimized_request and optimized_request["tools"]:
            api_kwargs["tools"] = optimized_request["tools"]
            
        try:
            logger.info(f"Sending optimized task {task_id} to ELM API...")
            response = await client.chat.completions.create(**api_kwargs)
            response_content = response.choices[0].message.content
        except Exception as e:
            logger.error(f"API Error for {task_id}: {str(e)}")
            response_content = ""
            
        # Extract code block using regex
        extracted_code = ""
        if response_content:
            match = re.search(r"```python\s*(.*?)\s*```", response_content, re.DOTALL)
            if match:
                extracted_code = match.group(1).strip()
            else:
                # Fallback if the LLM didn't use the markdown block
                extracted_code = response_content.strip()

        # Append result to JSONL
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"task_id": task_id, "solution": extracted_code}) + "\n")
            
        logger.info(f"Finished {task_id}")

async def main():
    api_key = os.environ.get("OPENAI_API_KEY", "dummy-elm-key")
    base_url = os.environ.get("BASE_URL", "https://api.openai.com/v1")
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    skill_plugin = SkillAwareContextPlugin(tool_registry=DUMMY_TOOL_REGISTRY)
    dedup_plugin = ContextDedupPlugin()
    
    # Pre-warm Dedup plugin with the initial messages to simulate conversation history
    turn_1 = {
        "user_id": "evaluator_1",
        "messages": [
            {"role": "system", "content": "You are a senior python developer. Always wrap your code in ```python blocks."},
            {"role": "user", "content": "Please help me write some code."},
            {"role": "assistant", "content": "Of course! I can help you with that."}
        ]
    }
    turn_1_res = await dedup_plugin.process(turn_1)
    turn_1_id = turn_1_res.get("current_id")

    # Load BigCodeBench dataset
    logger.info("Loading BigCodeBench dataset...")
    try:
        dataset = load_dataset("bigcode/bigcodebench", split="train")
    except Exception as e:
        logger.warning(f"Failed to load split='train'. Trying standard default split. Error: {e}")
        # Fallback to the common default split format if 'train' split does not exist
        try:
            dataset = load_dataset("bigcode/bigcodebench", split="v0.1.2")
        except Exception:
            dataset = load_dataset("bigcode/bigcodebench", split="v0.1.0_240822")
            
    # Select all tasks for full evaluation
    tasks = list(dataset)
    logger.info(f"Loaded {len(tasks)} tasks for full evaluation.")
    
    output_file = os.path.join(os.path.dirname(__file__), "elm_samples_full.jsonl")
    if os.path.exists(output_file):
        os.remove(output_file)
        
    # Use a Semaphore with 1 to process sequentially and avoid early rate limits
    semaphore = asyncio.Semaphore(1) 
    
    coroutines = [process_task(t, skill_plugin, dedup_plugin, client, semaphore, output_file, turn_1_id) for t in tasks]
    await asyncio.gather(*coroutines)
    
    print("\n=== Phase 2 Full Evaluation Complete ===")
    print(f"Results saved to {output_file}")
    
    print("\n=== Combined Cost-Savings Telemetry ===")
    metrics = {
        "skill_plugin_metrics": skill_plugin.get_plugin_metrics(),
        "dedup_plugin_metrics": dedup_plugin.get_plugin_metrics()
    }
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
