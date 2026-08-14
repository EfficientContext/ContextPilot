import argparse
import asyncio
import json
import logging
import os
import re

from datasets import load_dataset
from openai import AsyncOpenAI

from refactored_plugins.dynamic_pruning import DynamicPruningPlugin
from refactored_plugins.skill_index import SkillAwareContextPlugin

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

async def process_task(task, client, semaphore, output_file, turn_1_id, mode, model_name):
    """
    Processes a single BigCodeBench task through our ELM API (bypassing or routing to proxy).
    """
    async with semaphore:
        task_id = task.get("task_id", "unknown_task")
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
        if mode == "with_plugin":
            # Send extra_body for ContextPilot proxy to intercept
            request["_required_skills"] = ["tool_1", "tool_3", "tool_7"]
            
            # 1. Apply plugins on the client before sending
            request = await dynamic_plugin.process(request)
            request = await skill_plugin.process(request)
            
            api_kwargs = {
                "model": model_name,
                "messages": request["messages"],
                "tools": request["tools"]
            }
            
            api_kwargs["extra_body"] = {
                "user_id": request.get("user_id"),
                "parent_id": request.get("parent_id"),
                "_required_skills": request.get("_required_skills")
            }
        else:
            api_kwargs = {
                "model": model_name,
                "messages": request["messages"],
                "tools": request["tools"]
            }

        try:
            logger.info(f"[{mode}] Sending task {task_id}...")
            response = await client.chat.completions.create(**api_kwargs)
            response_content = response.choices[0].message.content
        except Exception as e:
            logger.error(f"[{mode}] API Error for {task_id}: {str(e)}")
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
            
        logger.info(f"[{mode}] Finished {task_id}")

async def run_evaluation(mode, args, tasks):
    # Route BOTH baseline and with_plugin through the proxy to intercept prompt_cache_hit_tokens
    client = AsyncOpenAI(api_key=args.api_key, base_url="http://localhost:8000/v1")
        
    # We use a dummy turn_1_id for simulation
    turn_1_id = "test-turn-1-id"
        
    output_file = os.path.join(os.path.dirname(__file__), f"results_dynamic_skill_{mode}_{args.model}.jsonl")
    if os.path.exists(output_file):
        os.remove(output_file)
        
    # Use configurable Semaphore to allow high concurrency
    semaphore = asyncio.Semaphore(args.concurrency)
    
    # Instantiate plugins
    from refactored_plugins.dynamic_pruning import DynamicPruningPlugin
    from refactored_plugins.skill_index import SkillAwareContextPlugin
    global dynamic_plugin, skill_plugin
    dynamic_plugin = DynamicPruningPlugin(similarity_threshold=0.3)
    skill_plugin = SkillAwareContextPlugin(DUMMY_TOOL_REGISTRY)

    coroutines = [process_task(t, client, semaphore, output_file, turn_1_id, mode, args.model) for t in tasks]
    await asyncio.gather(*coroutines)
    
    print(f"\n=== Evaluation Complete for mode: {mode} ===")
    print(f"Results saved to {output_file}")

    if mode == "with_plugin":
        print("\n=== ContextPilot Client Telemetry ===")
        dynamic_metrics = dynamic_plugin.get_plugin_metrics()
        skill_metrics = skill_plugin.get_plugin_metrics()
        print(f"[Dynamic Pruning] Chars Saved: {dynamic_metrics['total_chars_saved']} / {dynamic_metrics['total_original_chars']} ({dynamic_metrics['chars_saved_percentage']:.2f}%)")
        print(f"[Skill] Tools Filtered: {skill_metrics['total_tools_filtered']} / {skill_metrics.get('total_original_tools', 'N/A')} ({skill_metrics.get('tools_filtered_percentage', 0):.2f}%)")


async def main():
    parser = argparse.ArgumentParser(description="BigCodeBench ELM API Runner")
    parser.add_argument("--model", default="gpt-5.5", help="Model name to evaluate")
    parser.add_argument("--api_base", default=os.environ.get("BASE_URL", "https://api.openai.com/v1"), help="Baseline ELM API Base URL")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "dummy-elm-key"), help="API Key")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks to run (0 for all)")
    parser.add_argument("--eval_mode", choices=["baseline", "with_plugin", "all"], default="all", help="Evaluation mode")
    args = parser.parse_args()

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
    if args.limit > 0:
        tasks = tasks[:args.limit]
        logger.info(f"Loaded {len(tasks)} tasks (LIMITED) for evaluation.")
    else:
        logger.info(f"Loaded {len(tasks)} tasks for full evaluation.")
    
    if args.eval_mode == "all":
        modes = ["baseline", "with_plugin"]
    else:
        modes = [args.eval_mode]
        
    for mode in modes:
        logger.info(f"\n--- Starting Evaluation: {mode} ---")
        await run_evaluation(mode, args, tasks)
        
if __name__ == "__main__":
    asyncio.run(main())
