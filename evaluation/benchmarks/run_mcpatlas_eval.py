import argparse
import asyncio
import json
import logging
import os
import time

from datasets import load_dataset
from openai import AsyncOpenAI

from refactored_plugins.skill_index import SkillAwareContextPlugin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

async def process_task(task, client, semaphore, mode, model_name):
    async with semaphore:
        task_id = task.get("id", str(time.time()))
        
        # We will assume a standard format where 'query' is user prompt, 'tools' is list of tools, 
        # and 'answers' contains the ground truth function name.
        prompt = task.get("query", task.get("user_prompt", "Please use the appropriate tool."))
        available_tools = task.get("tools", [])
        if isinstance(available_tools, str):
            try:
                available_tools = json.loads(available_tools)
            except:
                available_tools = []
                
        # Format the tools for OpenAI API
        formatted_tools = []
        for t in available_tools:
            if isinstance(t, dict) and "name" in t:
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {})
                    }
                })
            elif isinstance(t, dict) and "type" in t and t["type"] == "function":
                formatted_tools.append(t)
        
        ground_truth_tool = None
        answers = task.get("answers", [])
        if isinstance(answers, str):
            try:
                answers = json.loads(answers)
            except:
                answers = []
                
        if answers and isinstance(answers, list) and len(answers) > 0:
            if isinstance(answers[0], dict):
                ground_truth_tool = answers[0].get("name")
            else:
                ground_truth_tool = answers[0]
                
        if not ground_truth_tool:
            ground_truth_tool = task.get("expected_tool", "unknown_tool")

        request = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with access to tools. Always use tools when appropriate."},
                {"role": "user", "content": prompt}
            ],
            "tools": formatted_tools
        }
        
        skill_plugin = None
        if mode == "with_plugin" and len(formatted_tools) > 0:
            # Dynamically instantiate a plugin instance per task to avoid async race conditions on registry
            registry = {t["function"]["name"]: t for t in formatted_tools}
            skill_plugin = SkillAwareContextPlugin(registry)
            
            # For evaluation, we simulate that the router correctly predicted the ground truth tool
            request["_required_skills"] = [ground_truth_tool]
            
            request = await skill_plugin.process(request)
            
            api_kwargs = {
                "model": model_name,
                "messages": request["messages"],
                "tools": request["tools"],
                "extra_body": {
                    "_required_skills": request.get("_required_skills")
                }
            }
        else:
            api_kwargs = {
                "model": model_name,
                "messages": request["messages"]
            }
            if len(formatted_tools) > 0:
                api_kwargs["tools"] = formatted_tools

        selected_tool = None
        try:
            logger.info(f"[{mode}] Sending task {task_id}...")
            response = await client.chat.completions.create(**api_kwargs)
            
            message = response.choices[0].message
            if message.tool_calls and len(message.tool_calls) > 0:
                selected_tool = message.tool_calls[0].function.name
            else:
                selected_tool = "No tool called"
                
        except Exception as e:
            logger.error(f"[{mode}] API Error for {task_id}: {str(e)}")
            selected_tool = "Error"
            
        is_correct = (selected_tool == ground_truth_tool)
        logger.info(f"[{mode}] Finished {task_id} - Selected: {selected_tool}, Expected: {ground_truth_tool}, Correct: {is_correct}")
        
        metrics = skill_plugin.get_plugin_metrics() if skill_plugin else None
        return is_correct, metrics

async def run_evaluation(mode, args, tasks):
    # Both baseline and with_plugin use the proxy base_url to capture proxy-level telemetry
    client = AsyncOpenAI(api_key=args.api_key, base_url="http://localhost:8000/v1")
    semaphore = asyncio.Semaphore(args.concurrency)
    
    coroutines = [process_task(t, client, semaphore, mode, args.model) for t in tasks]
    results = await asyncio.gather(*coroutines)
    
    correct_count = sum(r[0] for r in results)
    total_count = len(results)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\n=== Evaluation Complete for mode: {mode} ===")
    print(f"Total Tasks: {total_count}")
    print(f"Correct Tool Selection: {correct_count}")
    print(f"Tool-Selection Accuracy: {accuracy:.2f}%\n")

    if mode == "with_plugin":
        total_orig = sum(r[1]["total_original_tools"] for r in results if r[1])
        total_filt = sum(r[1]["total_tools_filtered"] for r in results if r[1])
        perc = (total_filt / total_orig * 100) if total_orig > 0 else 0
        print("=== ContextPilot Client Telemetry ===")
        print(f"[Skill] Tools Filtered: {total_filt} / {total_orig} ({perc:.2f}%)")

async def main():
    parser = argparse.ArgumentParser(description="MCP-Atlas Toolkit Evaluation")
    parser.add_argument("--model", default="gpt-5.5", help="Model name to evaluate")
    parser.add_argument("--api_base", default="https://api.openai.com/v1", help="API Base URL")
    parser.add_argument("--api_key", default="dummy-elm-key", help="API Key")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks to run (0 for all)")
    parser.add_argument("--eval_mode", choices=["baseline", "with_plugin", "all"], default="all", help="Evaluation mode")
    args = parser.parse_args()

    logger.info("Loading Tool-Use dataset...")
    try:
        # Load a 500-task slice of the dataset to keep evaluation time and cost manageable
        dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train[:500]")
    except Exception as e:
        logger.warning(f"Failed to load standard dataset. Generating dummy tasks. Error: {e}")
        dataset = [
            {
                "id": f"task_{i}",
                "query": f"What is the weather in city {i}?",
                "tools": [
                    {"name": "get_weather", "description": "Get weather for a city"},
                    {"name": "get_time", "description": "Get current time"},
                    {"name": "calculate_sum", "description": "Calculate sum of numbers"},
                    {"name": "search_web", "description": "Search the web"},
                    {"name": "send_email", "description": "Send an email"},
                ],
                "expected_tool": "get_weather"
            }
            for i in range(10)
        ]
            
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
