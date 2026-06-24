#!/usr/bin/env python3
"""
Custom local evaluator for BigCodeBench smoke-test subset.
Runs test cases for only the tasks present in our samples file,
bypassing the full-dataset assertion in bigcodebench.evaluate.

Usage (inside Apptainer container):
    python3 /app/results/run_local_eval.py --samples /app/results/elm_samples.jsonl
"""
import argparse
import json
import os
import sys
import traceback
import multiprocessing
import unittest
from typing import Dict, Any

# Attempt to import BigCodeBench data loader
try:
    from bigcodebench.data import get_bigcodebench
except ImportError:
    print("ERROR: bigcodebench package not available. Run inside the SIF container.")
    sys.exit(1)


def run_test(task_id: str, solution_code: str, test_code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute a solution + test harness in a subprocess with timeout.
    Returns a dict with pass/fail status and optional error message.
    """
    def _worker(solution_code: str, test_code: str, result_queue):
        try:
            exec_globals = {}
            # Execute the solution code
            exec(solution_code, exec_globals)
            # Execute the test harness (defines the unittest.TestCase class)
            exec(test_code, exec_globals)

            # Find all unittest.TestCase classes defined in the exec_globals
            suite = unittest.TestSuite()
            loader = unittest.TestLoader()
            test_cases_found = False

            for name, value in list(exec_globals.items()):
                if isinstance(value, type) and issubclass(value, unittest.TestCase) and value is not unittest.TestCase:
                    suite.addTests(loader.loadTestsFromTestCase(value))
                    test_cases_found = True

            if not test_cases_found:
                # Fallback for raw asserts
                result_queue.put({"passed": True, "error": None})
                return

            # Run the loaded unit tests
            runner = unittest.TextTestRunner(verbosity=0)
            result = runner.run(suite)

            if result.wasSuccessful():
                result_queue.put({"passed": True, "error": None})
            else:
                # Gather failure traceback summaries
                errors = []
                for test, tb_text in result.failures + result.errors:
                    last_line = tb_text.strip().splitlines()[-1] if tb_text.strip() else "Unknown failure"
                    errors.append(f"{test.id().split('.')[-1]}: {last_line}")
                result_queue.put({"passed": False, "error": "; ".join(errors)})

        except Exception as e:
            result_queue.put({"passed": False, "error": f"{type(e).__name__}: {str(e)}"})

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_worker, args=(solution_code, test_code, result_queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"passed": False, "error": "TimeoutError: Execution exceeded time limit"}

    if not result_queue.empty():
        return result_queue.get()
    else:
        return {"passed": False, "error": "Process exited without result (possible crash)"}


def main():
    parser = argparse.ArgumentParser(description="Local BigCodeBench subset evaluator")
    parser.add_argument("--samples", required=True, help="Path to JSONL samples file")
    parser.add_argument("--timeout", type=int, default=30, help="Per-task timeout in seconds")
    args = parser.parse_args()

    # Load samples
    print(f"Loading samples from {args.samples}...")
    samples = []
    with open(args.samples, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples")

    # Load BigCodeBench problems (ground truth test cases)
    print("Loading BigCodeBench dataset for test cases...")
    problems = get_bigcodebench()
    print(f"Loaded {len(problems)} problems from BigCodeBench")

    # Evaluate each sample
    results = {}
    passed_count = 0
    total_count = len(samples)

    print(f"\n{'='*60}")
    print(f"  Running evaluation on {total_count} tasks")
    print(f"{'='*60}\n")

    for i, sample in enumerate(samples):
        task_id = sample["task_id"]
        solution = sample["solution"]

        if task_id not in problems:
            print(f"[{i+1}/{total_count}] {task_id}: SKIP (not found in dataset)")
            results[task_id] = {"status": "skip", "error": "Task not found in dataset"}
            continue

        problem = problems[task_id]
        test_code = problem.get("test", "")

        if not test_code:
            print(f"[{i+1}/{total_count}] {task_id}: SKIP (no test code)")
            results[task_id] = {"status": "skip", "error": "No test code available"}
            continue

        # Run the test
        result = run_test(task_id, solution, test_code, timeout=args.timeout)

        if result["passed"]:
            passed_count += 1
            status_str = "PASS ✓"
        else:
            status_str = f"FAIL ✗ ({result['error']})"

        print(f"[{i+1}/{total_count}] {task_id}: {status_str}")
        results[task_id] = {
            "status": "pass" if result["passed"] else "fail",
            "error": result["error"]
        }

    # Calculate Pass@1
    pass_at_1 = passed_count / total_count if total_count > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total tasks:  {total_count}")
    print(f"  Passed:       {passed_count}")
    print(f"  Failed:       {total_count - passed_count}")
    print(f"  Pass@1:       {pass_at_1:.1%} ({passed_count}/{total_count})")
    print(f"{'='*60}\n")

    # Save results JSON
    output_path = args.samples.replace(".jsonl", "_eval_results.json")
    eval_output = {
        "pass_at_1": pass_at_1,
        "total": total_count,
        "passed": passed_count,
        "failed": total_count - passed_count,
        "details": results
    }
    with open(output_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()
