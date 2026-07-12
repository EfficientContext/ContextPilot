import argparse
import time
import requests
import json
import signal
import sys
import re

running = True

def signal_handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_metrics(metrics_text):
    # Search for vllm:gpu_cache_usage_perc{...} <value>
    # Format typically: vllm:gpu_cache_usage_perc{model_name="Qwen/Qwen2.5-7B-Instruct-AWQ"} 0.05
    peak_val = 0.0
    for line in metrics_text.split('\n'):
        if line.startswith('vllm:gpu_cache_usage_perc') or line.startswith('vllm_gpu_cache_usage_perc') or line.startswith('vllm:kv_cache_usage_perc') or line.startswith('vllm_kv_cache_usage_perc'):
            try:
                # Extract the float value at the end of the line
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    val = float(parts[1])
                    if val > peak_val:
                        peak_val = val
            except ValueError:
                pass
    return peak_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--interval", type=float, default=0.05)
    args = parser.parse_args()

    url = f"http://localhost:{args.port}/metrics"
    peak_cache_usage = 0.0
    
    print(f"Metrics logger started, polling {url} every {args.interval}s")

    while running:
        try:
            response = requests.get(url, timeout=2.0)
            if response.status_code == 200:
                with open("metrics_dump.txt", "w") as dump_f:
                    dump_f.write(response.text)
                current_usage = parse_metrics(response.text)
                if current_usage > peak_cache_usage:
                    peak_cache_usage = current_usage
        except Exception:
            # vLLM might be starting up or overloaded
            pass
        
        time.sleep(args.interval)

    # Save final peak usage
    result = {
        "peak_gpu_cache_usage_perc": peak_cache_usage,
        "peak_gpu_cache_usage_human": f"{peak_cache_usage * 100:.2f}%"
    }
    
    with open(args.output, "w") as f:
        json.dump(result, f, indent=4)
        
    print(f"Metrics logger stopped. Peak GPU cache usage: {result['peak_gpu_cache_usage_human']}")

if __name__ == "__main__":
    main()
