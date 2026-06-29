#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=final_deepseek_eval_%j.out
#SBATCH --error=final_deepseek_eval_%j.err

echo "============================================="
echo "  ContextPilot A/B Test DeepSeek (FULL)"
echo "============================================="

export PYTHONPATH="$(pwd):$PYTHONPATH"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

export BASE_URL="https://api.deepseek.com/v1"
export OPENAI_API_KEY=$DEEPSEEK_API_KEY

echo "[0/4] Installing dependencies..."
pip install -r requirements.txt
pip install fastapi uvicorn pydantic aiohttp

PROXY_TARGET_URL="${BASE_URL%/v1}"

echo "[1/4] Booting ContextPilot Proxy Server for Baseline..."
python -m contextpilot.server.http_server --port 8000 --infer-api-url "$PROXY_TARGET_URL" > proxy_final_deepseek_server_baseline.log 2>&1 &
PROXY_PID=$!
sleep 5

echo "[2/4] Running Baseline Evaluation (FULL)..."
python evaluation/benchmarks/run_bigcodebench_elm_full.py --model deepseek-v4-pro --api_base https://api.deepseek.com/v1 --api_key $OPENAI_API_KEY --concurrency 20 --eval_mode baseline

echo "Shutting down Proxy Server to flush Baseline telemetry..."
kill -INT $PROXY_PID
sleep 2

echo "=== Baseline Proxy Server Telemetry Logs ==="
cat proxy_final_deepseek_server_baseline.log

echo "[3/4] Booting ContextPilot Proxy Server for With-Plugin..."
python -m contextpilot.server.http_server --port 8000 --infer-api-url "$PROXY_TARGET_URL" > proxy_final_deepseek_server_plugin.log 2>&1 &
PROXY_PID=$!
sleep 5

echo "[4/4] Running With-Plugin Evaluation (FULL)..."
python evaluation/benchmarks/run_bigcodebench_elm_full.py --model deepseek-v4-pro --api_base https://api.deepseek.com/v1 --api_key $OPENAI_API_KEY --concurrency 20 --eval_mode with_plugin

echo "Shutting down Proxy Server to flush With-Plugin telemetry..."
kill -INT $PROXY_PID
sleep 2

echo "=== With-Plugin Proxy Server Telemetry Logs ==="
cat proxy_final_deepseek_server_plugin.log

echo "Executing Apptainer Sandbox Evaluations..."
cd evaluation/benchmarks

echo "--- Evaluating Baseline Results ---"
cp results_baseline_deepseek-v4-pro.jsonl elm_samples_full.jsonl
bash run_sandbox_eval_full.sh
mv ../sandbox/results/elm_samples_full_eval_results.json ../sandbox/results/baseline_deepseek_eval_results.json

echo "--- Evaluating With-Plugin Results ---"
cp results_with_plugin_deepseek-v4-pro.jsonl elm_samples_full.jsonl
bash run_sandbox_eval_full.sh
mv ../sandbox/results/elm_samples_full_eval_results.json ../sandbox/results/with_plugin_deepseek_eval_results.json

rm elm_samples_full.jsonl
cd ../..

echo "============================================="
echo "  Final Evaluation Pipeline Complete!"
echo "============================================="
