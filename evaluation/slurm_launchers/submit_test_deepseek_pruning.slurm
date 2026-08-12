#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=test_deepseek_pruning_%j.out
#SBATCH --error=test_deepseek_pruning_%j.err

echo "============================================="
echo "  DeepSeek Dynamic Pruning Test (LIMIT 5)"
echo "============================================="

export PYTHONPATH="$(pwd):$PYTHONPATH"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

export BASE_URL="https://api.deepseek.com/v1"
export OPENAI_API_KEY=$DEEPSEEK_API_KEY

echo "[0/6] Installing dependencies..."
pip install -r requirements.txt
pip install fastapi uvicorn pydantic aiohttp sentence-transformers

PROXY_TARGET_URL="${BASE_URL%/v1}"

echo "[1/6] Booting ContextPilot Proxy Server for Dynamic Pruning..."
python -m contextpilot.server.http_server --port 8000 --infer-api-url "$PROXY_TARGET_URL" > proxy_test_deepseek_dynamic.log 2>&1 &
PROXY_PID=$!
sleep 5

echo "[2/6] Running Configuration 1: Dynamic Pruning + Skill Filter (LIMIT 5)..."
python evaluation/benchmarks/run_bigcodebench_deepseek_dynamic.py --model deepseek-v4-pro --api_base https://api.deepseek.com/v1 --api_key $OPENAI_API_KEY --concurrency 2 --limit 5

echo "Shutting down Proxy Server to flush telemetry..."
kill -INT $PROXY_PID
sleep 2

echo "=== Dynamic Pruning Proxy Server Telemetry Logs ==="
cat proxy_test_deepseek_dynamic.log

echo "[3/6] Booting ContextPilot Proxy Server for ALL PLUGINS..."
python -m contextpilot.server.http_server --port 8000 --infer-api-url "$PROXY_TARGET_URL" > proxy_test_deepseek_all.log 2>&1 &
PROXY_PID=$!
sleep 5

echo "[4/6] Running Configuration 2: ALL PLUGINS (LIMIT 5)..."
python evaluation/benchmarks/run_bigcodebench_deepseek_all.py --model deepseek-v4-pro --api_base https://api.deepseek.com/v1 --api_key $OPENAI_API_KEY --concurrency 2 --limit 5

echo "Shutting down Proxy Server to flush telemetry..."
kill -INT $PROXY_PID
sleep 2

echo "=== ALL PLUGINS Proxy Server Telemetry Logs ==="
cat proxy_test_deepseek_all.log

echo "[5/6] Executing Apptainer Sandbox Evaluations..."
cd evaluation/benchmarks

echo "--- Evaluating Configuration 1 (Dynamic Pruning) Results ---"
cp results_deepseek_dynamic_with_plugin_deepseek-v4-pro.jsonl elm_samples_full.jsonl
bash run_sandbox_eval_full.sh
mv ../sandbox/results/elm_samples_full_eval_results.json ../sandbox/results/test_deepseek_dynamic_eval_results.json

echo "--- Evaluating Configuration 2 (ALL PLUGINS) Results ---"
cp results_deepseek_all_with_plugin_deepseek-v4-pro.jsonl elm_samples_full.jsonl
bash run_sandbox_eval_full.sh
mv ../sandbox/results/elm_samples_full_eval_results.json ../sandbox/results/test_deepseek_all_eval_results.json

rm elm_samples_full.jsonl
cd ../..

echo "[6/6] Pipeline Complete!"
