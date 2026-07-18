#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=test_all_plugins_%j.out
#SBATCH --error=test_all_plugins_%j.err

echo "============================================="
echo "  All Plugins A/B Test Evaluation (LIMIT 5)"
echo "============================================="

export PYTHONPATH="$(pwd):$PYTHONPATH"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

# Install missing dependencies on compute node before starting the proxy
echo "[0/4] Installing dependencies..."
pip install -r requirements.txt
pip install fastapi uvicorn pydantic aiohttp sentence-transformers

# 1. Boot the local ContextPilot Proxy Server in the background
echo "[1/4] Booting ContextPilot Proxy Server on port 8000..."
# Determine ELM API Base URL (stripping trailing /v1 if present for the proxy config)
ELM_BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
PROXY_TARGET_URL="${ELM_BASE_URL%/v1}"
python -m contextpilot.server.http_server --port 8000 --infer-api-url "$PROXY_TARGET_URL" > proxy_test_all_plugins.log 2>&1 &
PROXY_PID=$!

# Wait briefly for the server to be ready
sleep 5

# 2. Run the newly refactored A/B generation pipeline (LIMIT 5)
echo "[2/4] Running A/B API Evaluation Pipeline (Generation) with LIMIT 5..."
python evaluation/benchmarks/run_bigcodebench_all_plugins.py --model gpt-5.5 --concurrency 1 --limit 5

echo "[3/4] Shutting down Proxy Server to flush telemetry..."
kill -INT $PROXY_PID
sleep 2

# 3. Sandbox Evaluation for Baseline
echo "[4/4] Executing Apptainer Sandbox Evaluations..."
cd evaluation/benchmarks

echo "--- Evaluating Baseline Results ---"
cp results_all_plugins_baseline_gpt-5.5.jsonl elm_samples_full.jsonl
bash run_sandbox_eval_full.sh
mv ../sandbox/results/elm_samples_full_eval_results.json ../sandbox/results/test_all_plugins_baseline_eval_results.json

echo "--- Evaluating With-Plugin Results ---"
cp results_all_plugins_with_plugin_gpt-5.5.jsonl elm_samples_full.jsonl
bash run_sandbox_eval_full.sh
mv ../sandbox/results/elm_samples_full_eval_results.json ../sandbox/results/test_all_plugins_with_plugin_eval_results.json

rm elm_samples_full.jsonl

cd ../..
echo "============================================="
echo "  Test Pipeline Complete!"
echo "============================================="
