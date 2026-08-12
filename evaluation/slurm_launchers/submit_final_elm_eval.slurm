#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=final_eval_%j.out
#SBATCH --error=final_eval_%j.err

echo "============================================="
echo "  ContextPilot Final A/B Evaluation (ELM GPT-5.5)"
echo "============================================="

export PYTHONPATH="$(pwd):$PYTHONPATH"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

# Install missing dependencies on compute node before starting the proxy
echo "[0/4] Installing dependencies..."
pip install -r requirements.txt
pip install fastapi uvicorn pydantic aiohttp

# 1. Boot the local ContextPilot Proxy Server in the background
echo "[1/4] Booting ContextPilot Proxy Server on port 8000..."
# Determine ELM API Base URL (stripping trailing /v1 if present for the proxy config)
ELM_BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
PROXY_TARGET_URL="${ELM_BASE_URL%/v1}"
python -m contextpilot.server.http_server --port 8000 --infer-api-url "$PROXY_TARGET_URL" > proxy_server.log 2>&1 &
PROXY_PID=$!

# Wait briefly for the server to be ready
sleep 5

# 2. Run the newly refactored A/B generation pipeline
echo "[2/4] Running A/B API Evaluation Pipeline (Generation)..."
# Concurrency strictly set to 1 to respect ELM API Rate Limit policy
python evaluation/benchmarks/run_bigcodebench_elm_full.py --model gpt-5.5 --concurrency 1

echo "[3/4] Shutting down Proxy Server to flush telemetry..."
kill -INT $PROXY_PID
sleep 2

# 3. Sandbox Evaluation for Baseline
echo "[4/4] Executing Apptainer Sandbox Evaluations..."

# The sandbox script is hardcoded to look for "elm_samples_full.jsonl".
# We will temporarily copy each result set to that filename and run the sandbox.
cd evaluation/benchmarks

echo "--- Evaluating Baseline Results ---"
cp results_baseline_gpt-5.5.jsonl elm_samples_full.jsonl
bash run_sandbox_eval_full.sh
# Rename results so they aren't overwritten
mv ../sandbox/results/elm_samples_full_eval_results.json ../sandbox/results/baseline_eval_results.json

echo "--- Evaluating With-Plugin Results ---"
cp results_with_plugin_gpt-5.5.jsonl elm_samples_full.jsonl
bash run_sandbox_eval_full.sh
# Rename results
mv ../sandbox/results/elm_samples_full_eval_results.json ../sandbox/results/with_plugin_eval_results.json

# Cleanup the temporary copy
rm elm_samples_full.jsonl

cd ../..
echo "============================================="
echo "  Final Evaluation Pipeline Complete!"
echo "============================================="
