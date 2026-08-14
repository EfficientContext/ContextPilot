#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=final_ablation_elm_%j.out
#SBATCH --error=final_ablation_elm_%j.err

echo "============================================="
echo "  Dynamic Pruning Ablation Evaluation (FULL)"
echo "============================================="

export PYTHONPATH="$(pwd):$PYTHONPATH"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

# Install missing dependencies on compute node before starting the proxy
echo "[0/3] Installing dependencies..."
pip install -r requirements.txt
pip install fastapi uvicorn pydantic aiohttp sentence-transformers

# 1. Boot the local ContextPilot Proxy Server in the background
echo "[1/3] Booting ContextPilot Proxy Server on port 8000..."
ELM_BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
PROXY_TARGET_URL="${ELM_BASE_URL%/v1}"
python -m contextpilot.server.http_server --port 8000 --infer-api-url "$PROXY_TARGET_URL" > proxy_final_ablation.log 2>&1 &
PROXY_PID=$!
sleep 5

THRESHOLDS=(0.6 0.7)

echo "[2/3] Running Ablation Scripts (FULL)..."
for THRESHOLD in "${THRESHOLDS[@]}"; do
    echo "--- Running Threshold: $THRESHOLD ---"
    python evaluation/benchmarks/run_ablation_elm.py --model gpt-5.5 --concurrency 5 --eval_mode with_plugin --threshold $THRESHOLD
done

echo "Shutting down Proxy Server to flush telemetry..."
kill -INT $PROXY_PID
sleep 2

echo "[3/3] Executing Apptainer Sandbox Evaluations..."
cd evaluation/benchmarks

for THRESHOLD in "${THRESHOLDS[@]}"; do
    echo "--- Evaluating Threshold: $THRESHOLD ---"
    cp results_ablation_${THRESHOLD}_with_plugin_gpt-5.5.jsonl elm_samples_full.jsonl
    bash run_sandbox_eval_full.sh
    mv ../sandbox/results/elm_samples_full_eval_results.json ../sandbox/results/final_ablation_${THRESHOLD}_eval_results.json
done

rm elm_samples_full.jsonl
cd ../..

echo "============================================="
echo "  Final Ablation Complete!"
echo "============================================="
