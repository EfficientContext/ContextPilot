#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=final_eval_seeds_%j.out
#SBATCH --error=final_eval_seeds_%j.err

echo "============================================="
echo "  ContextPilot Final A/B Evaluation (ELM GPT-5.5) - 3 Seeds"
echo "============================================="

export PYTHONPATH="$(pwd):$PYTHONPATH"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

echo "[0/4] Installing dependencies..."
pip install -r requirements.txt
pip install fastapi uvicorn pydantic aiohttp

# 1. Boot the local ContextPilot Proxy Server in the background
echo "[1/4] Booting ContextPilot Proxy Server on port 8000..."
ELM_BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
PROXY_TARGET_URL="${ELM_BASE_URL%/v1}"
python -m contextpilot.server.http_server --port 8000 --infer-api-url "$PROXY_TARGET_URL" > proxy_server.log 2>&1 &
PROXY_PID=$!

sleep 5

# 2. Run the newly refactored A/B generation pipeline for seeds 0, 1, 2
echo "[2/4] Running A/B API Evaluation Pipeline (Generation)..."
for SEED in 0 1 2; do
    echo "--- Running Generation Pipeline for SEED ${SEED} ---"
    python evaluation/benchmarks/run_bigcodebench_elm_full.py --model gpt-5.5 --concurrency 1 --eval_mode all --seed ${SEED}
done

echo "[3/4] Shutting down Proxy Server to flush telemetry..."
kill -INT $PROXY_PID
sleep 2

# 3. Sandbox Evaluation for Baseline
echo "[4/4] Executing Apptainer Sandbox Evaluations..."

cd evaluation/benchmarks

for SEED in 0 1 2; do
    echo "--- Evaluating Baseline Results (SEED ${SEED}) ---"
    cp results_baseline_gpt-5.5_seed${SEED}.jsonl elm_samples_full.jsonl
    bash run_sandbox_eval_full.sh
    # Rename results so they aren't overwritten
    mv ../sandbox/results/elm_samples_full_eval_results.json ../sandbox/results/baseline_eval_results_seed${SEED}.json

    echo "--- Evaluating With-Plugin Results (SEED ${SEED}) ---"
    cp results_with_plugin_gpt-5.5_seed${SEED}.jsonl elm_samples_full.jsonl
    bash run_sandbox_eval_full.sh
    # Rename results
    mv ../sandbox/results/elm_samples_full_eval_results.json ../sandbox/results/with_plugin_eval_results_seed${SEED}.json
done

# Cleanup the temporary copy
rm -f elm_samples_full.jsonl

cd ../..
echo "============================================="
echo "  Final 3-Seed Evaluation Pipeline Complete!"
echo "============================================="
