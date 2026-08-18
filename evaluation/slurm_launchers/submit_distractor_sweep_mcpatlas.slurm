#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=distractor_sweep_mcpatlas_%j.out
#SBATCH --error=distractor_sweep_mcpatlas_%j.err

echo "============================================="
echo "  ContextPilot MCP-Atlas Distractor Sweep"
echo "============================================="

export PYTHONPATH="$(pwd):$PYTHONPATH"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

ELM_BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
PROXY_TARGET_URL="${ELM_BASE_URL%/v1}"

echo "[0/3] Installing dependencies..."
pip install -r requirements.txt
pip install fastapi uvicorn pydantic aiohttp datasets



echo "[2/3] Running Sweep..."
for R in 0.0 0.25 0.5 0.75 1.0; do
    echo "--- Distractor Ratio: $R ---"
    python evaluation/benchmarks/run_mcpatlas_eval.py --model gpt-5.5 --api_base "$ELM_BASE_URL" --api_key "$OPENAI_API_KEY" --concurrency 1 --eval_mode with_plugin --distractor_ratio $R
done



echo "============================================="
echo "  Pipeline Complete!"
echo "============================================="
