#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

LIMIT=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --limit) LIMIT="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

echo "============================================="
echo "  Dynamic Pruning Hyperparameter Ablation"
echo "  Limit: $LIMIT"
echo "============================================="

echo "[1/3] Booting ContextPilot Proxy Server..."
python -m contextpilot.server.http_server --port 8000 --infer-api-url "https://api.openai.com" > proxy_ablation.log 2>&1 &
PROXY_PID=$!
sleep 5

THRESHOLDS=(0.1 0.2 0.4 0.5)

echo "[2/3] Running Python Ablation Iterations..."
for THRESHOLD in "${THRESHOLDS[@]}"; do
    echo "--- Running Threshold: $THRESHOLD ---"
    python evaluation/benchmarks/run_bigcodebench.py \
        --model gpt-5.5 \
        --api_base "https://api.openai.com/v1" \
        --concurrency 5 \
        --limit "$LIMIT" \
        --plugins "all" \
        --eval_mode with_plugin \
        --threshold "$THRESHOLD"
        
    mv evaluation/benchmarks/results_with_plugin_gpt-5.5.jsonl evaluation/benchmarks/results_ablation_${THRESHOLD}_with_plugin_gpt-5.5.jsonl
done

echo "Shutting down Proxy Server to flush telemetry..."
kill -INT $PROXY_PID
sleep 2
cat proxy_ablation.log

echo "[3/3] Sandbox Evaluation..."
for THRESHOLD in "${THRESHOLDS[@]}"; do
    echo "--- Evaluating Sandbox for Threshold: $THRESHOLD ---"
    cp evaluation/benchmarks/results_ablation_${THRESHOLD}_with_plugin_gpt-5.5.jsonl evaluation/benchmarks/elm_samples_full.jsonl
    cd evaluation/benchmarks && bash run_sandbox_eval_full.sh && cd ../..
done

echo "Pipeline Complete!"
