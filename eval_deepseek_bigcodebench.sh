#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

LIMIT=0
PLUGINS="all"

while [[ $# -gt 0 ]]; do
  case $1 in
    --limit) LIMIT="$2"; shift 2 ;;
    --plugins) PLUGINS="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

export OPENAI_API_KEY="${DEEPSEEK_API_KEY:-dummy-key}"

echo "============================================="
echo "  DeepSeek (BigCodeBench) Evaluation"
echo "  Plugins: $PLUGINS | Limit: $LIMIT"
echo "============================================="

echo "[1/3] Booting ContextPilot Proxy Server..."
python -m contextpilot.server.http_server --port 8000 --infer-api-url "https://api.deepseek.com" > proxy_deepseek_bigcodebench.log 2>&1 &
PROXY_PID=$!
sleep 5

echo "[2/3] Running Python Evaluation Script..."
python evaluation/benchmarks/run_bigcodebench.py \
    --model deepseek-v4-pro \
    --api_base "https://api.deepseek.com/v1" \
    --api_key "$OPENAI_API_KEY" \
    --concurrency 20 \
    --limit "$LIMIT" \
    --plugins "$PLUGINS" \
    --eval_mode all

echo "Shutting down Proxy Server to flush telemetry..."
kill -INT $PROXY_PID
sleep 2
cat proxy_deepseek_bigcodebench.log

echo "[3/3] Sandbox Evaluation..."
cp evaluation/benchmarks/results_with_plugin_deepseek-v4-pro.jsonl evaluation/benchmarks/elm_samples_full.jsonl
cd evaluation/benchmarks && bash run_sandbox_eval_full.sh
cd ../..

echo "Pipeline Complete!"
