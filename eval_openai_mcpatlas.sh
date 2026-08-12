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

echo "============================================="
echo "  OpenAI (MCP-Atlas) Evaluation"
echo "  Plugins: $PLUGINS | Limit: $LIMIT"
echo "============================================="

echo "[1/2] Booting ContextPilot Proxy Server..."
python -m contextpilot.server.http_server --port 8000 --infer-api-url "https://api.openai.com" > proxy_openai_mcpatlas.log 2>&1 &
PROXY_PID=$!
sleep 5

echo "[2/2] Running Python Evaluation Script..."
python evaluation/benchmarks/run_mcpatlas.py \
    --model gpt-5.5 \
    --api_base "https://api.openai.com/v1" \
    --concurrency 5 \
    --limit "$LIMIT" \
    --plugins "$PLUGINS" \
    --eval_mode all

echo "Shutting down Proxy Server to flush telemetry..."
kill -INT $PROXY_PID
sleep 2
cat proxy_openai_mcpatlas.log

echo "Pipeline Complete!"
