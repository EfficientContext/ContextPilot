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

echo "=========================================================="
echo "  ContextPilot Quantized AWQ Evaluation (Local vLLM)"
echo "  Limit: $LIMIT"
echo "=========================================================="

export TMPDIR="$(pwd)/tmp"
export TMP="$(pwd)/tmp"
mkdir -p "$TMPDIR"

PORT=$(shuf -i 15000-20000 -n 1)
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-AWQ"

echo "[1/3] Booting Local vLLM Engine on port $PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$PORT" \
    --gpu-memory-utilization 0.8 \
    --quantization awq \
    --max-model-len 4096 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > vllm_quantized.log 2>&1 &
VLLM_PID=$!
sleep 30 # Give vLLM time to load weights into VRAM

echo "[2/3] Booting ContextPilot Proxy Server..."
python -m contextpilot.server.http_server --port 8000 --infer-api-url "http://localhost:$PORT" > proxy_quantized.log 2>&1 &
PROXY_PID=$!
sleep 5

echo "[3/3] Running Python Evaluation Script..."
python evaluation/benchmarks/run_mcpatlas.py \
    --model "$MODEL_NAME" \
    --api_base "http://localhost:$PORT/v1" \
    --concurrency 5 \
    --limit "$LIMIT" \
    --plugins "all" \
    --eval_mode all

echo "Shutting down services..."
kill -INT $PROXY_PID
kill -INT $VLLM_PID
sleep 2

cat proxy_quantized.log
echo "Pipeline Complete!"
