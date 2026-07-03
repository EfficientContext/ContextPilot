#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=test_mcpatlas_deepseek_%j.out
#SBATCH --error=test_mcpatlas_deepseek_%j.err

echo "============================================="
echo "  ContextPilot MCP-Atlas Test DeepSeek (LIMIT 5)"
echo "============================================="

export PYTHONPATH="$(pwd):$PYTHONPATH"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

export BASE_URL="https://api.deepseek.com/v1"
export OPENAI_API_KEY=$DEEPSEEK_API_KEY
PROXY_TARGET_URL="${BASE_URL%/v1}"

echo "[0/3] Installing dependencies..."
pip install -r requirements.txt
pip install fastapi uvicorn pydantic aiohttp datasets

echo "[1/3] Booting ContextPilot Proxy Server for Baseline..."
python -m contextpilot.server.http_server --port 8000 --infer-api-url "$PROXY_TARGET_URL" > proxy_test_mcpatlas_deepseek_baseline.log 2>&1 &
PROXY_PID=$!

echo "Waiting for Proxy Server to initialize (may take ~45s for PyTorch to load)..."
MAX_RETRIES=300
RETRY_COUNT=0
while ! curl -s http://localhost:8000/health > /dev/null; do
    if ! kill -0 $PROXY_PID 2>/dev/null; then
        echo "Proxy server crashed before starting!"
        exit 1
    fi
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Timeout waiting for proxy server."
        exit 1
    fi
done
echo "Proxy Server is ready!"

echo "[2/3] Running Baseline Evaluation (LIMIT 5)..."
python evaluation/benchmarks/run_mcpatlas_eval.py --model deepseek-v4-pro --api_base "$BASE_URL" --api_key "$OPENAI_API_KEY" --concurrency 20 --limit 5 --eval_mode baseline

echo "Shutting down Proxy Server to flush Baseline telemetry..."
kill -INT $PROXY_PID
sleep 2

echo "=== Baseline Proxy Server Telemetry Logs ==="
cat proxy_test_mcpatlas_deepseek_baseline.log

echo "[3/3] Booting ContextPilot Proxy Server for With-Plugin..."
python -m contextpilot.server.http_server --port 8000 --infer-api-url "$PROXY_TARGET_URL" > proxy_test_mcpatlas_deepseek_plugin.log 2>&1 &
PROXY_PID=$!

echo "Waiting for Proxy Server to initialize (may take ~45s for PyTorch to load)..."
MAX_RETRIES=300
RETRY_COUNT=0
while ! curl -s http://localhost:8000/health > /dev/null; do
    if ! kill -0 $PROXY_PID 2>/dev/null; then
        echo "Proxy server crashed before starting!"
        exit 1
    fi
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Timeout waiting for proxy server."
        exit 1
    fi
done
echo "Proxy Server is ready!"

echo "Running With-Plugin Evaluation (LIMIT 5)..."
python evaluation/benchmarks/run_mcpatlas_eval.py --model deepseek-v4-pro --api_base "$BASE_URL" --api_key "$OPENAI_API_KEY" --concurrency 20 --limit 5 --eval_mode with_plugin

echo "Shutting down Proxy Server to flush With-Plugin telemetry..."
kill -INT $PROXY_PID
sleep 2

echo "=== With-Plugin Proxy Server Telemetry Logs ==="
cat proxy_test_mcpatlas_deepseek_plugin.log

echo "============================================="
echo "  Test Pipeline Complete!"
echo "============================================="
