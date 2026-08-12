#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=quantized_eval_%j.out
#SBATCH --error=quantized_eval_%j.err

echo "=========================================================="
echo "  ContextPilot Quantized AWQ Evaluation (vLLM + MCP-Atlas)"
echo "=========================================================="

export PYTHONPATH="$(pwd):$PYTHONPATH"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

# Force JIT compilation to use our local tmp dir instead of system /tmp
export TMPDIR="$(pwd)/tmp"
export TMP="$(pwd)/tmp"

# Generate a dynamic high port to avoid collisions
PORT=$(shuf -i 15000-20000 -n 1)
echo "Generated vLLM Port: $PORT"

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-AWQ"

echo "[0/3] Installing dependencies..."
pip install --no-cache-dir -r requirements.txt 2>/dev/null || true
pip install --no-cache-dir vllm datasets 2>/dev/null || true
pip install --no-cache-dir --upgrade scipy "numpy<2.0" 2>/dev/null || true

# Function to manage vLLM lifecycle
start_vllm() {
    local log_file=$1
    echo "Booting vLLM on port $PORT..."
    # Boot vLLM with 0.4 gpu utilization for tiny footprint
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --port "$PORT" \
        --gpu-memory-utilization 0.8 \
        --quantization awq \
        --max-model-len 4096 \
        --enforce-eager \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        > "$log_file" 2>&1 &
    
    VLLM_PID=$!
    
    echo "Waiting for vLLM to initialize..."
    MAX_RETRIES=300
    RETRY_COUNT=0
    while ! curl -s http://localhost:$PORT/health > /dev/null; do
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "vLLM crashed before starting! Check $log_file"
            exit 1
        fi
        sleep 2
        RETRY_COUNT=$((RETRY_COUNT+1))
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "Timeout waiting for vLLM."
            kill -9 $VLLM_PID
            exit 1
        fi
    done
    echo "vLLM is ready on port $PORT!"
}

# ==========================================
# PHASE 1: Baseline Evaluation
# ==========================================
echo "[1/3] Phase 1: Baseline Evaluation"
start_vllm "vllm_baseline.log"

echo "Starting Metrics Logger for Baseline..."
python scripts/vllm_metrics_logger.py --port $PORT --output baseline_metrics.json &
LOGGER_PID=$!

echo "Running Baseline Evaluation (100 Tasks)..."
# Using concurrency=5 to simulate edge constraints
python evaluation/benchmarks/run_mcpatlas_eval.py \
    --model "$MODEL_NAME" \
    --api_base "http://localhost:$PORT/v1" \
    --api_key "dummy" \
    --concurrency 5 \
    --limit 100 \
    --eval_mode baseline

echo "Stopping Metrics Logger..."
kill -INT $LOGGER_PID
sleep 2

echo "Shutting down vLLM..."
kill -TERM $VLLM_PID
wait $VLLM_PID 2>/dev/null || true
sleep 10
PORT=$(shuf -i 15000-20000 -n 1)

# ==========================================
# PHASE 2: With-Plugin Evaluation
# ==========================================
echo "[2/3] Phase 2: With-Plugin Evaluation"
start_vllm "vllm_plugin.log"

echo "Starting Metrics Logger for With-Plugin..."
python scripts/vllm_metrics_logger.py --port $PORT --output plugin_metrics.json &
LOGGER_PID=$!

echo "Running With-Plugin Evaluation (100 Tasks)..."
python evaluation/benchmarks/run_mcpatlas_eval.py \
    --model "$MODEL_NAME" \
    --api_base "http://localhost:$PORT/v1" \
    --api_key "dummy" \
    --concurrency 5 \
    --limit 100 \
    --eval_mode with_plugin

echo "Stopping Metrics Logger..."
kill -INT $LOGGER_PID
sleep 2

echo "Shutting down vLLM..."
kill -9 $VLLM_PID

echo "[3/3] Displaying Metrics Summaries:"
echo "=== Baseline Peak VRAM ==="
cat baseline_metrics.json
echo ""
echo "=== With-Plugin Peak VRAM ==="
cat plugin_metrics.json
echo ""

echo "=========================================================="
echo "  Evaluation Pipeline Complete!"
echo "=========================================================="
