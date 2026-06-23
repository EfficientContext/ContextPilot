#!/bin/bash
# =============================================================================
# run_sandbox_eval_cluster.sh
# Adapted from run_sandbox_eval.ps1 for Edinburgh Informatics cluster
# Uses Apptainer (instead of Docker) to run BigCodeBench evaluation
# =============================================================================
set -euo pipefail

# --- Configuration -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SANDBOX_DIR="$SCRIPT_DIR/../sandbox"
SIF_IMAGE="$SANDBOX_DIR/my_project.sif"
RESULTS_DIR="$SANDBOX_DIR/results"
SAMPLES_FILE="$SCRIPT_DIR/elm_samples_full.jsonl"
DEST_FILE="$RESULTS_DIR/elm_samples_full.jsonl"

# BigCodeBench evaluation parameters
# SPLIT: "complete" or "instruct" (BigCodeBench task format)
SPLIT="${BCB_SPLIT:-complete}"
# SUBSET: "full" or "hard"
SUBSET="${BCB_SUBSET:-full}"

echo "============================================="
echo "  ContextPilot Sandbox Evaluation (Cluster)"
echo "============================================="
echo ""
echo "Configuration:"
echo "  SIF Image:    $SIF_IMAGE"
echo "  Results Dir:  $RESULTS_DIR"
echo "  Samples File: $SAMPLES_FILE"
echo "  Split:        $SPLIT"
echo "  Subset:       $SUBSET"
echo ""

# --- Step 1: Validate Prerequisites -----------------------------------------
echo "[1/4] Validating prerequisites..."

if [ ! -f "$SIF_IMAGE" ]; then
    echo "ERROR: SIF image not found at $SIF_IMAGE"
    echo "Build it first with:"
    echo "  cd $SANDBOX_DIR && apptainer build --fakeroot my_project.sif recipe.def"
    exit 1
fi

if [ ! -f "$SAMPLES_FILE" ]; then
    echo "ERROR: elm_samples_full.jsonl not found at $SAMPLES_FILE"
    echo ""
    echo "You need to generate it first by running run_bigcodebench_elm.py."
    echo "This script calls the LLM API (requires OPENAI_API_KEY) to produce code"
    echo "solutions, which are then evaluated inside the sandbox."
    echo ""
    echo "To generate samples:"
    echo "  cd $PROJECT_ROOT"
    echo "  export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\""
    echo "  export OPENAI_API_KEY='your-api-key'"
    echo "  python3 evaluation/benchmarks/run_bigcodebench_elm.py"
    echo ""
    exit 1
fi

echo "  ✓ SIF image found ($(du -h "$SIF_IMAGE" | cut -f1))"
echo "  ✓ Samples file found ($(wc -l < "$SAMPLES_FILE") samples)"
echo ""

# --- Step 2: Prepare Sandbox Environment ------------------------------------
echo "[2/4] Preparing sandbox environment..."

mkdir -p "$RESULTS_DIR"
cp "$SAMPLES_FILE" "$DEST_FILE"
echo "  ✓ Copied elm_samples_full.jsonl to $RESULTS_DIR"
echo ""

# --- Step 3: Run BigCodeBench Evaluation Inside Apptainer --------------------
echo "[3/4] Executing BigCodeBench evaluation inside Apptainer sandbox..."
echo "  Command: bigcodebench.evaluate $SPLIT $SUBSET --samples /app/results/elm_samples_full.jsonl --execution local --pass_k 1"
echo ""

# Bind-mount the results directory into the container at /app/results
# --no-home: don't mount home directory (isolation)
# --bind: mount results directory so we can read input and write output
# --execution local: run tests locally inside container (remote Gradio rejects partial sample sets)
# --pass_k 1: compute only Pass@1 (we have 1 sample per task)
CACHE_DIR="$SANDBOX_DIR/cache"
mkdir -p "$CACHE_DIR"

apptainer exec \
    --no-home \
    --writable-tmpfs \
    --bind "$RESULTS_DIR:/app/results" \
    --bind "$CACHE_DIR:/app/cache" \
    --env "HF_HOME=/app/cache" \
    --env "HF_DATASETS_CACHE=/app/cache/datasets" \
    --env "TMPDIR=/app/cache" \
    --env "XDG_CACHE_HOME=/app/cache" \
    "$SIF_IMAGE" \
    python3 /app/results/run_local_eval.py --samples /app/results/elm_samples_full.jsonl

echo ""

# --- Step 4: Report Results --------------------------------------------------
echo "[4/4] Evaluation complete!"
echo ""
echo "Results saved to: $RESULTS_DIR"
ls -la "$RESULTS_DIR"
echo ""
echo "============================================="
echo "  Evaluation Finished Successfully"
echo "============================================="
