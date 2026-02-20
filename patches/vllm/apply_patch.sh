#!/bin/bash
# install_vllm_patches.sh
# Install ContextPilot patches to vLLM

set -e

echo "=============================================="
echo "ContextPilot vLLM Patch Installer"
echo "=============================================="

# Find vLLM installation path
VLLM_PATH=$(python -c "import vllm; print(vllm.__path__[0])" 2>/dev/null)

if [ -z "$VLLM_PATH" ]; then
    echo "Error: vLLM not found. Please install vLLM first."
    exit 1
fi

echo "Found vLLM at: $VLLM_PATH"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if patch files exist
if [ ! -f "$SCRIPT_DIR/block_pool.py" ]; then
    echo "Error: Patch file block_pool.py not found in $SCRIPT_DIR"
    exit 1
fi

# Backup original files
echo ""
echo "Backing up original files..."
BACKUP_DIR="$VLLM_PATH/v1/core/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp "$VLLM_PATH/v1/core/block_pool.py" "$BACKUP_DIR/" 2>/dev/null || true

echo "  Backup saved to: $BACKUP_DIR"

# Copy patched files
echo ""
echo "Installing patches..."
cp "$SCRIPT_DIR/block_pool.py" "$VLLM_PATH/v1/core/"

echo "  ✓ block_pool.py"

echo ""
echo "=============================================="
echo "✅ vLLM patches installed successfully!"
echo "=============================================="
echo ""
echo "To use ContextPilot with vLLM:"
echo ""
echo "  1. Start ContextPilot server:"
echo "     python -m contextpilot.server.http_server --port 8765"
echo ""
echo "  2. Start vLLM with CONTEXTPILOT_INDEX_URL:"
echo "     CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m vllm.entrypoints.openai.api_server \\"
echo "         --model Qwen/Qwen2.5-7B-Instruct --port 8000"
echo ""
echo "To revert changes:"
echo "  cp $BACKUP_DIR/block_pool.py $VLLM_PATH/v1/core/"
