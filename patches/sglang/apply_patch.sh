#!/bin/bash
# install_sglang_patches.sh
# Install ContextPilot patches to SGLang

set -e

echo "=============================================="
echo "ContextPilot SGLang Patch Installer"
echo "=============================================="

# Find SGLang installation path
SGLANG_PATH=$(python -c "import sglang; print(sglang.__path__[0])" 2>/dev/null)

if [ -z "$SGLANG_PATH" ]; then
    echo "Error: SGLang not found. Please install SGLang first."
    exit 1
fi

echo "Found SGLang at: $SGLANG_PATH"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if patch files exist
if [ ! -f "$SCRIPT_DIR/cache_init_params.py" ] || \
   [ ! -f "$SCRIPT_DIR/common.py" ] || \
   [ ! -f "$SCRIPT_DIR/radix_cache.py" ]; then
    echo "Error: Patch files not found in $SCRIPT_DIR"
    exit 1
fi

# Backup original files
echo ""
echo "Backing up original files..."
BACKUP_DIR="$SGLANG_PATH/srt/mem_cache/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp "$SGLANG_PATH/srt/mem_cache/cache_init_params.py" "$BACKUP_DIR/" 2>/dev/null || true
cp "$SGLANG_PATH/srt/mem_cache/common.py" "$BACKUP_DIR/" 2>/dev/null || true
cp "$SGLANG_PATH/srt/mem_cache/radix_cache.py" "$BACKUP_DIR/" 2>/dev/null || true

echo "  Backup saved to: $BACKUP_DIR"

# Copy patched files
echo ""
echo "Installing patches..."
cp "$SCRIPT_DIR/cache_init_params.py" "$SGLANG_PATH/srt/mem_cache/"
cp "$SCRIPT_DIR/common.py" "$SGLANG_PATH/srt/mem_cache/"
cp "$SCRIPT_DIR/radix_cache.py" "$SGLANG_PATH/srt/mem_cache/"

echo "  ✓ cache_init_params.py"
echo "  ✓ common.py"
echo "  ✓ radix_cache.py"

echo ""
echo "=============================================="
echo "✅ SGLang patches installed successfully!"
echo "=============================================="
echo ""
echo "To use ContextPilot with SGLang:"
echo ""
echo "  1. Start ContextPilot server:"
echo "     python -m contextpilot.server.http_server --port 8765"
echo ""
echo "  2. Start SGLang with RAGBOOST_INDEX_URL:"
echo "     RAGBOOST_INDEX_URL=http://localhost:8765 python -m sglang.launch_server \\"
echo "         --model-path Qwen/Qwen3-4B --port 30000"
echo ""
echo "To revert changes:"
echo "  cp $BACKUP_DIR/* $SGLANG_PATH/srt/mem_cache/"
