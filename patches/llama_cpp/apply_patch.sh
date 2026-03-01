#!/usr/bin/env bash
# llama.cpp ContextPilot Eviction Proxy – setup & launch helper (Apple Silicon)
#
# Usage:
#   bash apply_patch.sh                   # start with defaults
#   CONTEXTPILOT_INDEX_URL=http://localhost:8765 bash apply_patch.sh
#   LLAMA_SERVER_URL=http://localhost:9000 PROXY_PORT=9001 bash apply_patch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Detect Apple Silicon ─────────────────────────────────────────────────────────
ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
    IS_APPLE_SILICON=1
    echo "✓ Apple Silicon (arm64) detected"
else
    IS_APPLE_SILICON=0
    echo "⚠  Not running on Apple Silicon ($ARCH) — GPU metrics will be unavailable"
fi

# ── Check Python is native ARM64 (not via Rosetta) ───────────────────────────────
if [ "$IS_APPLE_SILICON" = "1" ]; then
    PY_ARCH="$(python3 -c 'import platform; print(platform.machine())' 2>/dev/null || echo 'unknown')"
    if [ "$PY_ARCH" != "arm64" ]; then
        echo ""
        echo "⚠  Python is running under Rosetta ($PY_ARCH), not native arm64."
        echo "   This works but is slower.  For native ARM64 Python:"
        echo "     brew install python"
        echo "   or use a conda environment created with:"
        echo "     CONDA_SUBDIR=osx-arm64 conda create -n cp python=3.12"
        echo ""
    else
        echo "✓ Python is native arm64"
    fi
fi

# ── Check powermetrics sudo (needed for GPU metrics) ─────────────────────────────
if [ "$IS_APPLE_SILICON" = "1" ]; then
    if sudo -n /usr/bin/powermetrics --version >/dev/null 2>&1; then
        echo "✓ powermetrics passwordless sudo OK (GPU metrics enabled)"
    else
        echo ""
        echo "⚠  powermetrics requires passwordless sudo for GPU metrics."
        echo "   GPU metrics will be skipped.  To enable, add this line to /etc/sudoers"
        echo "   (run: sudo visudo):"
        echo ""
        echo "     $(whoami) ALL=(ALL) NOPASSWD: /usr/bin/powermetrics"
        echo ""
        echo "   Then re-run this script.  The proxy still works without GPU metrics."
        echo ""
    fi
fi

# ── Defaults ────────────────────────────────────────────────────────────────────
LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-http://localhost:8889}"
CONTEXTPILOT_INDEX_URL="${CONTEXTPILOT_INDEX_URL:-http://localhost:8765}"
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
PROXY_PORT="${PROXY_PORT:-8890}"
CHAT_TEMPLATE_FORMAT="${CHAT_TEMPLATE_FORMAT:-llama3}"
LOG_FILE="${LOG_FILE:-query_log.jsonl}"

# ── Install Python dependencies ──────────────────────────────────────────────────
echo "Checking Python dependencies..."
python3 -c "import fastapi, uvicorn, httpx" 2>/dev/null || {
    echo "Installing from requirements.txt..."
    pip3 install -r "$SCRIPT_DIR/requirements.txt"
}
echo "✓ Python dependencies OK"

# ── Print configuration ──────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " llama.cpp ContextPilot Eviction Proxy"
echo "=========================================="
echo "  llama.cpp backend:   $LLAMA_SERVER_URL"
echo "  ContextPilot server: $CONTEXTPILOT_INDEX_URL"
echo "  Proxy listen:        $PROXY_HOST:$PROXY_PORT"
echo "  Chat template:       $CHAT_TEMPLATE_FORMAT"
echo "  Log file:            $LOG_FILE"
echo ""
if [ "$IS_APPLE_SILICON" = "1" ]; then
    echo "Quick-start (Apple Silicon / Metal):"
    echo "  1. Start llama.cpp with Metal GPU offload:"
    echo "     llama-server -m models/your-model-Q4_K_M.gguf \\"
    echo "         --host 0.0.0.0 --port 8889 \\"
    echo "         -ngl 99 \\"
    echo "         --cache-reuse 256 --parallel 4 \\"
    echo "         -c 32768"
    echo ""
    echo "     (install via: brew install llama.cpp)"
else
    echo "Quick-start:"
    echo "  1. Start llama.cpp:"
    echo "     llama-server -m model.gguf --port 8889 --cache-reuse 256 --parallel 4"
fi
echo ""
echo "  2. Start ContextPilot index server:"
echo "     python3 -m contextpilot.server.http_server --port 8765 \\"
echo "         --infer-api-url http://localhost:$PROXY_PORT"
echo ""
echo "  3. Point your OpenAI client at http://localhost:8765"
echo "=========================================="
echo ""

# ── Launch proxy ────────────────────────────────────────────────────────────────
export LLAMA_SERVER_URL
export CONTEXTPILOT_INDEX_URL
export PROXY_HOST
export PROXY_PORT
export CHAT_TEMPLATE_FORMAT
export LOG_FILE

exec python3 "$SCRIPT_DIR/eviction_proxy.py"
