#!/usr/bin/env bash
set -euo pipefail

# ContextPilot + OpenClaw one-click setup
# Usage: bash setup.sh [anthropic|openai]

PROVIDER="${1:-anthropic}"
PORT="${CONTEXTPILOT_PORT:-8765}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Check Python ───────────────────────────────────────────────────────────
info "Checking Python version..."
if ! command -v python3 &>/dev/null; then
    error "Python 3 not found. Install Python 3.10+ first."
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    error "Python 3.10+ required, found $PY_VER"
fi
info "Python $PY_VER OK"

# ── Install ContextPilot ──────────────────────────────────────────────────
info "Installing ContextPilot..."
if python3 -c "import contextpilot" 2>/dev/null; then
    info "ContextPilot already installed, upgrading..."
    pip install --upgrade contextpilot -q
else
    pip install contextpilot -q
fi

# ── Determine backend URL and API type ────────────────────────────────────
case "$PROVIDER" in
    anthropic)
        BACKEND_URL="https://api.anthropic.com"
        API_KEY_VAR="ANTHROPIC_API_KEY"
        API_TYPE="anthropic-messages"
        MODEL_ID="claude-opus-4-6"
        MODEL_NAME="Claude Opus 4.6 (via ContextPilot)"
        CTX_WINDOW=200000
        MAX_TOKENS=32000
        ;;
    openai)
        BACKEND_URL="https://api.openai.com"
        API_KEY_VAR="OPENAI_API_KEY"
        API_TYPE="openai-completions"
        MODEL_ID="gpt-4o"
        MODEL_NAME="GPT-4o (via ContextPilot)"
        CTX_WINDOW=128000
        MAX_TOKENS=16384
        ;;
    *)
        error "Unknown provider: $PROVIDER. Use 'anthropic' or 'openai'."
        ;;
esac

# ── Check API key ─────────────────────────────────────────────────────────
if [ -z "${!API_KEY_VAR:-}" ]; then
    warn "$API_KEY_VAR not set. The proxy will start but requests will fail without a valid key."
    warn "Set it with: export $API_KEY_VAR=your-key"
fi

# ── Generate OpenClaw provider config ─────────────────────────────────────
OPENCLAW_DIR="$HOME/.openclaw"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/contextpilot-provider.json"

cat > "$PATCH_FILE" <<EOF
{
  "models": {
    "providers": {
      "contextpilot-${PROVIDER}": {
        "baseUrl": "http://localhost:${PORT}/v1",
        "apiKey": "\${${API_KEY_VAR}}",
        "api": "${API_TYPE}",
        "headers": {
          "X-ContextPilot-Scope": "all"
        },
        "models": [
          {
            "id": "${MODEL_ID}",
            "name": "${MODEL_NAME}",
            "reasoning": false,
            "input": ["text"],
            "contextWindow": ${CTX_WINDOW},
            "maxTokens": ${MAX_TOKENS}
          }
        ]
      }
    }
  }
}
EOF
info "Provider config written to $PATCH_FILE"

# ── Print next steps ──────────────────────────────────────────────────────
echo ""
info "=== Next Steps ==="
info "1. The proxy will start below on port $PORT"
info "2. In OpenClaw UI → Settings → Models → add a custom provider:"
info "     Name:    contextpilot-${PROVIDER}"
info "     Base URL: http://localhost:${PORT}/v1"
info "     API:     ${API_TYPE}"
info "     Model:   ${MODEL_ID}"
info ""
info "   Or merge $PATCH_FILE into ~/.openclaw/openclaw.json"
info "3. Select '${MODEL_ID}' as your model in OpenClaw"
echo ""
info "Starting ContextPilot proxy on port $PORT -> $BACKEND_URL"
info "Press Ctrl+C to stop."
echo ""

exec python3 -m contextpilot.server.http_server \
    --stateless \
    --port "$PORT" \
    --infer-api-url "$BACKEND_URL"
