#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="${CLAUDE_PLUGIN_DATA:?}"
STATE_FILE="$STATE_DIR/session-state.json"
LOG_FILE="$STATE_DIR/contextpilot.log"

mkdir -p "$STATE_DIR"

echo '{"reads":{},"stats":{"blocked":0,"allowed":0,"charsSaved":0}}' > "$STATE_FILE"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [ContextPilot] Session initialized" >> "$LOG_FILE"
