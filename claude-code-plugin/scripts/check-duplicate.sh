#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="${CLAUDE_PLUGIN_DATA:?}"
LOG_FILE="$STATE_DIR/contextpilot.log"
DEBUG_FILE="$STATE_DIR/debug-pre.json"

mkdir -p "$STATE_DIR"

# Capture raw stdin for debugging
cat > "$DEBUG_FILE"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [ContextPilot] PreToolUse captured — see debug-pre.json" >> "$LOG_FILE"

# Always allow during debug
echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow"}}'
