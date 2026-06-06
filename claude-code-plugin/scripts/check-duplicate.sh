#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="${CLAUDE_PLUGIN_DATA:?}"
STATE_FILE="$STATE_DIR/session-state.json"
LOG_FILE="$STATE_DIR/contextpilot.log"
LOCK_DIR="$STATE_DIR/.lock.d"

ALLOW='{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow"}}'

mkdir -p "$STATE_DIR"

# Read hook input from stdin into temp file
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT
cat > "$TMPFILE"

# Extract file path
FILE_PATH=$(jq -r '.tool_input.file_path // empty' < "$TMPFILE")
if [ -z "$FILE_PATH" ]; then
  echo "$ALLOW"
  exit 0
fi

# Check if state file exists and has this file
if [ ! -f "$STATE_FILE" ]; then
  echo "$ALLOW"
  exit 0
fi

PREV_MTIME=$(jq -r --arg path "$FILE_PATH" '.reads[$path].mtime // empty' "$STATE_FILE" 2>/dev/null || true)
if [ -z "$PREV_MTIME" ]; then
  echo "$ALLOW"
  exit 0
fi

PREV_CHARS=$(jq -r --arg path "$FILE_PATH" '.reads[$path].charCount // 0' "$STATE_FILE" 2>/dev/null || echo "0")

# Check if file changed on disk
if [ ! -f "$FILE_PATH" ]; then
  echo "$ALLOW"
  exit 0
fi

CURRENT_MTIME=$(stat -f %m "$FILE_PATH" 2>/dev/null || stat -c %Y "$FILE_PATH" 2>/dev/null || echo "0")

if [ "$CURRENT_MTIME" != "$PREV_MTIME" ]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [ContextPilot] Changed, re-read allowed: $FILE_PATH" >> "$LOG_FILE"
  echo "$ALLOW"
  exit 0
fi

# File unchanged — block
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [ContextPilot] Blocked: $FILE_PATH (~$PREV_CHARS chars saved)" >> "$LOG_FILE"

# Update stats with lock
TMPOUT="$STATE_DIR/.state-$$.tmp"
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    jq --argjson chars "${PREV_CHARS:-0}" '.stats.blocked += 1 | .stats.charsSaved += $chars' \
      "$STATE_FILE" > "$TMPOUT" && mv "$TMPOUT" "$STATE_FILE"
    rmdir "$LOCK_DIR" 2>/dev/null || true
    break
  fi
  sleep 0.$((RANDOM % 5 + 1))
done
rmdir "$LOCK_DIR" 2>/dev/null || true
rm -f "$TMPOUT"

jq -n --arg reason "File already in context. Content unchanged ($PREV_CHARS chars). Refer to the earlier read." '{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": $reason
  }
}'
