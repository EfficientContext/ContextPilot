#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="${CLAUDE_PLUGIN_DATA:?}"
STATE_FILE="$STATE_DIR/session-state.json"
LOG_FILE="$STATE_DIR/contextpilot.log"
LOCK_DIR="$STATE_DIR/.lock.d"

mkdir -p "$STATE_DIR"

# Read hook input from stdin into temp file
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT
cat > "$TMPFILE"

# Extract file path
FILE_PATH=$(jq -r '.tool_input.file_path // empty' < "$TMPFILE")
if [ -z "$FILE_PATH" ]; then
  exit 0
fi

# Get content length from tool_response.file.content
CHAR_COUNT=$(jq -r '.tool_response.file.content // empty' < "$TMPFILE" | wc -c | tr -d ' ')

# Compute content hash
CONTENT_HASH=$(jq -r '.tool_response.file.content // empty' < "$TMPFILE" | shasum -a 256 | cut -c1-16)

# Get file mtime
if [ -f "$FILE_PATH" ]; then
  MTIME=$(stat -f %m "$FILE_PATH" 2>/dev/null || stat -c %Y "$FILE_PATH" 2>/dev/null || echo "0")
else
  MTIME="0"
fi

# Initialize state file if missing
if [ ! -f "$STATE_FILE" ]; then
  echo '{"reads":{},"stats":{"blocked":0,"allowed":0,"charsSaved":0}}' > "$STATE_FILE"
fi

# Atomic update with mkdir lock
TMPOUT="$STATE_DIR/.state-$$.tmp"
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    jq --arg path "$FILE_PATH" \
       --arg hash "$CONTENT_HASH" \
       --arg mtime "$MTIME" \
       --argjson chars "${CHAR_COUNT:-0}" \
      '.reads[$path] = { hash: $hash, mtime: $mtime, charCount: $chars } | .stats.allowed += 1' \
      "$STATE_FILE" > "$TMPOUT" && mv "$TMPOUT" "$STATE_FILE"
    rmdir "$LOCK_DIR" 2>/dev/null || true
    break
  fi
  sleep 0.$((RANDOM % 5 + 1))
done
rmdir "$LOCK_DIR" 2>/dev/null || true
rm -f "$TMPOUT"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [ContextPilot] Tracked: $FILE_PATH ($CHAR_COUNT chars)" >> "$LOG_FILE"
