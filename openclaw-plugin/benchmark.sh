#!/bin/bash
#
# ContextPilot Benchmark
# Compares context size with and without ContextPilot
#

set -e

OPENCLAW_CONFIG="$HOME/.openclaw/openclaw.json"
BACKUP_CONFIG="$HOME/.openclaw/openclaw.json.bak"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="${SCRIPT_DIR}/src/engine/dedup.ts"

echo "=========================================="
echo "ContextPilot Benchmark"
echo "=========================================="

# Backup config
cp "$OPENCLAW_CONFIG" "$BACKUP_CONFIG"

cleanup() {
    echo ""
    echo "Restoring config..."
    cp "$BACKUP_CONFIG" "$OPENCLAW_CONFIG"
    rm -f "$BACKUP_CONFIG"
    openclaw gateway stop 2>/dev/null || true
}
trap cleanup EXIT

enable_contextpilot() {
    python3 << 'PYTHON'
import json, os
path = os.path.expanduser("~/.openclaw/openclaw.json")
with open(path) as f: c = json.load(f)
c.setdefault('plugins', {}).setdefault('slots', {})['contextEngine'] = 'contextpilot'
c['plugins'].setdefault('entries', {}).setdefault('contextpilot', {})['enabled'] = True
with open(path, 'w') as f: json.dump(c, f, indent=2)
PYTHON
}

disable_contextpilot() {
    python3 << 'PYTHON'
import json, os
path = os.path.expanduser("~/.openclaw/openclaw.json")
with open(path) as f: c = json.load(f)
if 'plugins' in c:
    c['plugins'].get('slots', {}).pop('contextEngine', None)
    if 'contextpilot' in c['plugins'].get('entries', {}):
        c['plugins']['entries']['contextpilot']['enabled'] = False
with open(path, 'w') as f: json.dump(c, f, indent=2)
PYTHON
}

restart_gateway() {
    local logfile=$1

    # Force stop any existing gateway
    echo "  Stopping existing gateway..."
    openclaw gateway stop 2>/dev/null || true
    sleep 1
    pkill -9 -f "openclaw-gateway" 2>/dev/null || true
    sleep 2

    # Verify it's stopped
    if pgrep -f "openclaw-gateway" > /dev/null 2>&1; then
        echo "  WARNING: Gateway still running, force killing..."
        pkill -9 -f "openclaw-gateway" 2>/dev/null || true
        sleep 2
    fi

    rm -f "$logfile"
    echo "  Starting gateway..."
    nohup openclaw gateway > "$logfile" 2>&1 &

    # Wait for gateway to start
    for i in {1..15}; do
        if pgrep -f "openclaw-gateway" > /dev/null 2>&1; then
            sleep 3
            echo "  Gateway running (PID $(pgrep -f openclaw-gateway | head -1))"
            return 0
        fi
        sleep 1
    done

    echo "ERROR: Gateway failed to start"
    cat "$logfile" | tail -20
    return 1
}

run_test() {
    local session_id="bench-$$-$(date +%s)"
    echo "  Session: $session_id"

    # Read the same file 3 times to trigger dedup
    timeout 60 openclaw agent --agent main --session-id "$session_id" \
        --message "Read $TEST_FILE and count functions" 2>/dev/null || true

    timeout 60 openclaw agent --agent main --session-id "$session_id" \
        --message "Read $TEST_FILE again, list exports" 2>/dev/null || true

    timeout 60 openclaw agent --agent main --session-id "$session_id" \
        --message "Read $TEST_FILE one more time, summarize" 2>/dev/null || true
}

extract_stats() {
    local logfile=$1
    # Try the captured log first
    local stats=$(grep "ContextPilot.*Stats:" "$logfile" 2>/dev/null | tail -1)
    if [ -n "$stats" ]; then
        echo "$stats"
        return
    fi
    # Fall back to main openclaw log
    local today=$(date +%Y-%m-%d)
    local mainlog="/tmp/openclaw/openclaw-${today}.log"
    grep "ContextPilot.*Stats:" "$mainlog" 2>/dev/null | tail -1 || echo ""
}

# ==========================================
# Test WITH ContextPilot
# ==========================================
echo ""
echo "Test: WITH ContextPilot enabled"
echo "----------------------------------------"
enable_contextpilot

LOG_WITH="/tmp/gw-with.log"
if ! restart_gateway "$LOG_WITH"; then
    exit 1
fi

echo "  Running 3 file reads in same session..."
run_test
echo "  Done."

sleep 2
WITH_STATS=$(extract_stats "$LOG_WITH")

if [ -n "$WITH_STATS" ]; then
    # Extract chars saved
    CHARS_SAVED=$(echo "$WITH_STATS" | sed 's/.*Stats:[^,]*, //' | sed 's/ chars saved.*//' | tr -d ',')
    TOKENS_SAVED=$(echo "$WITH_STATS" | grep -oP '~\K[0-9,]+(?= tokens)' | tr -d ',')

    echo ""
    echo "=========================================="
    echo "RESULTS"
    echo "=========================================="
    echo ""
    echo "ContextPilot deduplicated:"
    echo "  Chars saved:    ${CHARS_SAVED:-0}"
    echo "  Tokens saved:   ~${TOKENS_SAVED:-0}"

    if [ -n "$TOKENS_SAVED" ] && [ "$TOKENS_SAVED" -gt 0 ] 2>/dev/null; then
        COST=$(echo "scale=4; $TOKENS_SAVED * 0.003 / 1000" | bc 2>/dev/null || echo "N/A")
        echo "  Est. cost saved: ~\$$COST (at \$3/MTok)"
    fi

    echo ""
    echo "This represents content that was deduplicated"
    echo "across repeated file reads in the session."
else
    echo ""
    echo "No ContextPilot stats found in logs."
    echo ""
    echo "Debug info:"
    echo "  Gateway log: $LOG_WITH"
    echo "  Main log: /tmp/openclaw/openclaw-$(date +%Y-%m-%d).log"
    echo ""
    echo "Gateway log contents:"
    cat "$LOG_WITH" 2>/dev/null | grep -v "^\[" | head -10
    echo ""
    echo "Searching main log for ContextPilot..."
    grep -i "contextpilot" /tmp/openclaw/openclaw-$(date +%Y-%m-%d).log 2>/dev/null | tail -5 || echo "  (none found)"
fi
