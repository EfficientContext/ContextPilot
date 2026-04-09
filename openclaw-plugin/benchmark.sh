#!/bin/bash
#
# ContextPilot Deduplication Benchmark
# Measures chars saved by context deduplication
#

set -e

OPENCLAW_CONFIG="$HOME/.openclaw/openclaw.json"
BACKUP_CONFIG="$HOME/.openclaw/openclaw.json.bak"
LOG_WITH="/tmp/gw-with-cp.log"
LOG_WITHOUT="/tmp/gw-without-cp.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="${SCRIPT_DIR}/src/engine/dedup.ts"

echo "=========================================="
echo "ContextPilot Deduplication Benchmark"
echo "=========================================="
echo ""
echo "Note: OpenClaw does not expose raw token usage in logs."
echo "This benchmark measures chars saved by deduplication."
echo "Actual token reduction occurs at the LLM provider."

# Backup config
cp "$OPENCLAW_CONFIG" "$BACKUP_CONFIG"

cleanup() {
    echo ""
    echo "Restoring config..."
    cp "$BACKUP_CONFIG" "$OPENCLAW_CONFIG"
    rm -f "$BACKUP_CONFIG"
    openclaw gateway stop 2>/dev/null || true
    pkill -9 -f "openclaw-gateway" 2>/dev/null || true
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
    echo "  Stopping gateway..."
    openclaw gateway stop 2>/dev/null || true
    pkill -9 -f "openclaw-gateway" 2>/dev/null || true
    sleep 2
    # Verify stopped
    if pgrep -f "openclaw-gateway" > /dev/null; then
        echo "  WARNING: Gateway still running, force killing..."
        pkill -9 -f "openclaw-gateway" 2>/dev/null || true
        sleep 2
    fi
    echo "  Starting gateway..."
    openclaw gateway > "$logfile" 2>&1 &
    sleep 5
    if ! pgrep -f "openclaw-gateway" > /dev/null; then
        echo "  ERROR: Gateway failed to start"
        tail -10 "$logfile"
        exit 1
    fi
    echo "  Gateway running (PID $(pgrep -f 'openclaw-gateway'))."
}

run_test_sequence() {
    local session_id="benchmark-$$-$(date +%s)"
    echo "  Session: $session_id"
    echo "  Reading file 3 times in same session..."
    timeout 60 openclaw agent --agent main --session-id "$session_id" --message "Read $TEST_FILE and count the functions" > /dev/null 2>&1 || true
    timeout 60 openclaw agent --agent main --session-id "$session_id" --message "Read $TEST_FILE again and list the exports" > /dev/null 2>&1 || true
    timeout 60 openclaw agent --agent main --session-id "$session_id" --message "Read $TEST_FILE one more time and summarize what it does" > /dev/null 2>&1 || true
    echo "  Done."
}

extract_chars_saved() {
    local logfile=$1
    # Look for ContextPilot stats line - extract chars saved
    # Format: "[ContextPilot] Stats: N requests, X chars saved (~Y tokens, ~$Z)"
    local line=$(grep "ContextPilot.*Stats:" "$logfile" 2>/dev/null | tail -1)
    if [ -z "$line" ]; then
        echo "0"
        return
    fi
    # Extract the number before "chars saved"
    echo "$line" | sed 's/.*Stats:[^,]*, //' | sed 's/ chars saved.*//' | tr -d ','
}

extract_requests() {
    local logfile=$1
    local line=$(grep "ContextPilot.*Stats:" "$logfile" 2>/dev/null | tail -1)
    if [ -z "$line" ]; then
        echo "0"
        return
    fi
    # Extract the number after "Stats:" before "requests"
    echo "$line" | sed 's/.*Stats: //' | sed 's/ requests.*//'
}

# ==========================================
# Test WITH ContextPilot
# ==========================================
echo ""
echo "Test 1: WITH ContextPilot enabled"
echo "----------------------------------------"
enable_contextpilot
restart_gateway "$LOG_WITH"
run_test_sequence

WITH_CHARS=$(extract_chars_saved "$LOG_WITH")
WITH_REQUESTS=$(extract_requests "$LOG_WITH")

echo ""
echo "  Results:"
echo "    Requests processed: $WITH_REQUESTS"
echo "    Chars deduped:      $WITH_CHARS"
if [ "$WITH_CHARS" -gt 0 ] 2>/dev/null; then
    tokens_est=$((WITH_CHARS / 4))
    cost_est=$(echo "scale=4; $tokens_est * 0.003 / 1000" | bc 2>/dev/null || echo "0")
    echo "    Est. tokens saved:  ~$tokens_est"
    echo "    Est. cost saved:    ~\$$cost_est"
fi

# ==========================================
# Test WITHOUT ContextPilot
# ==========================================
echo ""
echo "Test 2: WITHOUT ContextPilot"
echo "----------------------------------------"
disable_contextpilot
restart_gateway "$LOG_WITHOUT"
run_test_sequence

echo ""
echo "  Results:"
echo "    (No ContextPilot stats - plugin disabled)"

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""

if [ -z "$WITH_CHARS" ] || [ "$WITH_CHARS" = "0" ]; then
    echo "No deduplication occurred."
    echo "This is expected if:"
    echo "  - The file wasn't read multiple times in the same session"
    echo "  - The file content was too short to chunk"
    echo "  - Each read was in a fresh session (gateway restart between reads)"
    echo ""
    echo "For best results, run multiple file reads in a single session."
else
    tokens_saved=$((WITH_CHARS / 4))
    cost_saved=$(echo "scale=4; $tokens_saved * 0.003 / 1000" | bc 2>/dev/null || echo "N/A")
    echo "ContextPilot Results:"
    echo "  Requests processed:   $WITH_REQUESTS"
    echo "  Chars deduplicated:   $WITH_CHARS"
    echo "  Est. tokens reduced:  ~$tokens_saved"
    echo "  Est. cost saved:      ~\$$cost_saved (at \$3/MTok input)"
    echo ""
    echo ">>> Deduplication removes repeated file content across tool calls"
    echo ">>> Actual token reduction occurs at the LLM provider"
fi
