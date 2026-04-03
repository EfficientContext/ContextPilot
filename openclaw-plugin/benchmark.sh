#!/bin/bash
#
# ContextPilot Token Usage Benchmark
# Compares prefill/input tokens with and without the plugin
#

set -e

OPENCLAW_CONFIG="$HOME/.openclaw/openclaw.json"
BACKUP_CONFIG="$HOME/.openclaw/openclaw.json.bak"
LOG_WITH="/tmp/gw-with-cp.log"
LOG_WITHOUT="/tmp/gw-without-cp.log"

TEST_FILE="/home/ryan/ContextPilot/openclaw-plugin/src/engine/dedup.ts"

echo "=========================================="
echo "ContextPilot Token Usage Benchmark"
echo "=========================================="

# Backup config
cp "$OPENCLAW_CONFIG" "$BACKUP_CONFIG"

cleanup() {
    echo ""
    echo "Restoring config..."
    cp "$BACKUP_CONFIG" "$OPENCLAW_CONFIG"
    rm -f "$BACKUP_CONFIG"
    openclaw gateway stop 2>/dev/null || pkill -9 -f "openclaw" 2>/dev/null || true
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
    pkill -9 -f "openclaw" 2>/dev/null || true
    sleep 3
    echo "  Starting gateway..."
    openclaw gateway > "$logfile" 2>&1 &
    sleep 6
    if ! pgrep -f "openclaw" > /dev/null; then
        echo "  ERROR: Gateway failed to start"
        cat "$logfile" | tail -10
        exit 1
    fi
    echo "  Gateway running."
}

run_test_sequence() {
    echo "  Reading file 3 times to build up context..."
    timeout 60 openclaw agent --agent main --message "Read $TEST_FILE and count functions" > /dev/null 2>&1 || true
    timeout 60 openclaw agent --agent main --message "Read $TEST_FILE again" > /dev/null 2>&1 || true
    timeout 60 openclaw agent --agent main --message "Read $TEST_FILE one more time and summarize" > /dev/null 2>&1 || true
    echo "  Done."
}

extract_last_usage() {
    local logfile=$1
    # Find the last complete usage block and extract values
    local input=$(grep '"input":' "$logfile" 2>/dev/null | tail -1 | grep -oP '\d+' || echo "0")
    local cache_read=$(grep '"cacheRead":' "$logfile" 2>/dev/null | tail -1 | grep -oP '\d+' || echo "0")
    local cache_write=$(grep '"cacheWrite":' "$logfile" 2>/dev/null | tail -1 | grep -oP '\d+' || echo "0")
    echo "$input $cache_read $cache_write"
}

extract_chars_saved() {
    local logfile=$1
    # Look for ContextPilot stats line
    grep "Stats:" "$logfile" 2>/dev/null | tail -1 | grep -oP '\d+(?= chars saved)' || echo "0"
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

WITH_USAGE=$(extract_last_usage "$LOG_WITH")
WITH_INPUT=$(echo $WITH_USAGE | cut -d' ' -f1)
WITH_CACHE_READ=$(echo $WITH_USAGE | cut -d' ' -f2)
WITH_CACHE_WRITE=$(echo $WITH_USAGE | cut -d' ' -f3)
WITH_CHARS=$(extract_chars_saved "$LOG_WITH")

echo ""
echo "  Results:"
echo "    Input tokens:  $WITH_INPUT"
echo "    Cache read:    $WITH_CACHE_READ"
echo "    Cache write:   $WITH_CACHE_WRITE"
echo "    Chars deduped: $WITH_CHARS"

# ==========================================
# Test WITHOUT ContextPilot
# ==========================================
echo ""
echo "Test 2: WITHOUT ContextPilot"
echo "----------------------------------------"
disable_contextpilot
restart_gateway "$LOG_WITHOUT"
run_test_sequence

WITHOUT_USAGE=$(extract_last_usage "$LOG_WITHOUT")
WITHOUT_INPUT=$(echo $WITHOUT_USAGE | cut -d' ' -f1)
WITHOUT_CACHE_READ=$(echo $WITHOUT_USAGE | cut -d' ' -f2)
WITHOUT_CACHE_WRITE=$(echo $WITHOUT_USAGE | cut -d' ' -f3)

echo ""
echo "  Results:"
echo "    Input tokens:  $WITHOUT_INPUT"
echo "    Cache read:    $WITHOUT_CACHE_READ"
echo "    Cache write:   $WITHOUT_CACHE_WRITE"
echo "    Chars deduped: 0 (plugin disabled)"

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "COMPARISON"
echo "=========================================="
echo ""
printf "%-20s %12s %12s\n" "" "WITH CP" "WITHOUT CP"
printf "%-20s %12s %12s\n" "--------------------" "------------" "------------"
printf "%-20s %12s %12s\n" "Input tokens" "$WITH_INPUT" "$WITHOUT_INPUT"
printf "%-20s %12s %12s\n" "Cache read" "$WITH_CACHE_READ" "$WITHOUT_CACHE_READ"
printf "%-20s %12s %12s\n" "Cache write" "$WITH_CACHE_WRITE" "$WITHOUT_CACHE_WRITE"
printf "%-20s %12s %12s\n" "Chars deduped" "$WITH_CHARS" "0"
echo ""

# Calculate differences
if [ "$WITH_INPUT" -gt 0 ] && [ "$WITHOUT_INPUT" -gt 0 ]; then
    if [ "$WITH_INPUT" -lt "$WITHOUT_INPUT" ]; then
        diff=$((WITHOUT_INPUT - WITH_INPUT))
        pct=$((diff * 100 / WITHOUT_INPUT))
        echo ">>> ContextPilot reduced input tokens by $diff ($pct% savings)"
    elif [ "$WITH_INPUT" -gt "$WITHOUT_INPUT" ]; then
        diff=$((WITH_INPUT - WITHOUT_INPUT))
        pct=$((diff * 100 / WITHOUT_INPUT))
        echo ">>> ContextPilot added $diff tokens ($pct% overhead)"
    else
        echo ">>> No difference in input tokens"
    fi
fi

if [ "$WITH_CHARS" -gt 0 ]; then
    tokens_saved=$((WITH_CHARS / 4))
    echo ">>> Deduplication removed ~$tokens_saved tokens worth of repeated content"
fi
