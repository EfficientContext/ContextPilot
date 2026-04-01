#!/bin/bash
#
# ContextPilot OpenClaw Plugin Benchmark
# Compares token usage and cache hits with and without the plugin
#
# Usage: ./benchmark.sh [num_iterations]
#

set -e

NUM_ITERATIONS=${1:-3}
OPENCLAW_CONFIG="$HOME/.openclaw/openclaw.json"
BACKUP_CONFIG="$HOME/.openclaw/openclaw.json.backup"
GATEWAY_LOG="/tmp/gw-benchmark.log"

# Test that triggers multiple file reads to show dedup benefit
TEST_FILES=(
    "/home/ryan/ContextPilot/openclaw-plugin/src/engine/dedup.ts"
    "/home/ryan/ContextPilot/openclaw-plugin/src/engine/cache-control.ts"
    "/home/ryan/ContextPilot/openclaw-plugin/src/index.ts"
)

echo "=========================================="
echo "ContextPilot OpenClaw Plugin Benchmark"
echo "=========================================="
echo "Iterations: $NUM_ITERATIONS"
echo ""

# Backup config
cp "$OPENCLAW_CONFIG" "$BACKUP_CONFIG"

cleanup() {
    echo ""
    echo "Restoring original config..."
    cp "$BACKUP_CONFIG" "$OPENCLAW_CONFIG"
    rm -f "$BACKUP_CONFIG"
    pkill -9 -f "openclaw gateway" 2>/dev/null || true
}
trap cleanup EXIT

restart_gateway() {
    pkill -9 -f "openclaw gateway" 2>/dev/null || true
    sleep 2
    openclaw gateway > "$GATEWAY_LOG" 2>&1 &
    sleep 5
}

run_multi_read_test() {
    local label=$1

    echo "Running $label test..."
    echo "  Reading ${#TEST_FILES[@]} files multiple times to trigger dedup..."

    # First, read all files
    for f in "${TEST_FILES[@]}"; do
        openclaw agent --agent main --message "Read $f" > /dev/null 2>&1
    done

    # Then read them again (should trigger dedup on second pass)
    for f in "${TEST_FILES[@]}"; do
        openclaw agent --agent main --message "Read $f again and count lines" > /dev/null 2>&1
    done

    echo "  Done."
}

extract_stats() {
    local log_file=$1

    # Extract chars saved
    local chars_saved=$(grep -oP "Chars saved: \K\d+" "$log_file" 2>/dev/null | tail -1 || echo "0")

    # Extract cache stats from usage blocks
    local cache_read=$(grep -oP '"cacheRead": \K\d+' "$log_file" 2>/dev/null | tail -1 || echo "0")
    local cache_write=$(grep -oP '"cacheWrite": \K\d+' "$log_file" 2>/dev/null | tail -1 || echo "0")
    local input_tokens=$(grep -oP '"input": \K\d+' "$log_file" 2>/dev/null | tail -1 || echo "0")

    echo "$chars_saved $cache_read $cache_write $input_tokens"
}

# ==========================================
# Test WITH ContextPilot enabled
# ==========================================
echo "----------------------------------------"
echo "Test 1: WITH ContextPilot enabled"
echo "----------------------------------------"

# Ensure plugin is enabled
python3 << 'PYTHON'
import json
config_path = "$HOME/.openclaw/openclaw.json".replace("$HOME", __import__("os").environ["HOME"])
with open(config_path, 'r') as f:
    config = json.load(f)
if 'plugins' not in config:
    config['plugins'] = {}
if 'slots' not in config['plugins']:
    config['plugins']['slots'] = {}
config['plugins']['slots']['contextEngine'] = 'contextpilot'
if 'entries' not in config['plugins']:
    config['plugins']['entries'] = {}
if 'contextpilot' not in config['plugins']['entries']:
    config['plugins']['entries']['contextpilot'] = {}
config['plugins']['entries']['contextpilot']['enabled'] = True
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
PYTHON

restart_gateway
run_multi_read_test "WITH_CONTEXTPILOT"

WITH_STATS=$(extract_stats "$GATEWAY_LOG")
WITH_CHARS=$(echo $WITH_STATS | cut -d' ' -f1)
WITH_CACHE_READ=$(echo $WITH_STATS | cut -d' ' -f2)
WITH_CACHE_WRITE=$(echo $WITH_STATS | cut -d' ' -f3)
WITH_INPUT=$(echo $WITH_STATS | cut -d' ' -f4)

echo ""
echo "  Chars saved by dedup: $WITH_CHARS"
echo "  Cache read tokens: $WITH_CACHE_READ"
echo "  Cache write tokens: $WITH_CACHE_WRITE"
echo "  Input tokens: $WITH_INPUT"

# ==========================================
# Test WITHOUT ContextPilot (disabled)
# ==========================================
echo ""
echo "----------------------------------------"
echo "Test 2: WITHOUT ContextPilot (disabled)"
echo "----------------------------------------"

# Disable the plugin
python3 << 'PYTHON'
import json
config_path = "$HOME/.openclaw/openclaw.json".replace("$HOME", __import__("os").environ["HOME"])
with open(config_path, 'r') as f:
    config = json.load(f)
if 'plugins' in config:
    if 'slots' in config['plugins']:
        config['plugins']['slots'].pop('contextEngine', None)
    if 'entries' in config['plugins'] and 'contextpilot' in config['plugins']['entries']:
        config['plugins']['entries']['contextpilot']['enabled'] = False
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
PYTHON

restart_gateway
run_multi_read_test "WITHOUT_CONTEXTPILOT"

WITHOUT_STATS=$(extract_stats "$GATEWAY_LOG")
WITHOUT_CHARS=$(echo $WITHOUT_STATS | cut -d' ' -f1)
WITHOUT_CACHE_READ=$(echo $WITHOUT_STATS | cut -d' ' -f2)
WITHOUT_CACHE_WRITE=$(echo $WITHOUT_STATS | cut -d' ' -f3)
WITHOUT_INPUT=$(echo $WITHOUT_STATS | cut -d' ' -f4)

echo ""
echo "  Chars saved by dedup: $WITHOUT_CHARS (expected: 0)"
echo "  Cache read tokens: $WITHOUT_CACHE_READ"
echo "  Cache write tokens: $WITHOUT_CACHE_WRITE"
echo "  Input tokens: $WITHOUT_INPUT"

# ==========================================
# Results Summary
# ==========================================
echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
echo ""
echo "                        WITH      WITHOUT"
echo "                     ContextPilot  Plugin"
echo "----------------------------------------"
printf "Chars deduped:       %8s    %8s\n" "$WITH_CHARS" "$WITHOUT_CHARS"
printf "Cache read tokens:   %8s    %8s\n" "$WITH_CACHE_READ" "$WITHOUT_CACHE_READ"
printf "Cache write tokens:  %8s    %8s\n" "$WITH_CACHE_WRITE" "$WITHOUT_CACHE_WRITE"
printf "Input tokens:        %8s    %8s\n" "$WITH_INPUT" "$WITHOUT_INPUT"
echo ""

if [ "$WITH_CHARS" -gt "0" ]; then
    echo "ContextPilot deduplication saved $WITH_CHARS characters"
    # Rough estimate: 4 chars per token
    tokens_saved=$((WITH_CHARS / 4))
    echo "Estimated token savings: ~$tokens_saved tokens"
fi
