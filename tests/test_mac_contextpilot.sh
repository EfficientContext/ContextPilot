#!/usr/bin/env bash
# test_mac_contextpilot.sh
# End-to-end MultihopRAG benchmark for Mac / Apple Silicon
# Runs the full ContextPilot pipeline and compares against baseline.
#
# Usage:
#   bash test_mac_contextpilot.sh                        # full run, 100 queries
#   bash test_mac_contextpilot.sh --num-queries 50       # fewer queries
#   bash test_mac_contextpilot.sh --skip-data-prep       # skip steps 1-2 (data already built)
#   bash test_mac_contextpilot.sh --model path/to/model.gguf
#
# Prerequisites:
#   - llama-server in PATH  (brew install llama.cpp  OR  built from source)
#   - Docker running        (for Elasticsearch)
#   - pip install -r requirements-mac.txt && pip install -e .

set -euo pipefail

# ── Configurable defaults ──────────────────────────────────────────────────────
MODEL="${MODEL:-models/Qwen3-8B-Q4_K_M.gguf}"
NUM_QUERIES="${NUM_QUERIES:-100}"
MAX_DOCS="${MAX_DOCS:-10}"
MAX_CHARS_PER_DOC="${MAX_CHARS_PER_DOC:-800}"
CORPUS_PATH="${CORPUS_PATH:-mulhoprag_corpus.jsonl}"
QUERY_PATH="${QUERY_PATH:-mulhoprag_queries.jsonl}"
BM25_PATH="${BM25_PATH:-mulhoprag_bm25_top20.jsonl}"
REORDERED_PATH="${REORDERED_PATH:-mulhoprag_reordered.jsonl}"
PROXY_LOG="${PROXY_LOG:-query_log.jsonl}"
OUTPUT_CP="${OUTPUT_CP:-results_contextpilot.jsonl}"
OUTPUT_BASELINE="${OUTPUT_BASELINE:-results_baseline.jsonl}"

LLAMA_PORT=8889
PROXY_PORT=8890
CP_PORT=8765

SKIP_DATA_PREP=false

# ── Parse CLI args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)            MODEL="$2";          shift 2 ;;
        --num-queries)      NUM_QUERIES="$2";    shift 2 ;;
        --max-docs)         MAX_DOCS="$2";       shift 2 ;;
        --skip-data-prep)   SKIP_DATA_PREP=true; shift   ;;
        --corpus-path)      CORPUS_PATH="$2";    shift 2 ;;
        --reordered-path)   REORDERED_PATH="$2"; shift 2 ;;
        --proxy-log)        PROXY_LOG="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Helpers ────────────────────────────────────────────────────────────────────
PIDS=()

cleanup() {
    echo ""
    echo "── Cleaning up background services ──────────────────────────────────"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping PID $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    echo "  Stopping Elasticsearch container (if started by this script)"
    docker stop contextpilot-es 2>/dev/null || true
}
trap cleanup EXIT

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
hr()   { echo "══════════════════════════════════════════════════════════════"; }

wait_for_http() {
    local url="$1" label="$2" timeout="${3:-60}"
    log "Waiting for $label at $url ..."
    local i=0
    until curl -sf "$url" >/dev/null 2>&1; do
        sleep 2; i=$((i+2))
        if [[ $i -ge $timeout ]]; then
            die "$label did not become ready within ${timeout}s (url: $url)"
        fi
    done
    log "$label is ready."
}

# ── Step 0: Prerequisites ──────────────────────────────────────────────────────
hr
log "Step 0 — Checking prerequisites"

command -v python  >/dev/null 2>&1 || die "python not found in PATH"
command -v docker  >/dev/null 2>&1 || die "docker not found in PATH"
command -v curl    >/dev/null 2>&1 || die "curl not found in PATH"

if ! command -v llama-server >/dev/null 2>&1; then
    die "llama-server not found in PATH.  Install with: brew install llama.cpp"
fi

python -c "import contextpilot" 2>/dev/null || \
    die "contextpilot not importable. Run: pip install -e ."

python -c "import elasticsearch" 2>/dev/null || \
    die "elasticsearch package not found. Run: pip install -r requirements-mac.txt"

if [[ ! -f "$MODEL" ]]; then
    die "Model file not found: $MODEL\nSet MODEL=path/to/your.gguf or use --model"
fi

log "All prerequisites satisfied."
log "  Model:       $MODEL"
log "  Queries:     $NUM_QUERIES"
log "  Max docs:    $MAX_DOCS  (chars/doc: $MAX_CHARS_PER_DOC)"

# ── Step 1: Start Elasticsearch ────────────────────────────────────────────────
if [[ "$SKIP_DATA_PREP" == false ]]; then
    hr
    log "Step 1 — Starting Elasticsearch"

    # Reuse existing container if already running
    if docker ps --format '{{.Names}}' | grep -q '^contextpilot-es$'; then
        log "Elasticsearch container 'contextpilot-es' already running — reusing."
    else
        docker run -d --name contextpilot-es \
            -p 9200:9200 \
            -e "discovery.type=single-node" \
            -e "xpack.security.enabled=false" \
            -e "xpack.security.http.ssl.enabled=false" \
            -e "xpack.security.transport.ssl.enabled=false" \
            -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
            docker.elastic.co/elasticsearch/elasticsearch:8.18.2
    fi

    wait_for_http "http://localhost:9200/_cluster/health" "Elasticsearch" 120

    # ── Step 2: Build BM25 retrieval data ─────────────────────────────────────
    hr
    log "Step 2 — Building BM25 retrieval data (downloads MultiHopRAG dataset)"
    log "  This can take several minutes on the first run."

    python examples/construct_rag_data/multihopRAG_bm25.py \
        --corpus_path "$CORPUS_PATH" \
        --query_path  "$QUERY_PATH" \
        --output_path "$BM25_PATH"

    log "BM25 retrieval data written to $BM25_PATH"

    # ── Step 3: Reorder with ContextPilot ─────────────────────────────────────
    hr
    log "Step 3 — Reordering contexts with ContextPilot (offline batch)"

    python examples/offline/prepare_batch.py \
        --context_path "$BM25_PATH" \
        --output_path  "$REORDERED_PATH"

    log "Reordered data written to $REORDERED_PATH"
else
    log "Skipping data preparation (--skip-data-prep set)."
    [[ -f "$REORDERED_PATH" ]] || die "Reordered file not found: $REORDERED_PATH"
    [[ -f "$CORPUS_PATH"    ]] || die "Corpus file not found: $CORPUS_PATH"
fi

# ── Step 4: Start llama-server ─────────────────────────────────────────────────
hr
log "Step 4 — Starting llama-server on port $LLAMA_PORT"

llama-server -m "$MODEL" \
    --host 0.0.0.0 --port "$LLAMA_PORT" \
    -ngl 99 --cache-reuse 256 --parallel 4 -c 32768 \
    > /tmp/llama_server.log 2>&1 &
LLAMA_PID=$!
PIDS+=("$LLAMA_PID")
log "llama-server PID=$LLAMA_PID  (log: /tmp/llama_server.log)"

wait_for_http "http://localhost:${LLAMA_PORT}/health" "llama-server" 120

# ── Step 5: Start eviction proxy ───────────────────────────────────────────────
hr
log "Step 5 — Starting eviction proxy on port $PROXY_PORT"

LLAMA_SERVER_URL="http://localhost:${LLAMA_PORT}" \
CONTEXTPILOT_INDEX_URL="http://localhost:${CP_PORT}" \
PROXY_PORT="$PROXY_PORT" \
CHAT_TEMPLATE_FORMAT=chatml \
LOG_FILE="$PROXY_LOG" \
    python patches/llama_cpp/eviction_proxy.py \
    > /tmp/eviction_proxy.log 2>&1 &
PROXY_PID=$!
PIDS+=("$PROXY_PID")
log "Eviction proxy PID=$PROXY_PID  (log: /tmp/eviction_proxy.log)"

wait_for_http "http://localhost:${PROXY_PORT}/stats" "Eviction proxy" 30

# ── Step 6: Start ContextPilot HTTP server ─────────────────────────────────────
hr
log "Step 6 — Starting ContextPilot HTTP server on port $CP_PORT"

python -m contextpilot.server.http_server \
    --port "$CP_PORT" \
    --infer-api-url "http://localhost:${PROXY_PORT}" \
    > /tmp/contextpilot_server.log 2>&1 &
CP_PID=$!
PIDS+=("$CP_PID")
log "ContextPilot HTTP server PID=$CP_PID  (log: /tmp/contextpilot_server.log)"

wait_for_http "http://localhost:${CP_PORT}/health" "ContextPilot server" 30

# ── Step 7: Run ContextPilot benchmark ────────────────────────────────────────
hr
log "Step 7 — Running ContextPilot benchmark ($NUM_QUERIES queries)"

python scripts/mac_multihop_bench.py \
    --reordered_path "$REORDERED_PATH" \
    --corpus_path    "$CORPUS_PATH" \
    --num_queries    "$NUM_QUERIES" \
    --max_docs       "$MAX_DOCS" \
    --max_chars_per_doc "$MAX_CHARS_PER_DOC" \
    --proxy_log      "$PROXY_LOG" \
    --output         "$OUTPUT_CP"

log "ContextPilot results saved to $OUTPUT_CP"

# ── Step 8: Restart llama-server for fair baseline comparison ─────────────────
hr
log "Step 8 — Restarting llama-server to clear KV cache for fair baseline"

kill "$LLAMA_PID" 2>/dev/null || true
# Remove from PIDS list so cleanup doesn't warn about it
PIDS=("${PIDS[@]/$LLAMA_PID}")

# Wait for port to free
sleep 3

llama-server -m "$MODEL" \
    --host 0.0.0.0 --port "$LLAMA_PORT" \
    -ngl 99 --cache-reuse 256 --parallel 4 -c 32768 \
    >> /tmp/llama_server.log 2>&1 &
LLAMA_PID=$!
PIDS+=("$LLAMA_PID")
log "llama-server restarted PID=$LLAMA_PID"

wait_for_http "http://localhost:${LLAMA_PORT}/health" "llama-server (restarted)" 120

# Reset proxy slot state
log "Resetting proxy slot state..."
curl -sf -X POST "http://localhost:${PROXY_PORT}/reset" | python -m json.tool || true

# ── Step 9: Run baseline benchmark ────────────────────────────────────────────
hr
log "Step 9 — Running baseline benchmark (original doc order, $NUM_QUERIES queries)"

python scripts/mac_multihop_bench.py \
    --reordered_path "$REORDERED_PATH" \
    --corpus_path    "$CORPUS_PATH" \
    --num_queries    "$NUM_QUERIES" \
    --max_docs       "$MAX_DOCS" \
    --max_chars_per_doc "$MAX_CHARS_PER_DOC" \
    --proxy_log      "$PROXY_LOG" \
    --output         "$OUTPUT_BASELINE" \
    --baseline

log "Baseline results saved to $OUTPUT_BASELINE"

# ── Step 10: Summary ───────────────────────────────────────────────────────────
hr
log "Step 10 — Summary"
echo ""
echo "  Results files:"
echo "    ContextPilot : $OUTPUT_CP"
echo "    Baseline     : $OUTPUT_BASELINE"
echo "    Proxy log    : $PROXY_LOG"
echo ""
echo "  Service logs:"
echo "    llama-server      : /tmp/llama_server.log"
echo "    Eviction proxy    : /tmp/eviction_proxy.log"
echo "    ContextPilot HTTP : /tmp/contextpilot_server.log"
echo ""
echo "  Reference numbers (SGLang + Qwen3-32B on 4×A6000):"
echo "    Without ContextPilot:  cache_hit=4.64%   prefill_tps=7,290   F1=60.42"
echo "    With    ContextPilot:  cache_hit=33.97%  prefill_tps=14,214  F1=64.39"
echo ""
echo "  Note: Apple Silicon will show lower absolute TPS than a GPU server,"
echo "        but the relative cache-hit improvement (~5x) should still be visible."
hr
