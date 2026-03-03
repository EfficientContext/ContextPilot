#!/usr/bin/env bash
set -euo pipefail

CONTEXTPILOT_PORT="${CONTEXTPILOT_PORT:-8765}"
SGLANG_PORT="${SGLANG_PORT:-30000}"

echo "Starting ContextPilot server on port ${CONTEXTPILOT_PORT}..."
python -m contextpilot.server.http_server \
    --port "${CONTEXTPILOT_PORT}" \
    --infer-api-url "http://localhost:${SGLANG_PORT}" &
CP_PID=$!

sleep 2

if ! kill -0 "${CP_PID}" 2>/dev/null; then
    echo "ERROR: ContextPilot server failed to start" >&2
    exit 1
fi
echo "ContextPilot server running (PID ${CP_PID})"

exec python3 -m sglang.launch_server \
    --host 0.0.0.0 \
    --port "${SGLANG_PORT}" \
    "$@"
