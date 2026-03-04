#!/usr/bin/env bash
set -euo pipefail

CONTEXTPILOT_PORT="${CONTEXTPILOT_PORT:-8765}"

exec python3 -m contextpilot.server.http_server \
    --port "${CONTEXTPILOT_PORT}" \
    "$@"
