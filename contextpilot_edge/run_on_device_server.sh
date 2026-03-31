# Terminal 1 — Launch llama-server with the native ContextPilot hook injected.
#
# contextpilot-llama-server is a drop-in for llama-server: same flags, same
# behavior. When CONTEXTPILOT_INDEX_URL is set it compiles a small C++ hook
# (once, cached in /tmp) and exec's llama-server with DYLD_INSERT_LIBRARIES
# injected — identical activation pattern to SGLang / vLLM.
#
# Requires: brew install llama.cpp  (or set LLAMA_SERVER_BIN=/path/to/llama-server)
CONTEXTPILOT_INDEX_URL=http://localhost:8765 contextpilot-llama-server \
  -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8889 -c 16384 --cache-reuse 256 --parallel 4 -ngl 99

# Terminal 2 — ContextPilot HTTP server (points directly at llama-server)
python -m contextpilot.server.http_server --port 8765 \
  --infer-api-url http://localhost:8889

# ── Optional: metrics proxy ───────────────────────────────────────────────────
# For Apple GPU metrics + per-request cache-reuse stats, start proxy_server.py.
# With CONTEXTPILOT_INDEX_URL set it injects id_slot into requests and calls
# POST /register_slot so the native hook can resolve evictions to request_ids.
#
#   Terminal 3 (proxy with ContextPilot integration):
#     CONTEXTPILOT_INDEX_URL=http://localhost:8765 \
#     python contextpilot_edge/proxy_server.py
#
#   Then point clients at http://localhost:8890 instead of :8889.
