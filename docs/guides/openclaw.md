# ContextPilot + OpenClaw Integration Guide

## Architecture

```
┌──────────────┐      ┌───────────────────┐      ┌─────────────┐
│  OpenClaw UI │─────▶│ ContextPilot Proxy│─────▶│ LLM Backend │
│  (browser)   │◀─────│   (localhost:8765) │◀─────│(Anthropic/  │
└──────────────┘      └───────────────────┘      │ OpenAI)     │
                                                  └─────────────┘
```

ContextPilot acts as a transparent HTTP proxy. OpenClaw sends requests to the proxy instead of directly to the LLM API. The proxy extracts documents, reorders them, and forwards to the real backend.

## Why This Matters for OpenClaw

OpenClaw's search and memory retrieval results appear as **tool_result messages** in the conversation history, not in the system prompt. When multiple search results are returned, their ordering affects the LLM's attention and response quality.

ContextPilot:
1. Extracts documents from tool_result content blocks
2. Clusters semantically related documents together
3. Reorders to minimize attention distance between related content
4. Preserves the original format (XML, numbered, etc.)

## Setup

### Quick Start (one command)

```bash
# Clone and run
git clone https://github.com/EfficientContext/ContextPilot.git
cd ContextPilot/examples/openclaw
bash setup.sh anthropic   # or: bash setup.sh openai
```

The script installs ContextPilot, generates a config, and starts the proxy.

### Docker

```bash
cd ContextPilot/examples/openclaw
docker compose up -d

# OpenAI instead of Anthropic:
CONTEXTPILOT_BACKEND_URL=https://api.openai.com docker compose up -d
```

### Manual

```bash
pip install contextpilot
python -m contextpilot.server.http_server \
  --stateless --port 8765 \
  --infer-api-url https://api.anthropic.com
```

## Configure OpenClaw

### Option A: UI (recommended)

1. Open OpenClaw
2. Go to **Settings → Models**
3. Add a custom provider:

| Field | Value |
|-------|-------|
| Name | `contextpilot-anthropic` |
| Base URL | `http://localhost:8765/v1` |
| API Key | your Anthropic API key |
| API | `anthropic-messages` |
| Model ID | `claude-opus-4-6` |

4. Select the model and start chatting

### Option B: Config file

Merge into `~/.openclaw/openclaw.json`:

```json
{
  "models": {
    "providers": {
      "contextpilot-anthropic": {
        "baseUrl": "http://localhost:8765/v1",
        "apiKey": "${ANTHROPIC_API_KEY}",
        "api": "anthropic-messages",
        "headers": {
          "X-ContextPilot-Scope": "all"
        },
        "models": [
          {
            "id": "claude-opus-4-6",
            "name": "Claude Opus 4.6 (via ContextPilot)",
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 200000,
            "maxTokens": 32000
          }
        ]
      }
    }
  }
}
```

For OpenAI, use `api: "openai-completions"` and point `--infer-api-url` to `https://api.openai.com`. See `examples/openclaw/openclaw.json.example` for both providers.

### Option C: Self-hosted model via SGLang

For self-hosted models, ContextPilot proxies between OpenClaw and SGLang:

```
OpenClaw ──▶ ContextPilot Proxy (server:8765) ──▶ SGLang (server:30000)
```

Start SGLang with tool calling support:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-27B \
  --tool-call-parser qwen3_coder \
  --port 30000
```

Start ContextPilot proxy:

```bash
python -m contextpilot.server.http_server \
  --port 8765 \
  --infer-api-url http://localhost:30000 \
  --model Qwen/Qwen3.5-27B
```

Merge into `~/.openclaw/openclaw.json`:

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "contextpilot-sglang": {
        "baseUrl": "http://<server-ip>:8765/v1",
        "apiKey": "placeholder",
        "api": "openai-completions",
        "headers": { "X-ContextPilot-Scope": "all" },
        "models": [
          {
            "id": "Qwen/Qwen3.5-27B",
            "name": "Qwen 3.5 27B (SGLang via ContextPilot)",
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 131072,
            "maxTokens": 8192
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": { "primary": "contextpilot-sglang/Qwen/Qwen3.5-27B" }
    }
  }
}
```

> **Important**: Use the server's IP address (not hostname) in `baseUrl` to avoid IPv6 DNS resolution issues in Node.js/WSL environments. `--tool-call-parser` is required for OpenClaw's tool loop to work.

## Verify

Check the `X-ContextPilot-Result` response header:

```
X-ContextPilot-Result: {"intercepted":true,"documents_reordered":true,"total_documents":8,"sources":{"system":1,"tool_results":2}}
```

If the header is absent, the request had fewer than 2 extractable documents (nothing to reorder).

## Document Extraction

ContextPilot auto-detects these formats in both system prompts and tool results:

| Format | Pattern | Typical Source |
|--------|---------|----------------|
| XML tags | `<documents><document>...</document></documents>` | RAG systems |
| File tags | `<files><file>...</file></files>` | Code search |
| Numbered | `[1] doc [2] doc` | Search rankings |
| Separator | docs split by `---` or `===` | Text chunking |
| Markdown headers | sections split by `#`/`##` | Structured docs |

Auto-detection priority: XML > Numbered > Separator > Markdown headers.

## Scope Control

| `X-ContextPilot-Scope` | System Prompt | Tool Results |
|:---:|:---:|:---:|
| `all` (default) | reordered | reordered |
| `system` | reordered | untouched |
| `tool_results` | untouched | reordered |

Set via headers in the OpenClaw provider config, or per-request.

## Full Header Reference

| Header | Description | Default |
|--------|-------------|---------|
| `X-ContextPilot-Enabled` | Enable/disable | `true` |
| `X-ContextPilot-Mode` | Extraction mode | `auto` |
| `X-ContextPilot-Scope` | Which messages to process | `all` |
| `X-ContextPilot-Tag` | Custom XML tag name | `document` |
| `X-ContextPilot-Separator` | Custom separator | `---` |
| `X-ContextPilot-Alpha` | Clustering distance parameter | `0.001` |
| `X-ContextPilot-Linkage` | Clustering linkage method | `average` |

## Troubleshooting

**No `X-ContextPilot-Result` header** — Request had < 2 extractable documents. Check that search/memory tools are returning multiple results.

**Connection refused** — Proxy not running. Check `curl http://localhost:8765/health`.

**`Connection error.` from OpenClaw (Node.js)** — IPv6 DNS resolution failure. Use IP address in `baseUrl`, or `export NODE_OPTIONS="--dns-result-order=ipv4first"`.

**401/403 from backend** — API key not set or invalid. The proxy forwards auth headers as-is.

**Tool call appears as XML text, agent stops** — SGLang not parsing tool calls into structured `tool_calls`. Add `--tool-call-parser qwen3_coder` (or the appropriate parser for your model) to SGLang launch command.

**Tool results not reordered** — Check scope is `all` or `tool_results`. Verify tool results use a supported format.
