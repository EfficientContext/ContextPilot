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

## Verify

Check the `_contextpilot` field in API responses:

```json
{
  "_contextpilot": {
    "intercepted": true,
    "documents_reordered": true,
    "total_documents": 8,
    "sources": {
      "system": 1,
      "tool_results": 2
    }
  }
}
```

If `_contextpilot` is absent, the request had fewer than 2 extractable documents (nothing to reorder).

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

**No `_contextpilot` in response** — Request had < 2 extractable documents. Check that search/memory tools are returning multiple results.

**Connection refused** — Proxy not running. Check `curl http://localhost:8765/health`.

**401/403 from backend** — API key not set or invalid. The proxy forwards auth headers as-is.

**Tool results not reordered** — Check scope is `all` or `tool_results`. Verify tool results use a supported format.
