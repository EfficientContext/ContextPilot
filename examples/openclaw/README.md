# ContextPilot + OpenClaw Quick Start

## Flow

```
1. Start ContextPilot proxy          (one command)
2. OpenClaw UI → Settings → Models   (add custom provider pointing to proxy)
3. Select model, start chatting       (done)
```

```
OpenClaw UI ──▶ ContextPilot Proxy (localhost:8765) ──▶ LLM API (Anthropic/OpenAI)
      ◀──────────────── responses flow back ◀──────────────────
```

## One-Click Setup

### Option A: Shell script

```bash
# Anthropic (default)
bash setup.sh anthropic

# OpenAI
bash setup.sh openai
```

The script will:
1. Check Python, install ContextPilot
2. Generate the provider config JSON
3. Print what to enter in OpenClaw UI
4. Start the proxy

### Option B: Docker Compose

```bash
# Anthropic (default)
docker compose up -d

# OpenAI
CONTEXTPILOT_BACKEND_URL=https://api.openai.com docker compose up -d
```

## Manual Setup

### Step 1: Start proxy

```bash
pip install contextpilot

python -m contextpilot.server.http_server \
  --stateless --port 8765 \
  --infer-api-url https://api.anthropic.com
```

### Step 2: Add provider in OpenClaw

Open OpenClaw UI → **Settings** → **Models** → add custom provider:

| Field | Value |
|-------|-------|
| Base URL | `http://localhost:8765/v1` |
| API Key | your Anthropic/OpenAI key |
| API | `anthropic-messages` or `openai-completions` |
| Headers | `X-ContextPilot-Scope: all` |

Or merge `openclaw.json.example` into `~/.openclaw/openclaw.json`:

```json
{
  "models": {
    "providers": {
      "contextpilot-anthropic": {
        "baseUrl": "http://localhost:8765/v1",
        "apiKey": "${ANTHROPIC_API_KEY}",
        "api": "anthropic-messages",
        "headers": { "X-ContextPilot-Scope": "all" },
        "models": [{ "id": "claude-opus-4-6", "name": "Claude Opus 4.6 (via ContextPilot)" }]
      }
    }
  }
}
```

### Step 3: Select model and chat

In OpenClaw, select the model from the ContextPilot provider. Search/memory results in tool_results will be automatically reordered.

### Step 4: Verify

Check the `X-ContextPilot-Result` response header for metadata:

```
X-ContextPilot-Result: {"intercepted":true,"documents_reordered":true,"total_documents":5,"sources":{"system":1,"tool_results":1}}
```

## Scope Control

| Header value | System prompt | Tool results |
|:---:|:---:|:---:|
| `all` (default) | reordered | reordered |
| `system` | reordered | untouched |
| `tool_results` | untouched | reordered |

See [SKILL.md](SKILL.md) for the full header reference.
