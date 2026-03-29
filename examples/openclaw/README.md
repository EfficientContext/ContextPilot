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

## Using with SGLang (Self-Hosted Models)

For self-hosted models served by SGLang, ContextPilot acts as a proxy between OpenClaw and SGLang, enabling automatic document reordering and KV cache tracking.

```
OpenClaw (WSL/local) ──▶ ContextPilot Proxy (server:8765) ──▶ SGLang (server:30000)
```

### Step 1: Start SGLang with tool calling support

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-27B \
  --tool-call-parser qwen3_coder \
  --port 30000
```

> `--tool-call-parser` is required for OpenClaw's tool loop to work. Without it, tool calls are output as plain text and the agent loop won't continue.

### Step 2: Start ContextPilot proxy

```bash
python -m contextpilot.server.http_server \
  --port 8765 \
  --infer-api-url http://localhost:30000 \
  --model Qwen/Qwen3.5-27B
```

### Step 3: Configure OpenClaw

Patch your OpenClaw config (replace `<server-ip>` with your server's IP):

```bash
# Requires jq (install: sudo apt install jq / brew install jq)
jq '
  .agents.defaults.model.primary = "contextpilot-sglang/Qwen/Qwen3.5-27B" |
  .models = {
    "mode": "merge",
    "providers": {
      "contextpilot-sglang": {
        "baseUrl": "http://<server-ip>:8765/v1",
        "apiKey": "placeholder",
        "api": "openai-completions",
        "headers": {"X-ContextPilot-Scope": "all"},
        "models": [{
          "id": "Qwen/Qwen3.5-27B",
          "name": "Qwen 3.5 27B (SGLang via ContextPilot)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 8192
        }]
      }
    }
  }
' ~/.openclaw/openclaw.json > /tmp/oc.json && mv /tmp/oc.json ~/.openclaw/openclaw.json
```

Then restart OpenClaw:

```bash
pkill -f openclaw && openclaw gateway start && openclaw tui
```

<details>
<summary>Without jq: manually add this JSON to <code>~/.openclaw/openclaw.json</code></summary>

Add a `"models"` key at the top level and change `agents.defaults.model.primary`:

```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "contextpilot-sglang/Qwen/Qwen3.5-27B"
      }
    }
  },
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
  }
}
```

</details>

> **Important**: Use the server's IP address (not hostname) in `baseUrl` to avoid IPv6 DNS resolution issues in Node.js/WSL environments.

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection error.` | IPv6 DNS resolution fails in Node.js | Use IP address in `baseUrl`, or `export NODE_OPTIONS="--dns-result-order=ipv4first"` |
| Tool call appears as XML text, agent stops | SGLang not parsing tool calls | Add `--tool-call-parser qwen3_coder` to SGLang launch command |
| `Invalid JSON body` | Multiline curl command broken by shell | Use single-line JSON in curl |

## Scope Control

| Header value | System prompt | Tool results |
|:---:|:---:|:---:|
| `all` (default) | reordered | reordered |
| `system` | reordered | untouched |
| `tool_results` | untouched | reordered |

See [SKILL.md](SKILL.md) for the full header reference.
