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

Configure OpenClaw (replace `<server-ip>` with your server's IP):

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

Then restart:

```bash
pkill -f openclaw && openclaw gateway start && openclaw tui
```

<details>
<summary>Without jq: manually edit <code>~/.openclaw/openclaw.json</code></summary>

1. Change `agents.defaults.model.primary` to `"contextpilot-sglang/Qwen/Qwen3.5-27B"`
2. Add a `"models"` key at the top level:

```json
"models": {
  "mode": "merge",
  "providers": {
    "contextpilot-sglang": {
      "baseUrl": "http://<server-ip>:8765/v1",
      "apiKey": "placeholder",
      "api": "openai-completions",
      "headers": { "X-ContextPilot-Scope": "all" },
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
```

</details>

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

## OpenClaw Benchmark (ClawBench)

Benchmark ContextPilot against baseline SGLang on the OpenClaw multi-turn RAG dataset (143 tasks with high document overlap across turns).

### Prerequisites

```bash
git clone https://github.com/yourname/ClawBench.git
cd ClawBench
pip install -r requirements.txt
pip install ddgs
```

Build the ContextPilot+SGLang Docker image (from the contextpilot repo):

```bash
cd /path/to/ContextPilot
docker build -t contextpilot-sglang -f docker/Dockerfile.sglang .
```

### Option A: Automated (recommended)

Runs both ContextPilot and baseline back-to-back, then compares:

```bash
cd ClawBench
bash scripts/run_full_comparison.sh
```

Filter to a single topic or change the model:

```bash
bash scripts/run_full_comparison.sh --topic paper-transformer
bash scripts/run_full_comparison.sh --model Qwen/Qwen2.5-7B-Instruct
```

### Option B: Manual step-by-step

**1. ContextPilot + SGLang**

```bash
docker run --gpus all --name cp-bench -p 8765:8765 -p 30000:30000 contextpilot-sglang
python scripts/run_bench.py --tasks-file openclaw_tasks_all.json --runner api --model Qwen/Qwen2.5-7B-Instruct
docker rm -f cp-bench
```

**2. Baseline SGLang**

```bash
docker run --gpus all --name bl-bench -p 30000:30000 lmsysorg/sglang:latest \
  python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000
OPENAI_BASE_URL=http://localhost:30000/v1 python scripts/run_bench.py --tasks-file openclaw_tasks_all.json --runner api --model Qwen/Qwen2.5-7B-Instruct
docker rm -f bl-bench
```

**3. Compare**

```bash
python scripts/compare_runs.py results/run_CP.json results/run_BL.json --label-a "ContextPilot" --label-b "Baseline"
```

## Multi-User Document Search Benchmark (ClawBench)

Benchmark ContextPilot's KV cache reuse optimization when multiple users concurrently query `memory_search` against a shared document corpus. Four users ask biology/zoology questions that cluster around related topics, causing overlapping documents to be fetched — exactly the scenario where ContextPilot's document reordering yields cache hits.

**Architecture:**
```
Users A-D ── OpenClaw (cp-bench profile) ──▶ CP Proxy (:8765) ──▶ SGLang (:30000) [CP hooks]
Users A-D ── OpenClaw (bl-bench profile) ──▶ SGLang (:30001) [baseline, no reordering]
```

- **Model:** `Qwen3-4B-Instruct-2507`
- **Corpus:** 18 biology/zoology documents with 5 hub documents shared across users
- **Tasks:** 4 users × 5 questions = 20 tasks, run concurrently via `asyncio`

### Prerequisites

```bash
git clone https://github.com/yourname/ClawBench.git
cd ClawBench

# Generate the 20 task files
python scripts/generate_tasks_multiuser.py
```

Build the ContextPilot+SGLang Docker image:

```bash
cd /path/to/ContextPilot
docker build -t contextpilot-sglang -f docker/Dockerfile.sglang .
```

### Infrastructure Setup

**Terminal 1 — ContextPilot + SGLang** (serves both proxy on :8765 and model on :30000):

```bash
docker run --gpus all --name cp-bench \
  -p 8765:8765 -p 30000:30000 \
  contextpilot-sglang \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --tool-call-parser qwen3_coder \
  --port 30000
```

**Terminal 2 — Baseline SGLang** (no ContextPilot, port 30001):

```bash
docker run --gpus all --name bl-bench -p 30001:30000 \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 --port 30000
```

### Running the Benchmark

```bash
# Preview all 20 tasks across 4 users
python scripts/run_bench_multiuser.py --dry-run

# Run both ContextPilot and baseline
python scripts/run_bench_multiuser.py --mode both

# Run only ContextPilot or baseline
python scripts/run_bench_multiuser.py --mode cp
python scripts/run_bench_multiuser.py --mode baseline

# Filter to a single user
python scripts/run_bench_multiuser.py --mode both --user user-a
```

### Reading Results

Results are saved to `results/run_multiuser_<timestamp>.json`:

```json
{
  "run_id": "20260320_143022",
  "benchmark": "multiuser-docsearch",
  "model": "Qwen3-4B-Instruct-2507",
  "modes": {
    "cp": {
      "results_by_user": { "user-a": [...], "user-b": [...], ... },
      "metrics": { "avg_ttft_ms": 142, "p50_ttft_ms": 128, "p90_ttft_ms": 210, ... }
    },
    "baseline": {
      "results_by_user": { ... },
      "metrics": { "avg_elapsed_seconds": 45.2, ... }
    }
  },
  "comparison": {
    "elapsed_speedup": 1.35,
    "total_elapsed_speedup": 1.28,
    "cp_avg_ttft_ms": 142
  }
}
```

The `comparison` section shows how ContextPilot's document reordering translates into faster responses through KV cache reuse. Hub documents like `reptile-thermoregulation.md` and `reptile-conservation.md` appear in queries from all 4 users, so ContextPilot can reorder them to maximize prefix overlap across requests.

## Troubleshooting

**No `X-ContextPilot-Result` header** — Request had < 2 extractable documents. Check that search/memory tools are returning multiple results.

**Connection refused** — Proxy not running. Check `curl http://localhost:8765/health`.

**`Connection error.` from OpenClaw (Node.js)** — IPv6 DNS resolution failure. Use IP address in `baseUrl`, or `export NODE_OPTIONS="--dns-result-order=ipv4first"`.

**401/403 from backend** — API key not set or invalid. The proxy forwards auth headers as-is.

**Tool call appears as XML text, agent stops** — SGLang not parsing tool calls into structured `tool_calls`. Add `--tool-call-parser qwen3_coder` (or the appropriate parser for your model) to SGLang launch command.

**Tool results not reordered** — Check scope is `all` or `tool_results`. Verify tool results use a supported format.
