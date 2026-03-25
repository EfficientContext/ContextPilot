# @contextpilot/openclaw-plugin

OpenClaw native plugin for [ContextPilot](https://github.com/EfficientContext/ContextPilot) — faster long-context inference via in-process context reuse. **Zero external dependencies** — no Python, no proxy server, just install and go.

## What It Does

ContextPilot optimizes every LLM request by:

1. **Extracting** documents from system prompts and tool results
2. **Reordering** documents for maximum prefix cache sharing across turns
3. **Deduplicating** repeated content blocks with compact reference hints
4. **Injecting** provider-specific cache control markers (Anthropic `cache_control`)

All processing happens in-process inside the OpenClaw plugin — no external services needed.

## Installation

```bash
openclaw plugins install @contextpilot/openclaw-plugin
```

## Configuration

In `~/.openclaw/openclaw.json`:

```json5
{
  plugins: {
    entries: {
      "contextpilot": {
        enabled: true,
        config: {
          // "anthropic" (default) or "openai"
          "backendProvider": "anthropic",
          
          // What to optimize: "all" (default), "system", or "tool_results"
          "scope": "all"
        }
      }
    }
  }
}
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-xxx"
# or
export OPENAI_API_KEY="sk-xxx"
```

Then select a ContextPilot model (e.g., `contextpilot/claude-sonnet-4-6`) and start using OpenClaw.

## Available Models

### Anthropic backend (default)

| Model ID | Name |
|----------|------|
| `contextpilot/claude-opus-4-6` | Claude Opus 4.6 (ContextPilot) |
| `contextpilot/claude-sonnet-4-6` | Claude Sonnet 4.6 (ContextPilot) |

### OpenAI backend

| Model ID | Name |
|----------|------|
| `contextpilot/gpt-4o` | GPT-4o (ContextPilot) |
| `contextpilot/gpt-4o-mini` | GPT-4o Mini (ContextPilot) |

Any model ID works via dynamic resolution — use `contextpilot/<any-model-id>`.

## How It Works

```
OpenClaw request
  ↓
ContextPilot Plugin (wrapStreamFn)
  ├─ Extract documents from system/tool_results
  ├─ Reorder for prefix cache sharing
  ├─ Deduplicate repeated blocks
  ├─ Inject cache_control markers
  ↓
Optimized request → LLM Backend (Anthropic/OpenAI)
```

The plugin registers as an OpenClaw provider and uses `wrapStreamFn` to intercept requests before they reach the backend. All optimization is done in-process in TypeScript.

## Agent Tool

| Tool | Description |
|------|-------------|
| `contextpilot_status` | Check engine status, request count, and chars saved |

## Scope Control

| Scope | System Prompt | Tool Results |
|:---:|:---:|:---:|
| `all` (default) | Optimized | Optimized |
| `system` | Optimized | Untouched |
| `tool_results` | Untouched | Optimized |

## License

Apache-2.0
