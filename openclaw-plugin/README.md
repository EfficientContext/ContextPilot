# @contextpilot/contextpilot

OpenClaw plugin for [ContextPilot](https://github.com/EfficientContext/ContextPilot) — faster long-context inference via in-process context optimization. **Zero external dependencies** — no Python, no proxy server, just install and go.

## What It Does

ContextPilot registers as an OpenClaw **Context Engine** and optimizes every LLM request by:

1. **Extracting** documents from tool results
2. **Reordering** documents for maximum prefix cache sharing across turns
3. **Deduplicating** repeated content blocks with compact reference hints
4. **Injecting** cache control markers (Anthropic `cache_control: { type: "ephemeral" }`)

All processing happens in-process — no external services needed.

## Installation

### From npm (when published)

```bash
openclaw plugins install @contextpilot/contextpilot
```

### From local path (development)

Add to `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "load": {
      "paths": [
        "/path/to/ContextPilot/openclaw-plugin"
      ]
    }
  }
}
```

## Configuration

In `~/.openclaw/openclaw.json`, enable the plugin and set it as the context engine:

```json
{
  "plugins": {
    "slots": {
      "contextEngine": "contextpilot"
    },
    "entries": {
      "contextpilot": {
        "enabled": true,
        "config": {
          "scope": "all"
        }
      }
    }
  },
  "tools": {
    "allow": ["contextpilot"]
  }
}
```

### Scope Options

| Scope | Tool Results | Description |
|:------|:------------:|:------------|
| `all` (default) | Optimized | Optimize all tool results |
| `tool_results` | Optimized | Same as `all` |

> **Note:** System prompt optimization is not currently available — OpenClaw's context engine API does not expose the system prompt to plugins.

## How It Works

```
OpenClaw agent request
  ↓
ContextPilot Context Engine (assemble hook)
  ├─ Convert OpenClaw message format (toolResult → tool_result)
  ├─ Extract documents from tool results
  ├─ Reorder for prefix cache sharing
  ├─ Deduplicate repeated blocks
  ├─ Inject cache_control markers
  ↓
Optimized context → LLM Backend
```

The plugin registers as an OpenClaw Context Engine using `api.registerContextEngine()`. The `assemble()` hook intercepts context assembly before each LLM call.

## Files

```
openclaw-plugin/
├── openclaw.plugin.json   # Plugin manifest (id: "contextpilot")
├── package.json           # npm package (@contextpilot/contextpilot)
├── src/
│   ├── index.ts           # Plugin entry point
│   └── engine/
│       ├── cache-control.ts   # Cache control injection
│       ├── dedup.ts           # Content deduplication
│       ├── extract.ts         # Document extraction
│       └── live-index.ts      # Reordering engine
└── tsconfig.json
```

## Agent Tool

| Tool | Description |
|------|-------------|
| `contextpilot_status` | Check engine status, request count, and chars saved |

> **Note:** The status tool is registered but may not be visible to agents due to OpenClaw plugin API limitations.

## Verifying It Works

Check the gateway logs for ContextPilot output:

```
[ContextPilot] assemble() called with 84 messages
[ContextPilot] Extractions found - system: 0 tool: 1 singleDoc: 3
[ContextPilot] Optimization complete. Chars saved: 2389
```

## License

Apache-2.0
