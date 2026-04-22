# ContextPilot + Hermes Agent Integration Guide

## Overview

ContextPilot integrates with [Hermes Agent](https://github.com/NousResearch/hermes-agent) as a native context engine plugin — zero changes to Hermes source code required. It intercepts every LLM call, reorders documents for prefix cache sharing, deduplicates repeated tool results, and deduplicates content blocks across turns.

Typical savings: **40–50% input tokens** on agentic workloads with repeated file reads.

## Installation

```bash
hermes plugins install EfficientContext/ContextPilot
```

This clones ContextPilot into `~/.hermes/plugins/ContextPilot/`. Hermes's plugin system discovers the `plugin.yaml` manifest and loads the context engine via the standard `register(ctx)` entry point.

## Activation

Open the plugin configuration TUI:

```bash
hermes plugins
```

Navigate to **Context Engine** → select **contextpilot** → press Enter.

Alternatively, edit `~/.hermes/config.yaml` directly:

```yaml
context:
  engine: contextpilot
```

Start a new session. Hermes will log the engine on startup:

```
Plugin 'contextpilot' registered context engine: contextpilot
```

## What it does

Every API call, before messages are sent to the LLM, ContextPilot runs a six-step pipeline:

| Step | Operation | Benefit |
|------|-----------|---------|
| 1 | Prefix replay | KV cache prefix stays identical across turns |
| 2 | Extract documents | Parse tool results into document arrays |
| 3 | Reorder | Cluster similar documents for prefix sharing |
| 4 | Cross-turn dedup | Replace repeated file reads with a pointer |
| 5 | Block-level dedup | Content-defined chunking within tool results |
| 6 | Cache | Store modified messages for next turn's prefix replay |

Steps 2–3 require `numpy` (already a Hermes dependency). If numpy is unavailable, ContextPilot falls back to dedup-only mode.

## Verifying it works

After a session with several tool calls, check the Hermes log:

```
[ContextPilot] Turn 3: 2814 chars saved, 12 blocks deduped, 4 docs reordered
```

Or query the engine status programmatically:

```python
from hermes_cli.plugins import get_plugin_manager
engine = get_plugin_manager()._context_engine
print(engine.get_status())
# {'engine': 'contextpilot', 'contextpilot_chars_saved': 18420, ...}
```

## Disabling

```bash
hermes plugins disable contextpilot
```

Or reset the context engine to the built-in compressor:

```yaml
context:
  engine: compressor
```

## Uninstalling

```bash
hermes plugins remove contextpilot
```

## How it differs from Hermes's built-in ContextCompressor

Hermes ships with `ContextCompressor`, a threshold-based LLM-summarization engine. ContextPilot wraps and extends it:

| | Built-in compressor | ContextPilot |
|---|---|---|
| Trigger | Token threshold (75% of context) | Every API call |
| Approach | Lossy LLM summarization | Lossless dedup + reorder |
| Cache-friendly | No | Yes — preserves prefix for KV cache |
| Cost | One summarization LLM call per compression | Zero extra LLM calls |
| Fallback | N/A | Delegates to built-in compressor when compression is actually needed |

ContextPilot runs *before* the threshold-based compressor, reducing how often the expensive summarization path is hit.

## Troubleshooting

**Plugin not discovered after install.** Check `~/.hermes/plugins/ContextPilot/plugin.yaml` exists and contains `type: context_engine`. Run `hermes plugins list` to confirm.

**Engine loads but `optimize_api_messages` is not called.** Your Hermes version may be older than the plugin hook interface. ContextPilot's engine still works — it wraps the built-in compressor transparently — but per-turn optimization requires Hermes's `optimize_api_messages` hook.

**`ModuleNotFoundError: No module named 'numpy'`.** Reorder requires numpy. Install it in the Hermes venv, or accept dedup-only mode (still provides 30–40% savings on most workloads).
