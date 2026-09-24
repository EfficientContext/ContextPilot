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

After installing, enable the plugin in the Hermes plugins menu:

```bash
hermes plugins
```

Navigate to **General Plugins** → toggle **contextpilot** to enabled.

Restart Hermes. On startup you'll see:

```
Plugin 'contextpilot' registered context engine: contextpilot
```

> **Note:** The context engine TUI submenu may show "contextpilot (not found)" — this is cosmetic. The engine is fully functional.

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

After a session with repeated tool calls (e.g. reading the same file twice), check the Hermes log:

```
[ContextPilot] Turn 14: saved 4408 chars (~1102 tokens) | cumulative: 19574 chars (~4893 tokens)
```

Or query the engine status programmatically:

```python
from hermes_cli.plugins import get_plugin_manager
engine = get_plugin_manager()._context_engine
print(engine.get_status())
# {'engine': 'contextpilot', 'contextpilot_chars_saved': 18420, ...}
```

## See processed-payload savings

Once ContextPilot has run for a bit, you can see realized savings from the
metadata-only telemetry with a single command from the ContextPilot repo or
plugin directory. Character savings are always measured from the actual
before/after LLM-bound payload after ContextPilot processing. Exact tokenizer
savings are shown only when telemetry recorded an exact tokenizer backend; the
legacy chars/4 counter is labelled as derived. To record tokenizer-based deltas,
configure an exact matching tokenizer explicitly, for example
`CONTEXTPILOT_EXACT_TOKENIZER=tiktoken` with
`CONTEXTPILOT_TIKTOKEN_ENCODING=<encoding>`; it is off by default to avoid
provider/tokenizer mismatches.

```bash
python scripts/contextpilot_savings.py
# or, after Hermes plugin install:
python ~/.hermes/plugins/ContextPilot/scripts/contextpilot_savings.py
```

```
ContextPilot savings (last 24h)
  Events:                  117
  Chars saved:             6,147,074
  Est. tokens saved (chars/4, derived): 1,536,728
  Actual tokens saved (tokenizer): unavailable (no exact tokenizer backend recorded)
  Telemetry file:          /root/.hermes/contextpilot/telemetry.jsonl
```

Useful options:

- `--all-time` — total savings since you enabled ContextPilot.
- `--since-hours N` — only the last `N` hours (default `24`).
- `--format json` — stable, machine-readable output for dashboards.

This reads only the metadata-only telemetry file
(`~/.hermes/contextpilot/telemetry.jsonl`); it never touches your conversations,
prompts, or tool output. If it reports no savings, make sure ContextPilot is
enabled, restart Hermes, and run a workload that reads the same content more than
once — savings only happen when content repeats across turns.

### Ask Hermes instead

If you'd rather not run the command yourself, install or copy the bundled
read-only Hermes skill at `skills/contextpilot-savings/SKILL.md` and just ask
Hermes ("how much did ContextPilot save?"). Hermes finds the script, runs it
with the right window/format, and summarizes the result. The skill is
observe-only: it reads nothing but the metadata-only telemetry via this script.
See [`hermes-monitor.md`](./hermes-monitor.md#ask-hermes-for-savings) for details.

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

**No savings logged.** Dedup only fires when the LLM reads the same file content more than once in a session. On first reads, content is indexed but not deduplicated.

**`ModuleNotFoundError: No module named 'numpy'`.** Reorder requires numpy. If unavailable, ContextPilot silently falls back to dedup-only mode.
