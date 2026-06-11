# ContextPilot + OpenCode Integration Guide

## Overview

ContextPilot integrates with [OpenCode](https://github.com/opencode-ai/opencode) as a native plugin. It intercepts every LLM call via the `experimental.chat.messages.transform` hook, deduplicates repeated tool results across turns, and removes shared content blocks within tool outputs — all lossless, zero extra LLM calls.

Typical savings: **40-50% input tokens** on agentic workloads with repeated file reads.

## Prerequisites

- OpenCode installed and working
- Node.js 18+ (or Bun)

## Installation

Add `@contextpilot-ai/opencode` to your project's `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["@contextpilot-ai/opencode"]
}
```

Restart OpenCode. The plugin is installed automatically via Bun at startup and cached in `~/.cache/opencode/node_modules/`.

## How it works

ContextPilot registers on the `experimental.chat.messages.transform` hook. Before every LLM call, OpenCode passes the full message array to the plugin. ContextPilot runs a two-stage pipeline and mutates messages in-place:

| Stage | Operation | Benefit |
|-------|-----------|---------|
| 1 | Single-doc cross-turn dedup | Identical tool outputs (by SHA-256) replaced with a short pointer to the earlier result |
| 2 | Block-level dedup | Shared content blocks across different tool outputs are deduplicated via content-defined chunking |

Because the plugin mutates the message array directly, OpenCode sends the optimized (shorter) context to the LLM. The LLM sees less redundant content, and inference backends with KV cache or prompt caching benefit from the reduced input.

## Monitoring

### Log output

ContextPilot writes to its own log file alongside OpenCode's logs:

```bash
tail -f ~/.local/share/opencode/log/contextpilot.log
```

Example output:

```
[ContextPilot] Turn 8: saved 3212 chars (~803 tokens) | docs deduped: 2 | tracked: 5 | cumulative: 12840 chars (~3210 tokens)
```

### Status tool

During a session, call the `contextpilot_status` tool to see cumulative statistics:

```
ContextPilot Status:
  Turns optimized: 8
  Chars saved: 12,840
  Tokens saved: ~3,210
  Docs deduped: 2
  Tracked hashes: 5
  Reorder: dedup-only
```

## What gets optimized

**Single-doc cross-turn dedup.** When a tool returns output identical to a previous tool result (e.g. reading the same file twice), the duplicate is replaced with a short hint: `[Duplicate — identical to previous tool result (...). Refer to the earlier result above.]`. The original stays intact.

**Block-level dedup.** When different tool outputs share large content blocks (e.g. overlapping file sections from different reads), the shared blocks are deduplicated using content-defined chunking. This catches partial overlaps that single-doc dedup misses.

Tool outputs shorter than 100 characters are skipped — the overhead of hashing and tracking outweighs any savings.

## Troubleshooting

**Plugin not loading.** Check OpenCode's logs at `~/.local/share/opencode/log/` for plugin discovery errors:

```bash
grep -i "failed.*plugin" ~/.local/share/opencode/log/*.log
cat ~/.local/share/opencode/log/contextpilot.log
```

**0 chars saved.** Normal for the first few turns. Dedup only fires when the same content appears more than once in the conversation. Once an agent re-reads a file or tool results overlap, savings appear.

**Reorder shows "dedup-only".** The reorder stage requires the ContextPilot engine. If it fails to initialize, the plugin falls back gracefully to dedup-only mode. This is the expected default for most setups.
