# ContextPilot + Claude Code Integration Guide

## Overview

ContextPilot integrates with [Claude Code](https://code.claude.com) as a native plugin. It prevents duplicate file reads from bloating the context window by tracking what Claude has already read and blocking redundant re-reads when file content hasn't changed.

Unlike the OpenCode and Hermes plugins which perform post-hoc deduplication, the Claude Code plugin uses **prevention** — duplicate content never enters the context at all.

## Prerequisites

- Claude Code installed and authenticated
- `jq` available in PATH (standard on most systems)

## Installation

Install from the Anthropic marketplace:

```
/plugin install contextpilot@claude-plugins-official
```

## How it works

ContextPilot registers two hooks on Claude Code's Read tool:

| Hook | Event | Action |
|------|-------|--------|
| `PostToolUse` | After a Read completes | Records file path, content hash, and mtime in session state |
| `PreToolUse` | Before a Read executes | Checks if file was already read and unchanged — blocks if duplicate |

When a duplicate read is blocked, Claude sees: "File already in context. Content unchanged (N chars). Refer to the earlier read." This is a few dozen characters instead of the full file contents.

If the file has been modified since the last read (detected via mtime), the re-read is allowed and tracking is updated.

## Monitoring

### Status skill

Run `/contextpilot:status` to see cumulative session statistics:

```
{
  "reads_blocked": 4,
  "reads_allowed": 12,
  "chars_saved": 28400,
  "estimated_tokens_saved": 7100,
  "files_tracked": 12
}
```

### Log file

View detailed per-event logs:

```bash
cat ~/.claude/plugins/data/contextpilot/contextpilot.log
```

## Troubleshooting

**Plugin not loading.** Run `/plugin` and check the Errors tab. Verify the plugin is enabled in the Installed tab.

**Reads not being blocked.** Normal for the first read of any file. Dedup only kicks in on the second read of the same unchanged file. Check the log to confirm tracking is working.

**False blocks after file edits.** If a file was edited but the mtime didn't change (rare), the plugin may incorrectly block the re-read. This resolves on the next edit.
