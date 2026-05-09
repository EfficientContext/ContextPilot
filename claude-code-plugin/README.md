# ContextPilot for Claude Code

Prevents duplicate file reads to reduce context bloat. Tracks what Claude has already read and blocks redundant re-reads when file content hasn't changed.

## Installation

From the Anthropic marketplace:

```
/plugin install contextpilot@claude-plugins-official
```

## How it works

ContextPilot uses Claude Code's hook system to intercept Read tool calls:

1. **PostToolUse** — after every Read, records the file path, content hash, and modification time
2. **PreToolUse** — before every Read, checks if the file was already read and hasn't changed on disk. If unchanged, blocks the read with a message pointing Claude to the earlier result

The duplicate content never enters the context window. Claude sees a short denial message instead of the full file contents again.

If the file has been modified since the last read, the re-read is allowed and the tracking is updated.

## Monitoring

Check session statistics:

```
/contextpilot:status
```

View the plugin log:

```bash
cat ~/.claude/plugins/data/contextpilot/contextpilot.log
```

## License

Apache-2.0
