---
description: Show ContextPilot statistics for this session — blocked reads, chars saved, and tracked files
disable-model-invocation: true
---

Read the ContextPilot session state and display the statistics.

Run this bash command and show the output to the user in a readable format:

```bash
cat "${CLAUDE_PLUGIN_DATA}/session-state.json" 2>/dev/null | jq '{
  reads_blocked: .stats.blocked,
  reads_allowed: .stats.allowed,
  chars_saved: .stats.charsSaved,
  estimated_tokens_saved: (.stats.charsSaved / 4 | floor),
  files_tracked: (.reads | length)
}' 2>/dev/null || echo '{"error": "No session data yet. ContextPilot starts tracking after the first file read."}'
```
