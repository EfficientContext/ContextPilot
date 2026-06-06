---
description: Show ContextPilot statistics for this session — blocked reads, chars saved, and tracked files
---

Run this exact bash command and display the JSON output:

cat ~/.claude/plugins/data/contextpilot-inline/session-state.json 2>/dev/null | jq '{reads_blocked: .stats.blocked, reads_allowed: .stats.allowed, chars_saved: .stats.charsSaved, estimated_tokens_saved: (.stats.charsSaved / 4 | floor), files_tracked: (.reads | length)}' 2>/dev/null || echo "No session data yet."
