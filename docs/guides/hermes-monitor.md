# ContextPilot Hermes Monitor

This is an opt-in, metadata-only monitor for testing ContextPilot inside Hermes Agent over a one-week window.

## What it reads

- `~/.hermes/state.db:sessions` metadata only: token counts, tool/API call counts, source, estimated cost, timestamps.
- `~/.hermes/logs/gateway.log` lines containing ContextPilot savings summaries.

It intentionally does **not** read:

- `messages.content`
- `sessions.system_prompt`
- reasoning fields
- raw tool call payloads
- raw user/assistant text

Session ids are salted SHA-256 hashes in reports.

## Daily run

```bash
python scripts/hermes_contextpilot_monitor.py \
  --out-dir ~/contextpilot/reports \
  --since-hours 24
```

Outputs:

- `~/contextpilot/reports/daily_YYYY-MM-DD.json`
- `~/contextpilot/reports/daily_YYYY-MM-DD.md`

## Suggested Hermes cron job

Use this as a read-only watchdog. It produces reports; it does not apply config/code changes.

```python
cronjob(
    action="create",
    name="contextpilot-hermes-monitor-7d",
    schedule="0 4 * * *",
    repeat=7,
    deliver="origin",
    enabled_toolsets=["terminal", "file"],
    prompt="""
Run /root/work/ContextPilot/scripts/hermes_contextpilot_monitor.py with --out-dir /root/contextpilot/reports --since-hours 24.
Then read the generated Markdown report for today and send a short Chinese summary: token savings, session count, whether ContextPilot log events were observed, and any blocker. Do not read raw conversation content. Do not modify source/config.
""",
)
```

## Accuracy gate

This monitor only measures token/cost savings and operational signals. Before shipping ContextPilot changes, run a fixed golden eval set and require:

- no task-success regression,
- no drop in context recall beyond the chosen threshold,
- no unsafe raw-content leakage in reports,
- no increase in failed tool calls.

If any gate fails, hold proposals and require human review.
