# ContextPilot Hermes Monitor

This is an opt-in, metadata-only monitor for testing ContextPilot inside Hermes Agent over a one-week window.

## What it reads

- `~/.hermes/state.db:sessions` metadata only: token counts, tool/API call counts, source, estimated cost, timestamps.
- `~/.hermes/contextpilot/telemetry.jsonl` metadata-only ContextPilot savings records (preferred source).
- `~/.hermes/logs/gateway.log` lines containing ContextPilot savings summaries (fallback source).

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
  --since-hours 24 \
  --telemetry-file ~/.hermes/contextpilot/telemetry.jsonl
```

The telemetry file is written by the ContextPilot Hermes plugin when savings occur. Set `CONTEXTPILOT_DISABLE_TELEMETRY=1` to disable writes, or `CONTEXTPILOT_TELEMETRY_FILE=/path/to/file.jsonl` to override the location.

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

## Opportunity scanning

`scripts/analyze_hermes_context_opportunities.py` is a companion scanner meant
for a continuous cron job. Where the monitor stays metadata-only, this analyzer
*does* read message content and tool outputs — but only in-memory, to compute
salted SHA-256 fingerprints and aggregate counters. Reports never contain raw
message/tool text, system prompts, reasoning, or raw session ids.

It surfaces concrete token-reduction opportunities:

- exact duplicate tool outputs (identical payloads re-sent across turns),
- repeated line/block fingerprints (shared boilerplate across outputs),
- large tool outputs grouped by `tool_name`,
- heavy sessions by input-token / tool-call / message counts (hashed ids),
- ContextPilot telemetry coverage and savings ratios.

```bash
python scripts/analyze_hermes_context_opportunities.py \
  --state-db /root/.hermes/state.db \
  --telemetry-file ~/.hermes/contextpilot/telemetry.jsonl \
  --out-dir ~/contextpilot/opportunities \
  --since-hours 24
```

Outputs:

- `~/contextpilot/opportunities/opportunities_YYYY-MM-DD.json`
- `~/contextpilot/opportunities/opportunities_YYYY-MM-DD.md`

Each estimated "wasted tokens" figure is a heuristic (chars / 4); treat the
report as a prioritized list of candidates and validate against the accuracy
gate below before changing ContextPilot config or code. A defensive guard in
`write_report` refuses to emit any forbidden raw-content key, so the reports are
safe to ship from an unattended cron job.

## Accuracy gate

This monitor only measures token/cost savings and operational signals. Before shipping ContextPilot changes, run a fixed golden eval set and require:

- no task-success regression,
- no drop in context recall beyond the chosen threshold,
- no unsafe raw-content leakage in reports,
- no increase in failed tool calls.

If any gate fails, hold proposals and require human review.
