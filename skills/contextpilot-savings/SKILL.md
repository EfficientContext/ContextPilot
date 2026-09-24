---
name: contextpilot-savings
description: Use when a user asks how many tokens (or how much context/cost) ContextPilot has saved, or wants a ContextPilot savings status/summary inside Hermes Agent — e.g. "how much did ContextPilot save?", "show ContextPilot token savings", "ContextPilot status", "savings in the last 24h / all time", or "give me the raw savings JSON". This skill runs the metadata-only `scripts/contextpilot_savings.py` reporter and summarizes its output in plain language. It is read-only and observe-only — it never changes ContextPilot code or config, never enables routing/dropping/summarization, never creates cron jobs, branches, or pull requests, and never reads conversation messages, tool outputs, system prompts, reasoning, or raw session ids.
version: 1.0.0
author: ContextPilot
license: MIT
metadata:
  hermes:
    tags: [contextpilot, hermes, telemetry, token-savings, observability, read-only]
    related_skills: []
    category: observability
    safety: read-only
---

# ContextPilot Savings (Hermes)

This is a small, **read-only** skill: when a user wants to know how many tokens
ContextPilot has saved, run the lightweight savings reporter and explain the
result in plain language. That is the entire job — measure and report, nothing
else.

> Scope: **show savings/status only.** This skill never modifies ContextPilot
> code or config, never enables context routing/dropping/summarization, never
> schedules background jobs, and never edits the repository or opens code-change
> requests. If a user wants any of that, this is the wrong tool — stop and tell
> them so.

## When to use this skill

- "How many tokens did ContextPilot save?"
- "Show me ContextPilot token savings" / "ContextPilot status".
- "Savings in the last 24 hours" / "savings all time".
- "Give me the raw savings data / JSON."

## Privacy boundary (read this first)

The only thing this skill is allowed to read is the **metadata-only** telemetry
file (`~/.hermes/contextpilot/telemetry.jsonl`), and it reads it **only**
through `scripts/contextpilot_savings.py`. That file contains nothing but
numeric counters (timestamps, chars saved, tokens saved).

Do **not**, under any circumstances:

- read `~/.hermes/state.db`, conversation `messages`, or message content,
- read system prompts, skill prompts, or reasoning fields,
- read raw tool-call payloads or tool outputs,
- read or print raw session ids.

If you are ever tempted to inspect any of the above to "explain" the savings,
don't — the script's metadata summary is the complete, safe answer.

## How to run it

### Step 1 — Locate the script

Use whichever of these exists (check in this order):

1. `scripts/contextpilot_savings.py` — when you are in a ContextPilot repo or
   plugin checkout (current working directory).
2. `~/.hermes/plugins/ContextPilot/scripts/contextpilot_savings.py` — when
   ContextPilot was installed as a Hermes plugin.

Pick the first one that exists and use that path as `<script>` below.

### Step 2 — Run the reporter

Default (last 24 hours):

```bash
python3 <script> --since-hours 24
```

Vary the command to match the request:

- **All time** — if the user asks for total/lifetime savings:

  ```bash
  python3 <script> --all-time
  ```

- **A different window** — e.g. last hour:

  ```bash
  python3 <script> --since-hours 1
  ```

- **JSON / raw data** — if the user asks for machine-readable output, raw data,
  or to feed a dashboard:

  ```bash
  python3 <script> --format json
  ```

  (Combine with `--all-time` or `--since-hours N` as needed.)

The script prints events, chars saved, telemetry tokens saved, the time window,
and average tokens per event.

### Step 3 — Summarize in plain language

Read the script's output and give the user a short, friendly summary: roughly
how many tokens ContextPilot saved, over what window, and the event count. If
they asked for JSON, show the JSON and add a one-line readout. Do not invent
numbers — only report what the script printed.

## When there are no savings yet

If the script reports no telemetry file or zero events in the window, tell the
user (in plain language) to:

1. Enable ContextPilot in Hermes (`hermes plugins` → toggle **contextpilot**).
2. Restart Hermes.
3. Repeat a workload that reads the same content more than once (e.g. read a
   file, then read it again) — savings only fire when content repeats across
   turns.

You can also suggest `--all-time` to widen the window in case recent activity
just fell outside the last 24 hours.

## Installation note

For normal use, installing or copying this skill does **not** require a
`--force` flag — only reach for `--force` if a skill installer reports an
existing-name collision you intend to overwrite. The skill itself runs a single
read-only Python command; it needs no special privileges.

## Out of scope (do not do these)

This skill is deliberately narrow. It must **not**:

- modify ContextPilot source code or configuration,
- enable or tune context routing, dropping, or summarization,
- schedule background or recurring runs,
- edit the repository, run a test/merge workflow, or open code-change requests,
- train or evaluate any router, or run a propose/apply improvement loop.

If the user wants deeper monitoring (state.db metadata, opportunity scanning),
point them at `docs/guides/hermes-monitor.md`. For integration mechanics, point
them at `docs/guides/hermes.md`. This skill only answers "how much did
ContextPilot save?".
