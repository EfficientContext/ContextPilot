---
name: contextpilot-self-evolve
description: Use when a user wants to install/enable ContextPilot inside Hermes Agent and then run a safe, repeatable "self-evolve" loop — collect metadata-only telemetry and content-aware shadow data, analyze realized token savings vs advisory candidate tokens, and propose ContextPilot improvements under strict safety gates. Use it for monitoring token spend, scanning context-redundancy opportunities, setting up read-only daily/weekly cron analysis, and preparing reviewed, branch-gated code/config changes. Do NOT use it to auto-apply routing/drop/summarization changes; this skill only proposes risky changes and requires tests, privacy checks, and independent review before anything ships.
version: 1.0.0
author: ContextPilot
license: MIT
metadata:
  hermes:
    tags: [contextpilot, hermes, telemetry, context-optimization, token-savings, safety-gated]
    related_skills: []
    category: observability
    safety: proposal-only
---

# ContextPilot Self-Evolve (Hermes)

This skill drives a **safe, repeatable** loop for running ContextPilot inside
Hermes Agent and continuously improving it from real telemetry — **without**
auto-applying any risky context change. You measure, you analyze, you *propose*;
a human (plus tests, privacy checks, and independent review) decides what ships.

> Core safety stance: **observe and propose only.** This skill never enables
> context routing, dropping, or summarization on its own. Shadow/advisory
> numbers are training/eval data, **not** realized savings, and must never be
> treated as something to "just turn on."

## When to use this skill

- A user asks to install or enable ContextPilot in Hermes and watch its impact.
- A user wants to know how many tokens/cost ContextPilot is actually saving.
- A user wants to find token-reduction *opportunities* (duplicate tool outputs,
  cross-role repeated blocks, oversized tool results, routing/dedup candidates).
- A user wants a daily/weekly read-only cron that reports savings + opportunities.
- A user wants to propose a ContextPilot config or code change and needs the safe
  workflow (branch, tests, privacy/no-raw-content checks, independent review).

If the user instead wants the low-level integration mechanics, point them at
`docs/guides/hermes.md`; for the metadata-only monitor details, see
`docs/guides/hermes-monitor.md`. This skill orchestrates both into one loop.

## Privacy boundary (read this first)

There are two analysis tools with **different** read scopes:

- `scripts/hermes_contextpilot_monitor.py` — **metadata only**. Never reads
  `messages.content`, `sessions.system_prompt`, reasoning, or raw tool payloads.
- `scripts/analyze_hermes_context_opportunities.py` — **content-aware**. It
  *may* read message/tool/system content **in-memory** to compute salted
  SHA-256 fingerprints and aggregate counters.

In **both** cases the rule is absolute: **reports must never emit raw
conversation text, tool-call payloads, system prompts, reasoning, or raw session
ids.** Session ids appear only as salted hashes. The analyzer has a defensive
`write_report` guard that refuses to emit forbidden raw-content keys; do not
weaken or bypass it. If you are ever unsure whether an output is safe to ship,
treat it as unsafe and stop.

## Workflow

### Step 1 — Install / enable ContextPilot in Hermes

Normal install (do **not** use `--force`):

```bash
hermes plugins install EfficientContext/ContextPilot --enable
hermes config set context.engine contextpilot
```

`--force` is **only** for an intentional update/reinstall over an existing
install — never as the default:

```bash
hermes plugins install EfficientContext/ContextPilot --enable --force
```

If your Hermes version does not support `--enable`, install first and then use the
plugin menu:

```bash
hermes plugins            # General Plugins -> toggle "contextpilot" enabled
```

### Step 2 — Verify the context engine + restart

Confirm Hermes is actually routing through ContextPilot. The active context
engine must be `contextpilot`:

```yaml
# ~/.hermes config
context:
  engine: contextpilot
```

```python
from hermes_cli.plugins import get_plugin_manager
engine = get_plugin_manager()._context_engine
print(engine.get_status())   # expect {'engine': 'contextpilot', ...}
```

Then **restart the Hermes gateway / start a fresh session** so the engine is
loaded. On startup you should see:

```
Plugin 'contextpilot' registered context engine: contextpilot
```

> The context-engine TUI submenu may show "contextpilot (not found)" — that is
> cosmetic; `get_status()` is the source of truth.

### Step 3 — Run the metadata-only monitor

Use this as the safe baseline. It reports realized savings and operational
signals from telemetry/metadata only:

```bash
python scripts/hermes_contextpilot_monitor.py \
  --out-dir ~/contextpilot/reports \
  --since-hours 24 \
  --telemetry-file ~/.hermes/contextpilot/telemetry.jsonl
```

Reports:

- `~/contextpilot/reports/daily_YYYY-MM-DD.json`
- `~/contextpilot/reports/daily_YYYY-MM-DD.md`

The telemetry file is written by the ContextPilot Hermes plugin when savings
occur. `CONTEXTPILOT_DISABLE_TELEMETRY=1` disables writes;
`CONTEXTPILOT_TELEMETRY_FILE=/path` overrides the location.

### Step 4 — Run the content-aware opportunity analyzer

Run for both a rolling day and a rolling week to separate noise from trend:

```bash
# last 24h
python scripts/analyze_hermes_context_opportunities.py \
  --state-db ~/.hermes/state.db \
  --telemetry-file ~/.hermes/contextpilot/telemetry.jsonl \
  --out-dir ~/contextpilot/opportunities \
  --since-hours 24

# last 7 days (168h)
python scripts/analyze_hermes_context_opportunities.py \
  --state-db ~/.hermes/state.db \
  --telemetry-file ~/.hermes/contextpilot/telemetry.jsonl \
  --out-dir ~/contextpilot/opportunities \
  --since-hours 168
```

For a one-shot whole-history audit, swap the window for `--all-sessions`.
Reports:

- `~/contextpilot/opportunities/opportunities_YYYY-MM-DD.json`
- `~/contextpilot/opportunities/opportunities_YYYY-MM-DD.md`

The analyzer surfaces: exact duplicate tool outputs, repeated line/block
fingerprints, large outputs by `tool_name`, heavy sessions (hashed ids),
ContextPilot telemetry coverage/ratios, **LLM-bound cross-type repeated
blocks**, **Worker Context Routing shadow labels**, and **Parent Aggregation
Artifact** dedup telemetry. The shadow/parent sections are **on by default** and
collect P0 data only; pass `--disable-worker-routing-shadow` or
`--disable-parent-aggregation` to omit a section.

### Step 5 — Interpret: realized savings vs advisory candidates

Keep these two numbers in separate mental buckets — never add them together:

- **Realized savings** (telemetry: `chars_saved`, `~tokens`, savings ratio,
  monitor report) — what ContextPilot *actually* saved via lossless dedup +
  reorder. This is real and bankable.
- **Advisory / shadow candidate tokens** (analyzer: routing-shadow
  `est_advisory_candidate_tokens`, parent-aggregation `est_duplicate_tokens`,
  cross-type redundant tokens) — an **upper-bound estimate** of what a *future*
  router/dedup *might* save. **Not realized.** It is training/eval data, and
  every token estimate is a heuristic (`chars/4`).

When reporting to the user, state realized savings as fact and label every
advisory number as a candidate that still needs validation. Do not imply that
advisory tokens are available simply by toggling a flag.

### Step 6 — Optional read-only cron jobs

Schedule the monitor and/or analyzer as **read-only watchdogs**. They produce
reports; they must not apply config or code changes.

```python
cronjob(
    action="create",
    name="contextpilot-self-evolve-daily",
    schedule="0 4 * * *",
    repeat=7,
    deliver="origin",
    enabled_toolsets=["terminal", "file"],
    prompt="""
Run /root/work/ContextPilot/scripts/hermes_contextpilot_monitor.py with
--out-dir /root/contextpilot/reports --since-hours 24, then run
analyze_hermes_context_opportunities.py with --since-hours 24. Read today's
Markdown reports and send a short summary: realized token savings, session
count, whether ContextPilot events were observed, and the top advisory
opportunities (clearly labeled as candidates, not realized). Do NOT read raw
conversation content. Do NOT modify source/config.
""",
)
```

For a weekly trend, add a second job with `--since-hours 168` on a `0 5 * * 1`
schedule. Both stay strictly read-only.

### Step 7 — Propose improvements (do NOT auto-apply risky changes)

From the reports, write a prioritized proposal. **Never** auto-enable context
**routing**, context **dropping**, or **summarization** based on shadow numbers.
Those are high-recall-sensitive changes that can silently drop needed context;
they require the accuracy gate plus human sign-off.

Before any ContextPilot change ships, run a fixed golden eval set and require:

- no task-success regression,
- no drop in context recall beyond the chosen threshold,
- no unsafe raw-content leakage in reports,
- no increase in failed tool calls.

If any gate fails, hold the proposal and require human review.

### Step 8 — Safe path for code/config changes

For anything beyond a read-only report, follow this gate every time:

1. **Branch.** Make changes on a dedicated branch; never on `main`. No
   destructive git operations, no commit/push unless the user explicitly asks.
2. **Tests.** Add/extend tests and run the relevant suite (see below). A change
   to analysis or routing logic must ship with coverage.
3. **Privacy / no-raw-content check.** Re-confirm no report path can emit raw
   conversation/tool/system text, reasoning, or raw session ids. Keep the
   `write_report` forbidden-key guard intact.
4. **Independent review.** Get a second, independent review (human or a separate
   reviewing agent) focused on correctness, recall safety, and privacy before
   merge.

### Optional — delegated coding + independent verification

If the user has a coding-agent workflow, you may delegate the *implementation*
of an approved proposal to a coding agent (e.g. Claude Code) on a branch, and
then run **independent verification** in Hermes (re-run tests, the privacy
guard, and the accuracy gate) rather than trusting the author's own check. This
two-party split (one writes, another verifies) is recommended but generic — any
"author + independent reviewer" arrangement satisfies the gate. The skill itself
never merges; a human approves.

## Report locations (quick reference)

| Tool | Scope | Default output |
|------|-------|----------------|
| `hermes_contextpilot_monitor.py` | metadata only | `~/contextpilot/reports/daily_YYYY-MM-DD.{json,md}` |
| `analyze_hermes_context_opportunities.py` | content-aware (hashes only in reports) | `~/contextpilot/opportunities/opportunities_YYYY-MM-DD.{json,md}` |

## Relevant tests

```bash
python -m pytest tests/test_hermes_contextpilot_monitor.py \
  tests/test_hermes_context_opportunity_analyzer.py \
  tests/test_contextpilot_self_evolve_skill.py -q
```

## Hard rules (never violate)

- Observe and **propose** only — never auto-apply routing/drop/summarization.
- Reports never contain raw conversation/tool/system text, reasoning, or raw
  session ids; session ids are salted hashes only.
- Realized savings and advisory/shadow candidate tokens are reported separately.
- `--force` install only for an intentional update/reinstall.
- Code/config changes require: branch, tests, privacy check, independent review.
- No destructive git operations; no commit/push unless the user asks.
