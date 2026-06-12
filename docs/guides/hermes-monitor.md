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
- ContextPilot telemetry coverage and savings ratios,
- **Worker Context Routing shadow labels** for future router training/eval,
- **Parent Aggregation Artifact telemetry** (exact duplicate worker/parent
  artifacts grouped by hash) for future parent-aggregation dedup eval.

### LLM-bound block redundancy

The analyzer also performs an **LLM-bound block scan** that looks *only* at
content Hermes would actually send to a model, and reports where the same block
is paid for more than once:

- `sessions.system_prompt`, classified heuristically as `system_prompt` or
  `skill_prompt` (skill frontmatter / "use this skill" style cues),
- active `messages.content` for roles `system` / `user` / `assistant` / `tool`,
  bucketed as `user_prompt`, `assistant_context`, `tool_result`, etc.,
- tool-result messages (`role='tool'` or `tool_name` set) as `tool_result`.

Inactive messages are skipped when an `active` column exists, and archived
sessions (and their messages) are skipped when an `archived` column exists. Each
block is split line-wise, fingerprinted with a salted SHA-256 hash, and
aggregated. The report then shows:

- **redundancy by block type** — per-type block / unique / repeated counts and
  estimated redundant tokens,
- **cross-type repeated blocks** — the headline signal: a single fingerprint
  observed in 2+ block types (e.g. the same chunk shipped from a skill/system
  prompt *and* a tool result *and* a user prompt). Reported only as a hash plus
  per-type counters — never the raw text.

### Worker Context Routing shadow mode

The analyzer now includes a **Worker Context Routing — shadow mode** section by
default. This is P0 data collection only: it never drops, summarizes, or mutates
context. It fingerprints each LLM-bound block and emits only low-cardinality
labels/counters such as:

- `policy_must_keep` for user/system/skill prompts and explicit safety /
  acceptance constraints,
- `direct_task_hint` for short actionable task/error hints,
- `likely_relevant` for conservative default-keep blocks,
- `summarizable_candidate` / `likely_drop_candidate` for large or repeated
  tool-like blocks that a future router might route away. Large diagnostic logs
  containing `error:` / `failed` / `traceback` cues are still only advisory
  summarization candidates, not must-drop decisions.

The report includes estimated advisory candidate tokens and salted candidate
block hashes. These are **not realized savings** and must be treated as training
/ evaluation data for a future high-recall router. Use
`--disable-worker-routing-shadow` only when you want to omit this section from a
scan.

### Parent Aggregation Artifacts — shadow mode

The analyzer also includes a **Parent Aggregation Artifacts — shadow mode**
section by default. This is **P0 telemetry only**: it collects data so a future
parent-aggregation dedup can be evaluated offline. It never drops, summarizes,
replaces, or mutates any context.

When a parent/orchestrator aggregates results from several workers, the same
artifact body (a test log, a diff, a file dump, a review summary, ...) is often
carried into the parent's LLM context once per worker and again in the parent's
own roll-up — paying for the same tokens several times. The analyzer groups
**EXACT** artifact bodies by salted content hash (near-duplicates never group),
classifies each body with a deterministic heuristic kind, and emits only
low-cardinality metadata + counters:

- `artifact_kind` — one of `test_log`, `terminal_output`, `file_content`,
  `diff`, `error_trace`, `review_findings`, `benchmark_result`,
  `worker_summary`, `unknown_large_block` (deterministic, first-match-wins),
- per-kind summary — distinct bodies, occurrences, duplicate-group count,
  estimated tokens, and advisory duplicate tokens,
- **provenance** — per duplicate group, `source_type_counts` such as
  `tool_result xN` and `assistant_context xM`, plus a deterministically chosen
  `canonical_source_type` (the dominant origin, tie-broken alphabetically),
- top duplicate artifact groups, reported **only** as a salted `content_hash`
  plus counters.

`est_duplicate_tokens` is computed as `(occurrences - 1) * est_tokens` and is an
**advisory upper bound** on what a future parent dedup might save — **not a
realized saving**, and payloads are never changed. No raw artifact / worker /
tool / system text, and no raw session ids, are ever emitted. Only sizeable
blocks (`>= --min-artifact-chars`, default 400) from parent/worker output origins
(`assistant_context` and `tool_result`) are considered candidates, so prompt
boilerplate and short hints never enter this telemetry. Use
`--disable-parent-aggregation` to omit this section from a scan.

Use `--all-sessions` to ignore the `--since-hours` window and scan **all**
non-archived sessions and active messages (useful for a one-shot, whole-history
audit rather than a rolling daily window):

```bash
# rolling daily window
python scripts/analyze_hermes_context_opportunities.py \
  --state-db /root/.hermes/state.db \
  --telemetry-file ~/.hermes/contextpilot/telemetry.jsonl \
  --out-dir ~/contextpilot/opportunities \
  --since-hours 24

# whole-history audit across every session and LLM-bound block
python scripts/analyze_hermes_context_opportunities.py \
  --state-db /root/.hermes/state.db \
  --telemetry-file ~/.hermes/contextpilot/telemetry.jsonl \
  --out-dir ~/contextpilot/opportunities \
  --all-sessions
```

Outputs:

- `~/contextpilot/opportunities/opportunities_YYYY-MM-DD.json`
- `~/contextpilot/opportunities/opportunities_YYYY-MM-DD.md`

Each estimated "wasted tokens" figure is a heuristic (chars / 4); treat the
report as a prioritized list of candidates and validate against the accuracy
gate below before changing ContextPilot config or code. A defensive guard in
`write_report` refuses to emit any forbidden raw-content key, so the reports are
safe to ship from an unattended cron job.

## Self-evolve skill (Hermes)

The monitor and analyzer above are bundled into a reusable Hermes skill so users
can run the same safe loop — install/enable ContextPilot, collect
telemetry/shadow data, analyze realized savings vs advisory candidates, and
propose improvements under strict safety gates:

- Skill path: `skills/contextpilot-self-evolve/SKILL.md`

The skill is **proposal-only**: it never auto-applies context routing, dropping,
or summarization, and it enforces the same privacy boundary (reports never emit
raw conversation/tool/system text, reasoning, or raw session ids).

To use it in Hermes, copy or load the skill into your Hermes skills directory,
then invoke it by name:

```bash
# copy into your Hermes skills directory (adjust path to your install)
mkdir -p ~/.hermes/skills/contextpilot-self-evolve
cp skills/contextpilot-self-evolve/SKILL.md \
   ~/.hermes/skills/contextpilot-self-evolve/SKILL.md
```

Then ask Hermes to run the **contextpilot-self-evolve** skill. It walks through
install/verify, the metadata-only monitor, the content-aware analyzer for
`--since-hours 24` and `168`, interpretation of realized vs advisory tokens,
optional read-only cron jobs, and the branch/tests/privacy/independent-review
gate required before any code or config change ships.

## Accuracy gate

This monitor only measures token/cost savings and operational signals. Before shipping ContextPilot changes, run a fixed golden eval set and require:

- no task-success regression,
- no drop in context recall beyond the chosen threshold,
- no unsafe raw-content leakage in reports,
- no increase in failed tool calls.

If any gate fails, hold proposals and require human review.
