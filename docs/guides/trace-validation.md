# Trace-derived validation sets for ContextPilot

ContextPilot changes can reduce token usage only if they preserve task accuracy.
For any future change that mutates the LLM-bound payload, do **not** rely on a
single live run. First build or reuse a fixed validation set derived from local
Hermes traces, then run the validation gate against the candidate mode.

## Privacy model

- The builder reads the local Hermes SQLite state DB in read-only mode.
- The generated JSONL corpus contains raw replay content and is **local-only**.
  Do not commit it, upload it, or paste it into reviews.
- The default output directory is outside the repo:
  `~/contextpilot/validation_sets/`.
- The in-repo convenience directory `.contextpilot_validation/` is gitignored.
- Reports and manifests are metadata-only: salted case ids, counters, enums and
  pass/fail flags. They must not contain raw conversation, tool output, system
  prompt, reasoning, API keys, or session ids.

## Build a validation set

Conservative last-24h sample:

```bash
python scripts/build_trace_validation_set.py
```

Heavier all-history sample for accuracy-sensitive changes:

```bash
python scripts/build_trace_validation_set.py \
  --all-sessions \
  --min-input-tokens 20000 \
  --limit 50 \
  --out ~/contextpilot/validation_sets
```

Exclude system/skill prompts if the change does not touch prompt handling:

```bash
python scripts/build_trace_validation_set.py --no-system-prompt
```

The command prints a privacy-safe JSON object with the corpus and manifest paths.
Only the corpus file contains raw replay content.

## Run the validation gate

Validate the current prompt-dedup canary candidate:

```bash
python scripts/run_trace_validation.py \
  ~/contextpilot/validation_sets/validation_set_YYYY-MM-DD.jsonl \
  --candidate-mode canary \
  --format markdown
```

Use the environment-configured mode instead:

```bash
CONTEXTPILOT_PROMPT_DEDUP_MODE=canary \
python scripts/run_trace_validation.py \
  ~/contextpilot/validation_sets/validation_set_YYYY-MM-DD.jsonl
```

Optional exact-token accounting, only when an exact tokenizer backend is
available:

```bash
python scripts/run_trace_validation.py \
  ~/contextpilot/validation_sets/validation_set_YYYY-MM-DD.jsonl \
  --candidate-mode canary \
  --tokenizer tiktoken:cl100k_base
```

If no tokenizer is configured, the report says actual-token savings are
`unavailable`. It does not substitute chars/4 as actual tokens.

## Gate semantics

The runner compares a baseline `off` pass with the candidate pass and exits
non-zero on any failed invariant:

- message count preserved;
- message order and roles preserved;
- protected user/assistant/tool/system content preserved;
- mutation confined to the explicitly allowed scope;
- realized savings accounting matches the actual processed-payload before/after
  character delta.

For the current canary, the only allowed mutation scope is
`same_type_skill_prompt_only`: later exact duplicate `skill_prompt` lines may be
replaced with a deterministic ContextPilot reference if and only if the reference
is shorter and the line is not safety-denylisted.

## When this is required

Run this gate before merging or enabling any change that can affect accuracy,
including:

- prompt/system/skill dedup or replacement;
- context routing, filtering, summarization or dropping;
- parent/child artifact aggregation rewrites;
- changes to runtime optimization order or telemetry accounting that affect the
  LLM-bound payload.

Passing this gate is necessary but not always sufficient. High-risk changes such
as system prompt replacement or context dropping still need shadow telemetry,
offline A/B evidence, golden evals and default-off canary rollout before default
enablement.
