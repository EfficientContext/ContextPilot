# General provenance linker training/shadow plan

## Goal

Move beyond artifact exact dedup by learning a **claim → evidence provenance graph**. Coding is only one domain example; the schema is domain-general.

## Data boundary

- **Committed dataset:** `datasets/provenance_linking/synthetic_v1.jsonl` — synthetic, safe to version.
- **Local real-trace dataset:** `~/contextpilot/provenance_datasets/*.jsonl` — raw Hermes trace content, never commit.
- **Schema:** examples contain `blocks`, extracted `claims`, optional `gold_links`, and optional `shadow_links`.

## Shadow first

Online payload is never changed by this path. The shadow linker only emits candidate links:

```text
claim_id -> evidence_block_id[start:end], relation, confidence, method
```

Initial baseline is deliberately cheap/high-precision lexical matching. It is expected to miss semantic support; that gap is the training target.

## Training target

Train or distill a provenance linker/verifier to replace `shadow_lexical_v1` scoring:

```text
input: claim + top-k candidate evidence snippets + metadata
output: support relation + evidence span + confidence
```

Relations should include `copied`, `extracted_support`, `summarized_support`, `aggregated_support`, `contradicts`, and `insufficient`.

## Action policy

The trained model still does not delete context. Context actions require a deterministic gate:

1. source exists and is earlier than claim;
2. pointer is recoverable;
3. protected user/system/developer/skill content is untouched;
4. source is not stale;
5. validator passes;
6. online path only consumes cached/shadow graph.

## Metrics

- shadow link precision/recall against gold examples;
- unsupported/contradicted claim rate;
- fold opportunity in chars/tokens;
- background latency only; zero user-facing e2e blocking.

## Current baseline result

`evals/provenance_shadow_synthetic_v1.json`:

- 3 examples, 4 gold links;
- precision 1.00, recall 0.75;
- high precision but misses aggregated coding claim evidence, showing why a learned model is needed.
