"""Parent Aggregation Artifacts — SHADOW MODE (P0 telemetry only).

When a parent/orchestrator aggregates results from several workers, the same
artifact body (a test log, a diff, a file dump, a review summary, ...) is often
carried into the parent's LLM context once per worker and again in the parent's
own roll-up -- paying for the same tokens several times. This section collects
*telemetry only* so a future parent-aggregation dedup can be evaluated offline:
it groups EXACT artifact bodies by salted content hash, classifies each body
with a deterministic heuristic kind, and emits low-cardinality metadata +
counters. It NEVER drops, summarizes, replaces, or mutates any context, and it
NEVER emits raw artifact text, worker text, tool output, session ids, or
system prompts.
"""
from __future__ import annotations

from typing import Iterable

from .models import (
    ArtifactKindStat,
    ArtifactSourceCount,
    ParentAggregationArtifacts,
    ParentAggregationGroup,
    _est_tokens,
    _LLMContent,
)
from .privacy import _salted_hash

# Heuristic P0 artifact kinds. Low-cardinality enums describing the *shape* of an
# aggregation artifact, never its text. Classification is deterministic.
ARTIFACT_KINDS = (
    "test_log",
    "terminal_output",
    "file_content",
    "diff",
    "error_trace",
    "review_findings",
    "benchmark_result",
    "worker_summary",
    "unknown_large_block",
)

# Conservative floor: only sizeable blocks are treated as candidate aggregation
# artifacts, so short prompts/hints never enter parent-aggregation telemetry.
DEFAULT_MIN_ARTIFACT_CHARS = 400

# Parent aggregation P0 focuses on content produced by workers/tools and then
# carried into the parent context. System/skill/user prompts are analyzed by the
# LLM-bound redundancy and worker-routing sections, but excluding them here keeps
# parent artifact telemetry from being polluted by prompt boilerplate.
PARENT_AGGREGATION_SOURCE_TYPES = ("assistant_context", "tool_result")


def classify_artifact_kind(content: str) -> str:
    """Deterministically classify a candidate aggregation artifact body.

    Pure P0 heuristic over in-memory text; returns a low-cardinality enum from
    ``ARTIFACT_KINDS`` and never the text. The check order is fixed so the same
    body always yields the same kind (first match wins).
    """
    low = content.lower()
    stripped = content.lstrip()

    # 1. Unified diff / patch.
    if (
        stripped.startswith("diff --git")
        or stripped.startswith("--- a/")
        or stripped.startswith("@@ ")
        or "\n@@ " in content
        or ("\n--- " in content and "\n+++ " in content)
    ):
        return "diff"

    # 2. Test/pytest log (checked before error_trace: a failing test log may
    #    embed a traceback but is still fundamentally a test log).
    if (
        "pytest" in low
        or "test session starts" in low
        or " passed in " in low
        or " failed in " in low
        or ("passed" in low and "failed" in low)
        or "=== " in content
    ):
        return "test_log"

    # 3. Error / exception trace.
    if (
        "traceback (most recent call last)" in low
        or "\n  at " in content
        or "stack trace" in low
        or ("exception" in low and "error" in low)
    ):
        return "error_trace"

    # 4. Benchmark / perf result.
    if (
        "benchmark" in low
        or "ops/sec" in low
        or "ops/s" in low
        or "req/sec" in low
        or "throughput" in low
        or "latency" in low
        or "iterations/sec" in low
    ):
        return "benchmark_result"

    # 5. Code-review findings.
    if (
        "code review" in low
        or "review findings" in low
        or "severity:" in low
        or "vulnerab" in low
        or "## findings" in low
    ):
        return "review_findings"

    # 6. File content / source dump (cat -n style numbering or code cues).
    if (
        "\n     1\t" in content
        or "\n   1\t" in content
        or "def " in content
        or "class " in content
        or "\nimport " in content
        or "#include" in content
        or "function " in content
    ):
        return "file_content"

    # 7. Worker / aggregation summary. Checked after source-code cues so files
    #    mentioning workers are still labeled as file_content.
    if (
        "## summary" in low
        or "in summary" in low
        or "summary:" in low
        or "tl;dr" in low
        or "aggregat" in low
        or "worker" in low
    ):
        return "worker_summary"

    # 8. Terminal / shell session output.
    if (
        "\n$ " in content
        or stripped.startswith("$ ")
        or "\n# " in content
        or "user@" in low
        or "bash-" in low
        or "exit code" in low
    ):
        return "terminal_output"

    # 9. Fallback: a large block we could not confidently classify.
    return "unknown_large_block"


def analyze_parent_aggregation_artifacts(
    contents: Iterable[_LLMContent],
    *,
    salt: str,
    min_artifact_chars: int,
    top_n: int,
    enabled: bool = True,
) -> ParentAggregationArtifacts:
    """Group EXACT aggregation-artifact bodies and emit provenance telemetry.

    P0 telemetry/advisory only: no context is dropped, summarized, replaced, or
    mutated. Each sizeable LLM-bound block is fingerprinted by EXACT salted
    content hash (near-duplicates never group), classified with a deterministic
    heuristic kind, and rolled up into low-cardinality metadata + counters.
    ``est_duplicate_tokens`` is an advisory upper bound on what a *future* parent
    dedup might save -- never a realized saving. No raw artifact/worker/tool/
    system text, and no raw session ids, are ever emitted.
    """
    if not enabled:
        return ParentAggregationArtifacts(
            enabled=False,
            item_count=0,
            artifact_body_count=0,
            total_occurrences=0,
            duplicate_group_count=0,
            est_total_tokens=0,
            est_duplicate_tokens=0,
            by_kind=[],
            source_type_counts=[],
            top_duplicate_groups=[],
            notes=["parent-aggregation artifact analysis disabled via flag"],
        )

    # --- group sizeable bodies by EXACT salted content hash ----------------
    groups: dict[str, dict] = {}
    item_count = 0
    source_totals: dict[str, int] = {}
    for item in contents:
        content = item.content
        bt = item.block_type
        if bt not in PARENT_AGGREGATION_SOURCE_TYPES:
            continue
        if not content or len(content) < min_artifact_chars:
            continue
        item_count += 1
        source_totals[bt] = source_totals.get(bt, 0) + 1
        h = _salted_hash(content, salt)
        g = groups.get(h)
        if g is None:
            groups[h] = {
                "char_length": len(content),
                "occurrences": 1,
                "sources": {bt: 1},
                # classify once from in-memory text; never stored/emitted.
                "kind": classify_artifact_kind(content),
            }
        else:
            g["occurrences"] += 1
            g["sources"][bt] = g["sources"].get(bt, 0) + 1

    # --- per-kind rollup + per-group records -------------------------------
    kind_agg: dict[str, dict] = {}
    group_records: list[ParentAggregationGroup] = []
    total_occurrences = 0
    est_total_tokens = 0
    est_duplicate_tokens = 0
    duplicate_group_count = 0

    for h, g in groups.items():
        occ = g["occurrences"]
        char_len = g["char_length"]
        est = _est_tokens(char_len)
        dup_tokens = est * (occ - 1)
        kind = g["kind"]
        is_dup = occ >= 2

        total_occurrences += occ
        est_total_tokens += est
        est_duplicate_tokens += dup_tokens
        if is_dup:
            duplicate_group_count += 1

        ka = kind_agg.setdefault(
            kind,
            {"groups": 0, "occ": 0, "dups": 0, "est": 0, "dup_tokens": 0},
        )
        ka["groups"] += 1
        ka["occ"] += occ
        ka["est"] += est
        ka["dup_tokens"] += dup_tokens
        if is_dup:
            ka["dups"] += 1

        if is_dup:
            # Provenance counts, sorted by source_type for determinism.
            source_counts = [
                ArtifactSourceCount(source_type=st, count=c)
                for st, c in sorted(g["sources"].items())
            ]
            # Canonical source: dominant origin, tie-broken alphabetically.
            canonical = min(
                g["sources"].items(), key=lambda kv: (-kv[1], kv[0])
            )[0]
            group_records.append(
                ParentAggregationGroup(
                    content_hash=h,
                    artifact_kind=kind,
                    canonical_source_type=canonical,
                    occurrences=occ,
                    char_length=char_len,
                    est_tokens=est,
                    est_duplicate_tokens=dup_tokens,
                    source_type_counts=source_counts,
                )
            )

    by_kind = [
        ArtifactKindStat(
            artifact_kind=kind,
            group_count=kind_agg[kind]["groups"],
            occurrence_count=kind_agg[kind]["occ"],
            duplicate_group_count=kind_agg[kind]["dups"],
            est_tokens=kind_agg[kind]["est"],
            est_duplicate_tokens=kind_agg[kind]["dup_tokens"],
        )
        for kind in ARTIFACT_KINDS
        if kind in kind_agg
    ]
    source_type_counts = [
        ArtifactSourceCount(source_type=st, count=c)
        for st, c in sorted(source_totals.items())
    ]
    group_records.sort(
        key=lambda g: (g.est_duplicate_tokens, g.occurrences, g.content_hash),
        reverse=True,
    )

    notes = [
        "SHADOW MODE P0: telemetry only -- no aggregation artifact was deduped, replaced, summarized, or mutated",
        "artifact_kind/source_type/canonical_source_type are low-cardinality enums; content_hash is a salted SHA-256 fingerprint",
        "grouping is EXACT (same salted content hash): near-duplicate artifacts never group",
        "est_duplicate_tokens is ADVISORY ((occurrences-1) * est_tokens), an upper bound for a FUTURE parent dedup -- not a realized saving",
        "provenance source_type_counts show how many copies came from each parent/worker output origin (assistant_context, tool_result)",
    ]

    return ParentAggregationArtifacts(
        enabled=True,
        item_count=item_count,
        artifact_body_count=len(groups),
        total_occurrences=total_occurrences,
        duplicate_group_count=duplicate_group_count,
        est_total_tokens=est_total_tokens,
        est_duplicate_tokens=est_duplicate_tokens,
        by_kind=by_kind,
        source_type_counts=source_type_counts,
        top_duplicate_groups=group_records[:top_n],
        notes=notes,
    )
