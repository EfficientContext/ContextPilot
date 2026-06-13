"""Report assembly and serialization (privacy-safe output).

``build_report`` runs every detector over the in-memory carriers and assembles
a privacy-safe ``OpportunityReport`` (hashes + counters + enums only).
``write_report`` serializes it to JSON + Markdown, guarded by
``_assert_no_forbidden_keys`` so no raw content can ever reach disk.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .aggregation import (
    DEFAULT_MIN_ARTIFACT_CHARS,
    analyze_parent_aggregation_artifacts,
)
from .detection import (
    analyze_llm_bound_blocks,
    detect_exact_duplicate_tool_outputs,
    detect_repeated_blocks,
    summarize_tool_sizes,
)
from .models import (
    DEFAULT_LARGE_OUTPUT_CHARS,
    DEFAULT_MIN_BLOCK_CHARS,
    DEFAULT_MIN_BLOCK_REPEAT,
    DEFAULT_TOP_N,
    HeavySession,
    OpportunityReport,
    TelemetryCoverage,
    _est_tokens,
    _LLMContent,
    _ToolMessage,
)
from .privacy import _assert_no_forbidden_keys, _salt_fingerprint
from .routing import analyze_worker_routing_shadow


def build_report(
    *,
    date: str,
    since_hours: int,
    salt: str,
    tool_messages: list[_ToolMessage],
    heavy_sessions: list[HeavySession],
    telemetry: TelemetryCoverage,
    llm_contents: list[_LLMContent] | None = None,
    all_sessions: bool = False,
    min_block_chars: int = DEFAULT_MIN_BLOCK_CHARS,
    min_block_repeat: int = DEFAULT_MIN_BLOCK_REPEAT,
    large_output_chars: int = DEFAULT_LARGE_OUTPUT_CHARS,
    top_n: int = DEFAULT_TOP_N,
    worker_routing_shadow: bool = True,
    parent_aggregation_shadow: bool = True,
    min_artifact_chars: int = DEFAULT_MIN_ARTIFACT_CHARS,
) -> OpportunityReport:
    dups = detect_exact_duplicate_tool_outputs(tool_messages, salt=salt, top_n=top_n)
    blocks = detect_repeated_blocks(
        tool_messages,
        salt=salt,
        min_block_chars=min_block_chars,
        min_repeat=min_block_repeat,
        top_n=top_n,
    )
    sizes = summarize_tool_sizes(
        tool_messages, large_output_chars=large_output_chars, top_n=top_n
    )

    llm_contents = llm_contents or []
    block_type_stats, cross_groups = analyze_llm_bound_blocks(
        llm_contents,
        salt=salt,
        min_block_chars=min_block_chars,
        min_repeat=min_block_repeat,
        top_n=top_n,
    )

    worker_routing = analyze_worker_routing_shadow(
        llm_contents,
        salt=salt,
        large_output_chars=large_output_chars,
        min_repeat=min_block_repeat,
        top_n=top_n,
        enabled=worker_routing_shadow,
    )

    parent_aggregation = analyze_parent_aggregation_artifacts(
        llm_contents,
        salt=salt,
        min_artifact_chars=min_artifact_chars,
        top_n=top_n,
        enabled=parent_aggregation_shadow,
    )

    total_chars = sum(len(m.content) for m in tool_messages)
    dup_wasted = sum(d.est_wasted_tokens for d in dups)
    block_wasted = sum(b.est_wasted_tokens for b in blocks)
    cross_wasted = sum(g.est_wasted_tokens for g in cross_groups)

    notes = [
        "content-aware analysis: message/tool text was hashed in-memory only and never written to reports",
        "all identifiers are salted SHA-256 fingerprints; counters are aggregates",
        "wasted-token figures are heuristic estimates (chars/4); validate before acting",
        "session 'source', 'tool_name', and block_type are emitted verbatim as low-cardinality enums, not raw text",
        "llm-bound scan covers only content sent to the LLM: system/skill prompts, active user/assistant/tool messages",
        "worker-routing section is SHADOW MODE P0: it labels blocks for a future router but never drops/summarizes context",
        "parent-aggregation section is SHADOW MODE P0 telemetry: it groups exact artifact bodies but never dedups/replaces context",
    ]
    if all_sessions:
        notes.append("all-sessions mode: time window ignored; scanned all non-archived sessions/active messages")
    if not tool_messages:
        notes.append("no tool-output messages observed in the selected window")
    if not llm_contents:
        notes.append("no llm-bound content observed in the selected window")

    return OpportunityReport(
        date=date,
        since_hours=since_hours,
        all_sessions=all_sessions,
        salt_fingerprint=_salt_fingerprint(salt),
        tool_message_count=len(tool_messages),
        total_tool_output_chars=total_chars,
        total_tool_output_est_tokens=_est_tokens(total_chars),
        exact_duplicate_groups=dups,
        duplicate_tool_output_groups=len(dups),
        duplicate_tool_output_wasted_tokens=dup_wasted,
        repeated_block_count=len(blocks),
        repeated_block_wasted_tokens=block_wasted,
        repeated_blocks=blocks,
        large_tool_outputs_by_tool=sizes,
        heavy_sessions=heavy_sessions,
        telemetry=telemetry,
        llm_bound_item_count=len(llm_contents),
        llm_block_types=block_type_stats,
        cross_type_block_groups=cross_groups,
        cross_type_wasted_tokens=cross_wasted,
        worker_routing=worker_routing,
        parent_aggregation=parent_aggregation,
        notes=notes,
    )


def write_report(report: OpportunityReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = asdict(report)
    _assert_no_forbidden_keys(data)

    json_path = out_dir / f"opportunities_{report.date}.json"
    md_path = out_dir / f"opportunities_{report.date}.md"
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    t = report.telemetry
    window = "all sessions (no time window)" if report.all_sessions else f"last {report.since_hours}h"
    md = [
        f"# ContextPilot Hermes opportunity scan — {report.date}",
        "",
        f"Window: {window}",
        f"Salt fingerprint: `{report.salt_fingerprint}`",
        "",
        "## Summary",
        f"- Tool-output messages: {report.tool_message_count}",
        f"- Total tool-output tokens (est): {report.total_tool_output_est_tokens}",
        f"- Exact duplicate groups: {report.duplicate_tool_output_groups} "
        f"(~{report.duplicate_tool_output_wasted_tokens} wasted tokens)",
        f"- Repeated blocks: {report.repeated_block_count} "
        f"(~{report.repeated_block_wasted_tokens} wasted tokens)",
        f"- LLM-bound items scanned: {report.llm_bound_item_count}",
        f"- Cross-type repeated blocks: {len(report.cross_type_block_groups)} "
        f"(~{report.cross_type_wasted_tokens} wasted tokens)",
        f"- Telemetry: {t.events} events, {t.chars_saved} chars saved by processing; "
        f"derived chars/4 tokens={t.tokens_saved}, ratio={t.coverage_ratio_pct}%",
        f"- Worker routing (shadow): {report.worker_routing.classified_block_count} blocks "
        f"classified, {report.worker_routing.must_keep_block_count} must-keep, "
        f"~{report.worker_routing.est_candidate_tokens_total} advisory candidate tokens",
        f"- Parent aggregation (shadow): {report.parent_aggregation.duplicate_group_count} "
        f"duplicate artifact groups, "
        f"~{report.parent_aggregation.est_duplicate_tokens} advisory duplicate tokens",
        "",
        "## LLM-bound redundancy by block type",
    ]
    for bt in report.llm_block_types:
        md.append(
            f"- {bt.block_type}: items={bt.item_count} blocks={bt.block_count} "
            f"unique={bt.unique_block_count} repeated={bt.repeated_block_count} "
            f"~redundant={bt.est_redundant_tokens} tokens"
        )
    md.append("")
    md.append("## Cross-type repeated blocks (same block, multiple sources)")
    for g in report.cross_type_block_groups:
        spread = ", ".join(f"{tc.block_type}x{tc.count}" for tc in g.type_occurrences)
        md.append(
            f"- `{g.block_hash}` types=[{', '.join(g.block_types)}] ({spread}) "
            f"chars={g.char_length} ~wasted={g.est_wasted_tokens} tokens"
        )
    md.append("")
    md.append("## Top exact-duplicate tool outputs")
    for d in report.exact_duplicate_groups:
        md.append(
            f"- `{d.content_hash}` tool={d.tool_name} x{d.occurrences} "
            f"chars={d.char_length} ~wasted={d.est_wasted_tokens} tokens"
        )
    md.append("")
    md.append("## Top repeated blocks")
    for b in report.repeated_blocks:
        md.append(
            f"- `{b.block_hash}` x{b.occurrences} chars={b.char_length} "
            f"~wasted={b.est_wasted_tokens} tokens"
        )
    md.append("")
    md.append("## Large tool outputs by tool")
    for s in report.large_tool_outputs_by_tool:
        md.append(
            f"- {s.tool_name}: count={s.output_count} total_chars={s.total_chars} "
            f"max={s.max_chars} avg={s.avg_chars} large(>=thresh)={s.large_output_count}"
        )
    md.append("")
    md.append("## Heavy sessions (hashed)")
    for h in report.heavy_sessions:
        md.append(
            f"- `{h.session_hash}` source={h.source} input={h.input_tokens} "
            f"output={h.output_tokens} msgs={h.message_count} tools={h.tool_call_count} "
            f"apis={h.api_call_count}"
        )
    md.append("")
    md.append("## Telemetry from processed payload")
    md.extend(
        [
            f"- Events: {t.events}",
            f"- Chars saved by ContextPilot processing: {t.chars_saved}",
            f"- Derived chars/4 token counter: {t.tokens_saved} (not actual tokenizer/API usage)",
            f"- Avg derived chars/4 tokens / event: {t.avg_tokens_saved_per_event}",
            f"- Derived ratio: {t.coverage_ratio_pct}%",
            f"- Malformed records skipped: {t.malformed_records_skipped}",
        ]
    )
    md.append("")
    wr = report.worker_routing
    md.append("## Worker Context Routing — shadow mode (P0, advisory only)")
    if not wr.enabled:
        md.append("- disabled")
    else:
        md.append(
            f"- Items classified: {wr.item_count} "
            f"(distinct fingerprints: {wr.classified_block_count}, "
            f"occurrences: {wr.total_occurrences})"
        )
        md.append(
            f"- Must-keep: {wr.must_keep_block_count} blocks / "
            f"{wr.must_keep_occurrence_count} occurrences "
            f"(~{wr.est_must_keep_tokens} tokens, never routable)"
        )
        md.append(
            f"- Advisory candidate tokens: ~{wr.est_candidate_tokens_total} "
            f"(drop ~{wr.est_drop_candidate_tokens}, "
            f"summarize ~{wr.est_summarizable_candidate_tokens}) — NOT a realized saving"
        )
        md.append("")
        md.append("### Router labels")
        for lc in wr.label_counts:
            md.append(
                f"- {lc.route_label}: blocks={lc.block_count} "
                f"occ={lc.occurrence_count} tokens={lc.total_est_tokens} "
                f"~candidate={lc.est_candidate_tokens}"
            )
        md.append("")
        md.append("### Reason codes (block_type / label / reason)")
        for rc in wr.reason_counts:
            md.append(
                f"- {rc.block_type} / {rc.route_label} / {rc.reason_code}: "
                f"blocks={rc.block_count} occ={rc.occurrence_count} "
                f"tokens={rc.total_est_tokens} ~candidate={rc.est_candidate_tokens}"
            )
        md.append("")
        md.append("### Top routable-candidate blocks (hashed)")
        for cb in wr.top_candidate_blocks:
            md.append(
                f"- `{cb.block_hash}` type={cb.block_type} "
                f"label={cb.route_label} reason={cb.reason_code} "
                f"x{cb.occurrences} chars={cb.char_length} ~candidate={cb.est_candidate_tokens}"
            )
    md.append("")
    pa = report.parent_aggregation
    md.append("## Parent Aggregation Artifacts — shadow mode")
    if not pa.enabled:
        md.append("- disabled")
    else:
        md.append(
            f"- Candidate artifact items: {pa.item_count} "
            f"(distinct bodies: {pa.artifact_body_count}, "
            f"occurrences: {pa.total_occurrences})"
        )
        md.append(
            f"- Duplicate artifact groups: {pa.duplicate_group_count} "
            f"(~{pa.est_duplicate_tokens} advisory duplicate tokens of "
            f"~{pa.est_total_tokens} distinct-body tokens) — NOT a realized saving, "
            f"payloads are unchanged"
        )
        md.append("")
        md.append("### By artifact kind")
        for ks in pa.by_kind:
            md.append(
                f"- {ks.artifact_kind}: bodies={ks.group_count} "
                f"occ={ks.occurrence_count} dup_groups={ks.duplicate_group_count} "
                f"tokens={ks.est_tokens} ~dup={ks.est_duplicate_tokens}"
            )
        md.append("")
        md.append("### Provenance (artifact source types)")
        for sc in pa.source_type_counts:
            md.append(f"- {sc.source_type}: {sc.count}")
        md.append("")
        md.append("### Top duplicate artifact groups (hashed)")
        for g in pa.top_duplicate_groups:
            spread = ", ".join(
                f"{sc.source_type}x{sc.count}" for sc in g.source_type_counts
            )
            md.append(
                f"- `{g.content_hash}` kind={g.artifact_kind} "
                f"canonical={g.canonical_source_type} x{g.occurrences} "
                f"({spread}) chars={g.char_length} ~dup={g.est_duplicate_tokens} tokens"
            )
    md.append("")
    md.append("## Notes")
    for note in report.notes:
        md.append(f"- {note}")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return json_path, md_path
