#!/usr/bin/env python3
"""Privacy-safe Hermes context opportunity analyzer for ContextPilot.

Unlike ``hermes_contextpilot_monitor.py`` (which never reads message bodies),
this analyzer *does* inspect message content and tool outputs in order to find
concrete token-reduction opportunities: exact duplicate tool outputs, repeated
line/block fingerprints, oversized tool outputs per tool, heavy sessions, and
ContextPilot telemetry coverage.

It reads content only in-memory to compute salted hashes and aggregate
counters. Reports never contain raw message/tool text, system prompts, or raw
session ids -- only salted SHA-256 fingerprints and numeric aggregates. This
makes it safe to run continuously from a cron job and ship the reports.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

# Columns we are explicitly forbidden from EMITTING in any report. We may read
# message content in-memory for hashing, but it must never reach an output file.
FORBIDDEN_OUTPUT_KEYS = {
    "content",
    "system_prompt",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "tool_calls",
    "codex_reasoning_items",
    "codex_message_items",
}

# Tunables (overridable via CLI).
DEFAULT_MIN_BLOCK_CHARS = 40       # ignore trivial lines when fingerprinting
DEFAULT_MIN_BLOCK_REPEAT = 3       # a block must recur this often to be a "repeat"
DEFAULT_LARGE_OUTPUT_CHARS = 8000  # tool outputs at/above this are "large"
DEFAULT_TOP_N = 20
EST_CHARS_PER_TOKEN = 4


def _est_tokens(chars: int) -> int:
    return chars // EST_CHARS_PER_TOKEN


def _salted_hash(text: str, salt: str, *, length: int = 16) -> str:
    return hashlib.sha256(f"{salt}:{text}".encode("utf-8", "replace")).hexdigest()[:length]


def _salt_fingerprint(salt: str) -> str:
    # Confirms a salt was applied without revealing it.
    return hashlib.sha256(f"fingerprint:{salt}".encode()).hexdigest()[:12]


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


# ---------------------------------------------------------------------------
# Data structures (all privacy-safe: hashes + counters only)
# ---------------------------------------------------------------------------


@dataclass
class DuplicateToolOutput:
    content_hash: str
    tool_name: str | None
    occurrences: int
    char_length: int
    est_tokens: int
    est_wasted_tokens: int  # tokens spent re-sending identical output: (n-1) * est_tokens


@dataclass
class RepeatedBlock:
    block_hash: str
    occurrences: int
    char_length: int
    est_tokens: int
    est_wasted_tokens: int  # (n-1) * est_tokens


@dataclass
class ToolSizeStat:
    tool_name: str
    output_count: int
    total_chars: int
    max_chars: int
    avg_chars: int
    total_est_tokens: int
    large_output_count: int  # outputs >= large_output_chars threshold


@dataclass
class HeavySession:
    session_hash: str
    source: str | None
    input_tokens: int
    output_tokens: int
    message_count: int
    tool_call_count: int
    api_call_count: int


@dataclass
class TelemetryCoverage:
    events: int
    chars_saved: int
    tokens_saved: int
    avg_tokens_saved_per_event: float
    coverage_ratio_pct: float           # tokens_saved / (tokens_saved + total_input_tokens)
    malformed_records_skipped: int


@dataclass
class OpportunityReport:
    date: str
    since_hours: int
    salt_fingerprint: str
    tool_message_count: int
    total_tool_output_chars: int
    total_tool_output_est_tokens: int
    exact_duplicate_groups: list[DuplicateToolOutput]
    duplicate_tool_output_groups: int
    duplicate_tool_output_wasted_tokens: int
    repeated_block_count: int
    repeated_block_wasted_tokens: int
    repeated_blocks: list[RepeatedBlock]
    large_tool_outputs_by_tool: list[ToolSizeStat]
    heavy_sessions: list[HeavySession]
    telemetry: TelemetryCoverage
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class _ToolMessage:
    tool_name: str | None
    content: str


def load_tool_messages(
    db_path: Path, *, since_hours: int
) -> list[_ToolMessage]:
    """Load tool-output messages within the window.

    Content is returned for in-memory hashing only; callers must not emit it.
    A message is treated as tool output when ``role='tool'`` or ``tool_name``
    is set.
    """
    cutoff = dt.datetime.now(dt.timezone.utc).timestamp() - since_hours * 3600
    conn = _connect_readonly(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        if "content" not in cols:
            return []
        has_tool_name = "tool_name" in cols
        has_ts = "timestamp" in cols
        select_tool = "tool_name" if has_tool_name else "NULL AS tool_name"
        where = []
        params: list[object] = []
        if has_ts:
            where.append("timestamp >= ?")
            params.append(cutoff)
        tool_pred = "role = 'tool'"
        if has_tool_name:
            tool_pred = "(role = 'tool' OR tool_name IS NOT NULL)"
        where.append(tool_pred)
        sql = (
            f"SELECT {select_tool}, content FROM messages "
            f"WHERE {' AND '.join(where)}"
        )
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    out: list[_ToolMessage] = []
    for tool_name, content in rows:
        if content is None:
            continue
        out.append(_ToolMessage(tool_name=tool_name, content=str(content)))
    return out


def load_heavy_sessions(
    db_path: Path, *, since_hours: int, salt: str, top_n: int
) -> list[HeavySession]:
    cutoff = dt.datetime.now(dt.timezone.utc).timestamp() - since_hours * 3600
    conn = _connect_readonly(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "id" not in cols:
            return []
        wanted = [
            "id",
            "source",
            "input_tokens",
            "output_tokens",
            "message_count",
            "tool_call_count",
            "api_call_count",
        ]
        select_cols = [c if c in cols else f"NULL AS {c}" for c in wanted]
        where = []
        params: list[object] = []
        if "started_at" in cols:
            where.append("started_at >= ?")
            params.append(cutoff)
        if "archived" in cols:
            where.append("archived = 0")
        sql = f"SELECT {', '.join(select_cols)} FROM sessions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY input_tokens DESC"
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    sessions: list[HeavySession] = []
    for sid, source, inp, out_tok, msgs, tools, apis in rows:
        sessions.append(
            HeavySession(
                session_hash=_salted_hash(str(sid), salt),
                source=source,
                input_tokens=int(inp or 0),
                output_tokens=int(out_tok or 0),
                message_count=int(msgs or 0),
                tool_call_count=int(tools or 0),
                api_call_count=int(apis or 0),
            )
        )
    sessions.sort(key=lambda s: (s.input_tokens, s.tool_call_count), reverse=True)
    return sessions[:top_n]


def total_input_tokens(db_path: Path, *, since_hours: int) -> int:
    """Sum input tokens across ALL in-window sessions (not just the top-N)."""
    cutoff = dt.datetime.now(dt.timezone.utc).timestamp() - since_hours * 3600
    conn = _connect_readonly(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "input_tokens" not in cols:
            return 0
        where = []
        params: list[object] = []
        if "started_at" in cols:
            where.append("started_at >= ?")
            params.append(cutoff)
        if "archived" in cols:
            where.append("archived = 0")
        sql = "SELECT COALESCE(SUM(input_tokens), 0) FROM sessions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        (total,) = conn.execute(sql, params).fetchone()
    finally:
        conn.close()
    return int(total or 0)


def parse_telemetry(
    telemetry_path: Path, *, since_hours: int, total_input_tokens: int
) -> TelemetryCoverage:
    """Aggregate the metadata-only ContextPilot telemetry file.

    Tolerates malformed lines (non-JSON, non-dict, missing counters) by
    skipping and counting them. Never reads message content.
    """
    events = 0
    chars = 0
    tokens = 0
    malformed = 0
    if telemetry_path and telemetry_path.exists():
        cutoff = dt.datetime.now(dt.timezone.utc).timestamp() - since_hours * 3600
        with telemetry_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    malformed += 1
                    continue
                if not isinstance(record, dict):
                    malformed += 1
                    continue
                ts = record.get("ts")
                if isinstance(ts, (int, float)) and ts < cutoff:
                    continue
                cs = record.get("chars_saved")
                if not isinstance(cs, (int, float)):
                    malformed += 1
                    continue
                saved_tokens = record.get("tokens_saved")
                events += 1
                chars += int(cs)
                tokens += (
                    int(saved_tokens)
                    if isinstance(saved_tokens, (int, float))
                    else int(cs) // EST_CHARS_PER_TOKEN
                )

    denom = tokens + total_input_tokens
    coverage = (tokens / denom * 100.0) if denom else 0.0
    avg = (tokens / events) if events else 0.0
    return TelemetryCoverage(
        events=events,
        chars_saved=chars,
        tokens_saved=tokens,
        avg_tokens_saved_per_event=round(avg, 2),
        coverage_ratio_pct=round(coverage, 2),
        malformed_records_skipped=malformed,
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_exact_duplicate_tool_outputs(
    messages: Iterable[_ToolMessage], *, salt: str, top_n: int
) -> list[DuplicateToolOutput]:
    groups: dict[str, dict] = {}
    for msg in messages:
        content = msg.content
        if not content:
            continue
        h = _salted_hash(content, salt)
        g = groups.get(h)
        if g is None:
            groups[h] = {
                "tool_name": msg.tool_name,
                "occurrences": 1,
                "char_length": len(content),
            }
        else:
            g["occurrences"] += 1
            if g["tool_name"] != msg.tool_name:
                g["tool_name"] = None  # mixed tools produced identical output

    dups: list[DuplicateToolOutput] = []
    for h, g in groups.items():
        if g["occurrences"] < 2:
            continue
        est = _est_tokens(g["char_length"])
        dups.append(
            DuplicateToolOutput(
                content_hash=h,
                tool_name=g["tool_name"],
                occurrences=g["occurrences"],
                char_length=g["char_length"],
                est_tokens=est,
                est_wasted_tokens=est * (g["occurrences"] - 1),
            )
        )
    dups.sort(key=lambda d: d.est_wasted_tokens, reverse=True)
    return dups[:top_n]


def detect_repeated_blocks(
    messages: Iterable[_ToolMessage],
    *,
    salt: str,
    min_block_chars: int,
    min_repeat: int,
    top_n: int,
) -> list[RepeatedBlock]:
    counts: dict[str, dict] = {}
    for msg in messages:
        seen_in_msg: set[str] = set()
        for line in msg.content.splitlines():
            block = line.strip()
            if len(block) < min_block_chars:
                continue
            h = _salted_hash(block, salt)
            # Count cross-message recurrence; collapse repeats within one
            # message so a single noisy output cannot dominate.
            if h in seen_in_msg:
                continue
            seen_in_msg.add(h)
            c = counts.get(h)
            if c is None:
                counts[h] = {"occurrences": 1, "char_length": len(block)}
            else:
                c["occurrences"] += 1

    blocks: list[RepeatedBlock] = []
    for h, c in counts.items():
        if c["occurrences"] < min_repeat:
            continue
        est = _est_tokens(c["char_length"])
        blocks.append(
            RepeatedBlock(
                block_hash=h,
                occurrences=c["occurrences"],
                char_length=c["char_length"],
                est_tokens=est,
                est_wasted_tokens=est * (c["occurrences"] - 1),
            )
        )
    blocks.sort(key=lambda b: b.est_wasted_tokens, reverse=True)
    return blocks[:top_n]


def summarize_tool_sizes(
    messages: Iterable[_ToolMessage], *, large_output_chars: int, top_n: int
) -> list[ToolSizeStat]:
    agg: dict[str, dict] = {}
    for msg in messages:
        name = msg.tool_name or "(unknown)"
        length = len(msg.content)
        a = agg.get(name)
        if a is None:
            agg[name] = {
                "output_count": 1,
                "total_chars": length,
                "max_chars": length,
                "large_output_count": 1 if length >= large_output_chars else 0,
            }
        else:
            a["output_count"] += 1
            a["total_chars"] += length
            a["max_chars"] = max(a["max_chars"], length)
            if length >= large_output_chars:
                a["large_output_count"] += 1

    stats: list[ToolSizeStat] = []
    for name, a in agg.items():
        stats.append(
            ToolSizeStat(
                tool_name=name,
                output_count=a["output_count"],
                total_chars=a["total_chars"],
                max_chars=a["max_chars"],
                avg_chars=a["total_chars"] // a["output_count"],
                total_est_tokens=_est_tokens(a["total_chars"]),
                large_output_count=a["large_output_count"],
            )
        )
    stats.sort(key=lambda s: s.total_chars, reverse=True)
    return stats[:top_n]


# ---------------------------------------------------------------------------
# Build + write
# ---------------------------------------------------------------------------


def build_report(
    *,
    date: str,
    since_hours: int,
    salt: str,
    tool_messages: list[_ToolMessage],
    heavy_sessions: list[HeavySession],
    telemetry: TelemetryCoverage,
    min_block_chars: int = DEFAULT_MIN_BLOCK_CHARS,
    min_block_repeat: int = DEFAULT_MIN_BLOCK_REPEAT,
    large_output_chars: int = DEFAULT_LARGE_OUTPUT_CHARS,
    top_n: int = DEFAULT_TOP_N,
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

    total_chars = sum(len(m.content) for m in tool_messages)
    dup_wasted = sum(d.est_wasted_tokens for d in dups)
    block_wasted = sum(b.est_wasted_tokens for b in blocks)

    notes = [
        "content-aware analysis: message/tool text was hashed in-memory only and never written to reports",
        "all identifiers are salted SHA-256 fingerprints; counters are aggregates",
        "wasted-token figures are heuristic estimates (chars/4); validate before acting",
        "session 'source' and 'tool_name' are emitted verbatim as low-cardinality enums, not raw text",
    ]
    if not tool_messages:
        notes.append("no tool-output messages observed in the selected window")

    return OpportunityReport(
        date=date,
        since_hours=since_hours,
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
        notes=notes,
    )


def _assert_no_forbidden_keys(data: dict) -> None:
    """Defensive guard: ensure no forbidden raw-content key reached the output."""

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in FORBIDDEN_OUTPUT_KEYS:
                    raise RuntimeError(f"refusing to emit forbidden key: {k}")
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)


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
    md = [
        f"# ContextPilot Hermes opportunity scan — {report.date}",
        "",
        f"Window: last {report.since_hours}h",
        f"Salt fingerprint: `{report.salt_fingerprint}`",
        "",
        "## Summary",
        f"- Tool-output messages: {report.tool_message_count}",
        f"- Total tool-output tokens (est): {report.total_tool_output_est_tokens}",
        f"- Exact duplicate groups: {report.duplicate_tool_output_groups} "
        f"(~{report.duplicate_tool_output_wasted_tokens} wasted tokens)",
        f"- Repeated blocks: {report.repeated_block_count} "
        f"(~{report.repeated_block_wasted_tokens} wasted tokens)",
        f"- Telemetry: {t.events} events, ~{t.tokens_saved} tokens saved, "
        f"coverage {t.coverage_ratio_pct}%",
        "",
        "## Top exact-duplicate tool outputs",
    ]
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
    md.append("## Telemetry coverage")
    md.extend(
        [
            f"- Events: {t.events}",
            f"- Tokens saved: {t.tokens_saved} (chars {t.chars_saved})",
            f"- Avg tokens saved / event: {t.avg_tokens_saved_per_event}",
            f"- Coverage ratio: {t.coverage_ratio_pct}%",
            f"- Malformed records skipped: {t.malformed_records_skipped}",
        ]
    )
    md.append("")
    md.append("## Notes")
    for note in report.notes:
        md.append(f"- {note}")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", type=Path, default=Path("/root/.hermes/state.db"))
    parser.add_argument(
        "--telemetry-file",
        type=Path,
        default=Path.home() / ".hermes" / "contextpilot" / "telemetry.jsonl",
        help="metadata-only ContextPilot telemetry file",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path.home() / "contextpilot" / "opportunities"
    )
    parser.add_argument("--since-hours", type=int, default=24)
    parser.add_argument(
        "--salt",
        default="contextpilot-hermes-opportunity-v1",
        help="salt for stable per-install content/session fingerprints",
    )
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--min-block-chars", type=int, default=DEFAULT_MIN_BLOCK_CHARS)
    parser.add_argument("--min-block-repeat", type=int, default=DEFAULT_MIN_BLOCK_REPEAT)
    parser.add_argument(
        "--large-output-chars", type=int, default=DEFAULT_LARGE_OUTPUT_CHARS
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()

    if not args.state_db.exists():
        raise SystemExit(f"Hermes state DB not found: {args.state_db}")

    # Harden for unattended cron use: never dump a traceback (which would echo
    # the DB path / SQL); emit only the exception class name and a non-zero code.
    try:
        tool_messages = load_tool_messages(args.state_db, since_hours=args.since_hours)
        heavy_sessions = load_heavy_sessions(
            args.state_db, since_hours=args.since_hours, salt=args.salt, top_n=args.top_n
        )
        total_input = total_input_tokens(args.state_db, since_hours=args.since_hours)
        telemetry = parse_telemetry(
            args.telemetry_file,
            since_hours=args.since_hours,
            total_input_tokens=total_input,
        )
        report = build_report(
            date=args.date,
            since_hours=args.since_hours,
            salt=args.salt,
            tool_messages=tool_messages,
            heavy_sessions=heavy_sessions,
            telemetry=telemetry,
            min_block_chars=args.min_block_chars,
            min_block_repeat=args.min_block_repeat,
            large_output_chars=args.large_output_chars,
            top_n=args.top_n,
        )
        json_path, md_path = write_report(report, args.out_dir)
    except Exception as exc:  # noqa: BLE001 - cron-safe: report class only, no payload
        print(json.dumps({"ok": False, "error": type(exc).__name__}))
        return 1

    print(
        json.dumps(
            {"ok": True, "json": str(json_path), "markdown": str(md_path)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
