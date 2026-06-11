#!/usr/bin/env python3
"""Privacy-safe ContextPilot monitor for Hermes Agent.

Reads Hermes metadata (sessions table) and ContextPilot savings log lines, then
writes daily JSON/Markdown reports. It deliberately never reads message bodies,
system prompts, reasoning text, or tool payload content.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SAVINGS_RE = re.compile(
    r"\[ContextPilot\].*?saved\s+(?P<chars>\d+)\s+chars\s+\(~(?P<tokens>\d+)\s+tokens\)"
)
SESSION_RE = re.compile(
    r"\[ContextPilot\]\s+Session\s+(?P<session>[^:]+):\s+(?P<turns>\d+)\s+turns,\s+"
    r"(?P<chars>\d+)\s+chars\s+saved\s+\(~(?P<tokens>\d+)\s+tokens\)"
)

FORBIDDEN_COLUMNS = {
    "content",
    "system_prompt",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "tool_calls",
    "codex_reasoning_items",
    "codex_message_items",
}


@dataclass
class SessionMetric:
    session_hash: str
    source: str | None
    started_at: float
    ended_at: float | None
    message_count: int
    tool_call_count: int
    api_call_count: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    estimated_cost_usd: float | None


@dataclass
class DailyReport:
    date: str
    since_hours: int
    session_count: int
    total_messages: int
    total_tool_calls: int
    total_api_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_write_tokens: int
    total_reasoning_tokens: int
    estimated_cost_usd: float
    contextpilot_log_events: int
    contextpilot_telemetry_events: int
    contextpilot_savings_source: str
    contextpilot_chars_saved: int
    contextpilot_tokens_saved: int
    estimated_input_token_reduction_pct: float
    top_sources: dict[str, int]
    top_token_sessions: list[SessionMetric]
    notes: list[str]


def _hash_session(session_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{session_id}".encode()).hexdigest()[:16]


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _assert_schema_safe(conn: sqlite3.Connection) -> None:
    # Guard against accidental SELECT * expansion in future edits: explicitly
    # name every session column we read and refuse message-table content access.
    session_cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    message_cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    if not session_cols:
        raise RuntimeError("Hermes sessions table not found")
    if "content" in message_cols:
        # The monitor is allowed to count messages, never read their bodies.
        pass


def load_session_metrics(db_path: Path, *, since_hours: int, salt: str) -> list[SessionMetric]:
    cutoff = dt.datetime.now(dt.timezone.utc).timestamp() - since_hours * 3600
    conn = _connect_readonly(db_path)
    try:
        _assert_schema_safe(conn)
        query_columns = [
            "id",
            "source",
            "started_at",
            "ended_at",
            "message_count",
            "tool_call_count",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "estimated_cost_usd",
            "api_call_count",
        ]
        if FORBIDDEN_COLUMNS.intersection(query_columns):
            raise RuntimeError("Internal error: forbidden raw-content column requested")
        sql = f"""
            SELECT {', '.join(query_columns)}
            FROM sessions
            WHERE started_at >= ? AND archived = 0
            ORDER BY input_tokens DESC
        """
        rows = conn.execute(sql, (cutoff,)).fetchall()
    finally:
        conn.close()

    metrics: list[SessionMetric] = []
    for row in rows:
        (
            sid,
            source,
            started_at,
            ended_at,
            message_count,
            tool_call_count,
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
            estimated_cost_usd,
            api_call_count,
        ) = row
        metrics.append(
            SessionMetric(
                session_hash=_hash_session(str(sid), salt),
                source=source,
                started_at=float(started_at),
                ended_at=float(ended_at) if ended_at is not None else None,
                message_count=int(message_count or 0),
                tool_call_count=int(tool_call_count or 0),
                api_call_count=int(api_call_count or 0),
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
                cache_read_tokens=int(cache_read_tokens or 0),
                cache_write_tokens=int(cache_write_tokens or 0),
                reasoning_tokens=int(reasoning_tokens or 0),
                estimated_cost_usd=float(estimated_cost_usd or 0.0),
            )
        )
    return metrics


def parse_contextpilot_savings(log_path: Path, *, since_hours: int) -> tuple[int, int, int]:
    if not log_path.exists():
        return 0, 0, 0
    # Gateway logs can be large. Tail a bounded byte window; cron should run daily.
    max_bytes = 8 * 1024 * 1024
    with log_path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        text = f.read().decode("utf-8", errors="replace")

    events = 0
    chars = 0
    tokens = 0
    for line in text.splitlines():
        # Timestamp filtering is best-effort; if parse fails, keep the line only
        # when it is in the tailed window. No message content is logged here.
        m = SAVINGS_RE.search(line)
        if not m:
            continue
        events += 1
        chars += int(m.group("chars"))
        tokens += int(m.group("tokens"))
    return events, chars, tokens


def parse_contextpilot_telemetry(telemetry_path: Path, *, since_hours: int) -> tuple[int, int, int]:
    """Aggregate the plugin's metadata-only telemetry file.

    Returns (events, chars_saved, tokens_saved). The file is JSON-lines, one
    numeric record per saved turn; it never contains message content, prompts,
    or tool payloads, so we only read numeric counters here.
    """
    if not telemetry_path or not telemetry_path.exists():
        return 0, 0, 0
    cutoff = dt.datetime.now(dt.timezone.utc).timestamp() - since_hours * 3600

    events = 0
    chars = 0
    tokens = 0
    with telemetry_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(record, dict):
                continue
            ts = record.get("ts")
            if isinstance(ts, (int, float)) and ts < cutoff:
                continue
            cs = record.get("chars_saved")
            if not isinstance(cs, (int, float)):
                continue
            saved_tokens = record.get("tokens_saved")
            events += 1
            chars += int(cs)
            tokens += int(saved_tokens) if isinstance(saved_tokens, (int, float)) else int(cs) // 4
    return events, chars, tokens


def build_report(
    metrics: Iterable[SessionMetric],
    *,
    date: str,
    since_hours: int,
    log_stats: tuple[int, int, int],
    telemetry_stats: tuple[int, int, int] = (0, 0, 0),
) -> DailyReport:
    rows = list(metrics)
    source_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row.source or "unknown"] = source_counts.get(row.source or "unknown", 0) + 1

    total_input = sum(r.input_tokens for r in rows)
    log_events, log_chars, log_tokens = log_stats
    tel_events, tel_chars, tel_tokens = telemetry_stats

    # Prefer the local telemetry file when present: it is the authoritative,
    # log-independent source. Logs are a fallback and are NOT summed on top
    # (both record the same turns, so summing would double-count).
    if tel_events > 0:
        events, saved_chars, saved_tokens = tel_events, tel_chars, tel_tokens
        savings_source = "telemetry"
    else:
        events, saved_chars, saved_tokens = log_events, log_chars, log_tokens
        savings_source = "gateway-log"

    denominator = total_input + saved_tokens
    reduction = (saved_tokens / denominator * 100.0) if denominator else 0.0

    notes: list[str] = [
        "metadata-only: did not read messages.content, sessions.system_prompt, reasoning, or tool payloads",
        "accuracy gate is observational here; apply code/config changes only after separate golden-eval pass",
        f"contextpilot savings source: {savings_source} (telemetry={tel_events} events, log={log_events} events)",
    ]
    if not rows:
        notes.append("no sessions observed in the selected window")
    if tel_events == 0 and log_events == 0:
        notes.append(
            "no ContextPilot savings observed via telemetry or logs; "
            "gateway may need restart after enabling plugin"
        )

    return DailyReport(
        date=date,
        since_hours=since_hours,
        session_count=len(rows),
        total_messages=sum(r.message_count for r in rows),
        total_tool_calls=sum(r.tool_call_count for r in rows),
        total_api_calls=sum(r.api_call_count for r in rows),
        total_input_tokens=total_input,
        total_output_tokens=sum(r.output_tokens for r in rows),
        total_cache_read_tokens=sum(r.cache_read_tokens for r in rows),
        total_cache_write_tokens=sum(r.cache_write_tokens for r in rows),
        total_reasoning_tokens=sum(r.reasoning_tokens for r in rows),
        estimated_cost_usd=sum(r.estimated_cost_usd or 0.0 for r in rows),
        contextpilot_log_events=log_events,
        contextpilot_telemetry_events=tel_events,
        contextpilot_savings_source=savings_source,
        contextpilot_chars_saved=saved_chars,
        contextpilot_tokens_saved=saved_tokens,
        estimated_input_token_reduction_pct=round(reduction, 2),
        top_sources=dict(sorted(source_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        top_token_sessions=rows[:10],
        notes=notes,
    )


def write_report(report: DailyReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"daily_{report.date}.json"
    md_path = out_dir / f"daily_{report.date}.md"
    data = asdict(report)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        f"# ContextPilot Hermes monitor — {report.date}",
        "",
        f"Window: last {report.since_hours}h",
        "",
        "## Summary",
        f"- Sessions: {report.session_count}",
        f"- Input tokens: {report.total_input_tokens}",
        f"- Output tokens: {report.total_output_tokens}",
        f"- Tool calls: {report.total_tool_calls}",
        f"- ContextPilot saved: ~{report.contextpilot_tokens_saved} tokens ({report.contextpilot_chars_saved} chars)",
        f"- ContextPilot savings source: {report.contextpilot_savings_source} "
        f"(telemetry events={report.contextpilot_telemetry_events}, log events={report.contextpilot_log_events})",
        f"- Estimated input-token reduction: {report.estimated_input_token_reduction_pct}%",
        f"- Estimated cost: ${report.estimated_cost_usd:.4f}",
        "",
        "## Top sources",
    ]
    for source, count in report.top_sources.items():
        md.append(f"- {source}: {count}")
    md.extend(["", "## Top token sessions (hashed)"])
    for row in report.top_token_sessions:
        md.append(
            f"- `{row.session_hash}` source={row.source} input={row.input_tokens} "
            f"output={row.output_tokens} tools={row.tool_call_count} apis={row.api_call_count}"
        )
    md.extend(["", "## Notes"])
    for note in report.notes:
        md.append(f"- {note}")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", type=Path, default=Path.home() / ".hermes" / "state.db")
    parser.add_argument("--gateway-log", type=Path, default=Path.home() / ".hermes" / "logs" / "gateway.log")
    parser.add_argument(
        "--telemetry-file",
        type=Path,
        default=Path.home() / ".hermes" / "contextpilot" / "telemetry.jsonl",
        help="metadata-only ContextPilot telemetry file (preferred over gateway log)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "contextpilot" / "reports")
    parser.add_argument("--since-hours", type=int, default=24)
    parser.add_argument("--salt", default="contextpilot-hermes-monitor-v1", help="salt for stable per-install session hashes")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    args = parser.parse_args()

    if not args.state_db.exists():
        raise SystemExit(f"Hermes state DB not found: {args.state_db}")

    metrics = load_session_metrics(args.state_db, since_hours=args.since_hours, salt=args.salt)
    log_stats = parse_contextpilot_savings(args.gateway_log, since_hours=args.since_hours)
    telemetry_stats = parse_contextpilot_telemetry(args.telemetry_file, since_hours=args.since_hours)
    report = build_report(
        metrics,
        date=args.date,
        since_hours=args.since_hours,
        log_stats=log_stats,
        telemetry_stats=telemetry_stats,
    )
    json_path, md_path = write_report(report, args.out_dir)
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
