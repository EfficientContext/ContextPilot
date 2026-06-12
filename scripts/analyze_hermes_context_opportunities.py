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


# Recognized LLM-bound block types. These are low-cardinality enums, safe to
# emit verbatim (they describe the *origin* of a block, never its text).
BLOCK_TYPES = (
    "system_prompt",
    "skill_prompt",
    "user_prompt",
    "assistant_context",
    "tool_result",
    "unknown",
)


@dataclass
class TypeCount:
    block_type: str
    count: int


@dataclass
class BlockTypeStat:
    """Aggregate redundancy within a single LLM-bound block type."""

    block_type: str
    item_count: int            # source items (prompts/messages) of this type
    block_count: int           # total fingerprintable block instances
    unique_block_count: int    # distinct fingerprints
    repeated_block_count: int  # fingerprints recurring >= min_repeat within type
    est_redundant_tokens: int  # sum over repeats of (occ-1) * est_tokens


@dataclass
class CrossTypeBlockGroup:
    """A single block fingerprint observed in 2+ distinct block types.

    This is the headline signal: the same chunk of text is being shipped to the
    LLM from, e.g., a skill/system prompt *and* a tool result, so it is paying
    for the same tokens twice from different sources.
    """

    block_hash: str
    block_types: list[str]               # sorted distinct types this block spans
    type_occurrences: list[TypeCount]    # per-type occurrence counts
    occurrences: int                     # total occurrences across all types
    char_length: int
    est_tokens: int
    est_wasted_tokens: int               # (occurrences - 1) * est_tokens


# ---------------------------------------------------------------------------
# Worker Context Routing — SHADOW MODE (P0 data collection only)
# ---------------------------------------------------------------------------
# Low-cardinality router labels. These are the *training/eval* labels a future
# small worker-context router would predict. P0 is data-collection only: nothing
# here ever drops, summarizes, or mutates context — it only classifies blocks
# and emits aggregate counters + salted hashes so the labels can be evaluated
# offline before any online pruning is built.
ROUTER_LABELS = (
    "policy_must_keep",        # never droppable (user/system/skill/safety constraints)
    "direct_task_hint",        # short actionable task signal — keep
    "likely_relevant",         # default keep; not obviously prunable
    "summarizable_candidate",  # large single block that *might* be summarized later
    "likely_drop_candidate",   # large/repeated tool-like block, candidate to route away
)

# Labels whose blocks a future router might safely route away. Used only to
# tally *advisory* candidate tokens; P0 never acts on them.
_ROUTABLE_LABELS = ("summarizable_candidate", "likely_drop_candidate")

# Block-type priority when one fingerprint spans multiple origins: the most
# "must-keep" origin wins, so cross-origin blocks are classified conservatively.
_TYPE_KEEP_PRIORITY = {
    "user_prompt": 5,
    "system_prompt": 4,
    "skill_prompt": 4,
    "assistant_context": 2,
    "tool_result": 1,
    "unknown": 0,
}

# Cues marking content that must NEVER be dropped even from a tool/assistant
# block: explicit safety / acceptance / hard-constraint language. Matching here
# is intentionally generous — over-keeping is the safe direction for P0.
_SAFETY_CONSTRAINT_CUES = (
    "must not",
    "must never",
    "never drop",
    "do not delete",
    "do not remove",
    "do not modify",
    "acceptance criteria",
    "acceptance test",
    "safety",
    "must keep",
    "you must",
    "required:",
    "constraint",
    "forbidden",
    "policy",
)

# Cues marking a short, actionable task hint worth keeping verbatim.
_TASK_HINT_CUES = (
    "todo",
    "next step",
    "error:",
    "traceback",
    "failed",
    "fixme",
    "task:",
    "goal:",
    "implement",
    "reproduce",
)


@dataclass
class RouterLabelCount:
    """Aggregate over all blocks assigned one router label."""

    route_label: str
    block_count: int            # distinct fingerprints with this label
    occurrence_count: int       # total occurrences across the window
    total_est_tokens: int       # est tokens these blocks occupy (occ * est)
    est_candidate_tokens: int   # ADVISORY routable tokens (0 unless routable)


@dataclass
class RouterReasonCount:
    """Aggregate keyed by (block_type, route_label, reason_code)."""

    block_type: str
    route_label: str
    reason_code: str
    block_count: int
    occurrence_count: int
    total_est_tokens: int
    est_candidate_tokens: int


@dataclass
class RouterCandidateBlock:
    """A single routable-candidate fingerprint (salted hash + counters only)."""

    block_hash: str
    block_type: str
    route_label: str
    reason_code: str
    occurrences: int
    char_length: int
    est_tokens: int
    est_candidate_tokens: int   # ADVISORY upper bound only


@dataclass
class WorkerRoutingShadow:
    """Shadow-mode worker-context routing report (P0: data collection only)."""

    enabled: bool
    item_count: int                 # LLM-bound items classified
    classified_block_count: int     # distinct fingerprints classified
    total_occurrences: int
    must_keep_block_count: int
    must_keep_occurrence_count: int
    est_must_keep_tokens: int
    est_candidate_tokens_total: int          # ADVISORY routable ceiling
    est_drop_candidate_tokens: int           # ADVISORY
    est_summarizable_candidate_tokens: int   # ADVISORY
    label_counts: list[RouterLabelCount]
    reason_counts: list[RouterReasonCount]
    top_candidate_blocks: list[RouterCandidateBlock]
    notes: list[str] = field(default_factory=list)


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
    all_sessions: bool
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
    # LLM-bound block analysis (system/skill prompts, prompts, tool results).
    llm_bound_item_count: int
    llm_block_types: list[BlockTypeStat]
    cross_type_block_groups: list[CrossTypeBlockGroup]
    cross_type_wasted_tokens: int
    # Worker Context Routing shadow mode (P0 data collection; never prunes).
    worker_routing: WorkerRoutingShadow
    # Parent Aggregation Artifacts shadow mode (P0 telemetry; never dedups).
    parent_aggregation: ParentAggregationArtifacts
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class _ToolMessage:
    tool_name: str | None
    content: str


@dataclass
class _LLMContent:
    """A chunk of content that Hermes would actually send to the LLM.

    Held in-memory only for hashing; ``content`` must never be emitted.
    """

    block_type: str
    content: str


def _window_cutoff(since_hours: int, all_sessions: bool) -> float | None:
    """Return the epoch cutoff, or ``None`` to scan all history.

    ``all_sessions=True`` disables the time window so old sessions/messages are
    included regardless of ``since_hours``.
    """
    if all_sessions:
        return None
    return dt.datetime.now(dt.timezone.utc).timestamp() - since_hours * 3600


def _classify_system_prompt(text: str) -> str:
    """Heuristically label a system prompt as skill material or a plain prompt.

    Operates on in-memory text only; returns a low-cardinality enum, never the
    text itself.
    """
    low = text.lower()
    stripped = low.lstrip()
    # Skill-style frontmatter block (e.g. "---\nname: ...\ndescription: ...").
    if stripped.startswith("---") and "name:" in low[:300]:
        return "skill_prompt"
    cues = (
        "use this skill",
        "available skills",
        "when to use",
        "invoke it via skill",
        "<skill",
        "# skill",
        "skill tool",
    )
    if any(c in low for c in cues):
        return "skill_prompt"
    return "system_prompt"


def _message_block_type(role: str | None, tool_name: str | None) -> str:
    if role == "tool" or tool_name is not None:
        return "tool_result"
    if role == "user":
        return "user_prompt"
    if role == "assistant":
        return "assistant_context"
    if role == "system":
        return "system_prompt"
    return "unknown"


def load_tool_messages(
    db_path: Path, *, since_hours: int, all_sessions: bool = False
) -> list[_ToolMessage]:
    """Load tool-output messages within the window.

    Content is returned for in-memory hashing only; callers must not emit it.
    A message is treated as tool output when ``role='tool'`` or ``tool_name``
    is set. With ``all_sessions=True`` the time window is ignored.
    """
    cutoff = _window_cutoff(since_hours, all_sessions)
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
        if has_ts and cutoff is not None:
            where.append("timestamp >= ?")
            params.append(cutoff)
        if "active" in cols:
            where.append("active = 1")
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


def load_llm_bound_content(
    db_path: Path, *, since_hours: int, all_sessions: bool = False
) -> list[_LLMContent]:
    """Load only content Hermes would actually send to an LLM.

    Sources, all read in-memory for hashing (never emitted):
      * ``sessions.system_prompt`` -> ``system_prompt`` or ``skill_prompt``,
      * ``messages.content`` for active messages with role in
        ``system``/``user``/``assistant``/``tool`` -> per-role block type,
      * tool-result messages (role=tool or ``tool_name`` set) -> ``tool_result``.

    Inactive messages are skipped when an ``active`` column exists; archived
    sessions (and their messages) are skipped when an ``archived`` column
    exists. With ``all_sessions=True`` the time window is ignored.
    """
    cutoff = _window_cutoff(since_hours, all_sessions)
    conn = _connect_readonly(db_path)
    out: list[_LLMContent] = []
    try:
        scols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        mcols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}

        # --- system / skill prompts from sessions -------------------------
        if "system_prompt" in scols:
            where = ["system_prompt IS NOT NULL"]
            params: list[object] = []
            if cutoff is not None and "started_at" in scols:
                where.append("started_at >= ?")
                params.append(cutoff)
            if "archived" in scols:
                where.append("archived = 0")
            sql = f"SELECT system_prompt FROM sessions WHERE {' AND '.join(where)}"
            for (sp,) in conn.execute(sql, params):
                if sp is None:
                    continue
                text = str(sp)
                out.append(
                    _LLMContent(block_type=_classify_system_prompt(text), content=text)
                )

        # --- active messages bound for the LLM ----------------------------
        if "content" in mcols:
            has_role = "role" in mcols
            has_tool_name = "tool_name" in mcols
            select = [
                "messages.role" if has_role else "NULL AS role",
                "messages.content",
                "messages.tool_name" if has_tool_name else "NULL AS tool_name",
            ]
            where = ["messages.content IS NOT NULL"]
            params = []
            if has_role:
                where.append(
                    "messages.role IN ('system', 'user', 'assistant', 'tool')"
                )
            if cutoff is not None and "timestamp" in mcols:
                where.append("messages.timestamp >= ?")
                params.append(cutoff)
            if "active" in mcols:
                where.append("messages.active = 1")
            join = ""
            if "archived" in scols and "session_id" in mcols and "id" in scols:
                join = " JOIN sessions ON sessions.id = messages.session_id"
                where.append("sessions.archived = 0")
            sql = (
                f"SELECT {', '.join(select)} FROM messages{join} "
                f"WHERE {' AND '.join(where)}"
            )
            for role, content, tool_name in conn.execute(sql, params):
                if content is None:
                    continue
                out.append(
                    _LLMContent(
                        block_type=_message_block_type(role, tool_name),
                        content=str(content),
                    )
                )
    finally:
        conn.close()
    return out


def load_heavy_sessions(
    db_path: Path, *, since_hours: int, salt: str, top_n: int, all_sessions: bool = False
) -> list[HeavySession]:
    cutoff = _window_cutoff(since_hours, all_sessions)
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
        if cutoff is not None and "started_at" in cols:
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


def total_input_tokens(
    db_path: Path, *, since_hours: int, all_sessions: bool = False
) -> int:
    """Sum input tokens across ALL in-window sessions (not just the top-N)."""
    cutoff = _window_cutoff(since_hours, all_sessions)
    conn = _connect_readonly(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "input_tokens" not in cols:
            return 0
        where = []
        params: list[object] = []
        if cutoff is not None and "started_at" in cols:
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
    telemetry_path: Path,
    *,
    since_hours: int,
    total_input_tokens: int,
    all_sessions: bool = False,
) -> TelemetryCoverage:
    """Aggregate the metadata-only ContextPilot telemetry file.

    Tolerates malformed lines (non-JSON, non-dict, missing counters) by
    skipping and counting them. Never reads message content. With
    ``all_sessions=True`` the time window is ignored.
    """
    events = 0
    chars = 0
    tokens = 0
    malformed = 0
    if telemetry_path and telemetry_path.exists():
        cutoff = _window_cutoff(since_hours, all_sessions)
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
                if cutoff is not None and isinstance(ts, (int, float)) and ts < cutoff:
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


def _iter_blocks(content: str, min_block_chars: int) -> Iterable[str]:
    """Yield the distinct fingerprintable lines of one item (deduped in-item)."""
    seen: set[str] = set()
    for line in content.splitlines():
        block = line.strip()
        if len(block) < min_block_chars:
            continue
        if block in seen:
            continue
        seen.add(block)
        yield block


def analyze_llm_bound_blocks(
    contents: Iterable[_LLMContent],
    *,
    salt: str,
    min_block_chars: int,
    min_repeat: int,
    top_n: int,
) -> tuple[list[BlockTypeStat], list[CrossTypeBlockGroup]]:
    """Fingerprint LLM-bound blocks and report redundancy.

    Returns (per-type stats, cross-type repeated block groups). All output is
    salted hashes / counters / block-type enums -- no raw text.
    """
    # block_hash -> {char_length, types: {block_type: occ}}
    agg: dict[str, dict] = {}
    # block_type -> source item count
    item_counts: dict[str, int] = {}

    for item in contents:
        bt = item.block_type
        item_counts[bt] = item_counts.get(bt, 0) + 1
        for block in _iter_blocks(item.content, min_block_chars):
            h = _salted_hash(block, salt)
            entry = agg.get(h)
            if entry is None:
                agg[h] = {"char_length": len(block), "types": {bt: 1}}
            else:
                entry["types"][bt] = entry["types"].get(bt, 0) + 1

    # --- per block-type aggregate redundancy ------------------------------
    per_type: dict[str, dict] = {}
    for entry in agg.values():
        est = _est_tokens(entry["char_length"])
        for bt, occ in entry["types"].items():
            t = per_type.setdefault(
                bt,
                {
                    "block_count": 0,
                    "unique": 0,
                    "repeated": 0,
                    "redundant_tokens": 0,
                },
            )
            t["block_count"] += occ
            t["unique"] += 1
            if occ >= min_repeat:
                t["repeated"] += 1
                t["redundant_tokens"] += est * (occ - 1)

    block_type_stats: list[BlockTypeStat] = []
    for bt in sorted(set(per_type) | set(item_counts)):
        t = per_type.get(
            bt, {"block_count": 0, "unique": 0, "repeated": 0, "redundant_tokens": 0}
        )
        block_type_stats.append(
            BlockTypeStat(
                block_type=bt,
                item_count=item_counts.get(bt, 0),
                block_count=t["block_count"],
                unique_block_count=t["unique"],
                repeated_block_count=t["repeated"],
                est_redundant_tokens=t["redundant_tokens"],
            )
        )

    # --- cross-type repeated blocks ---------------------------------------
    cross: list[CrossTypeBlockGroup] = []
    for h, entry in agg.items():
        types = entry["types"]
        if len(types) < 2:
            continue
        total_occ = sum(types.values())
        est = _est_tokens(entry["char_length"])
        cross.append(
            CrossTypeBlockGroup(
                block_hash=h,
                block_types=sorted(types.keys()),
                type_occurrences=[
                    TypeCount(block_type=bt, count=occ)
                    for bt, occ in sorted(types.items())
                ],
                occurrences=total_occ,
                char_length=entry["char_length"],
                est_tokens=est,
                est_wasted_tokens=est * (total_occ - 1),
            )
        )
    cross.sort(key=lambda g: g.est_wasted_tokens, reverse=True)
    return block_type_stats, cross[:top_n]


def classify_router_label(
    block_type: str,
    content: str,
    *,
    occurrences: int,
    large_output_chars: int,
    min_repeat: int,
) -> tuple[str, str]:
    """Heuristically assign a worker-routing label + reason code to a block.

    Pure P0 heuristic: no ML, no network, no mutation. Operates on in-memory
    text only and returns two low-cardinality enums (``route_label``,
    ``reason_code``) -- never the text. The bias is deliberately conservative:
    when in doubt, keep. Anything that is a user prompt, a system/skill prompt,
    or carries explicit safety/acceptance-constraint language is pinned to
    ``policy_must_keep`` and can never become a routable candidate.
    """
    low = content.lower()

    # 1. Never-drop by origin: prompts the user/system/skills authored.
    if block_type == "user_prompt":
        return "policy_must_keep", "user_prompt_never_drop"
    if block_type in ("system_prompt", "skill_prompt"):
        return "policy_must_keep", "system_or_skill_constraint_never_drop"

    # 2. Never-drop by content: explicit safety / acceptance / hard constraints,
    #    even inside an assistant or tool block.
    if any(cue in low for cue in _SAFETY_CONSTRAINT_CUES):
        return "policy_must_keep", "safety_or_acceptance_constraint"

    char_len = len(content)
    has_task_hint = any(cue in low for cue in _TASK_HINT_CUES)

    # 3. Short actionable task hints -> keep verbatim. Very large diagnostic
    #    logs often contain "error:"/"failed"/"traceback"; keep collecting
    #    them as summarization candidates instead of pinning the whole log.
    if has_task_hint and char_len < large_output_chars:
        return "direct_task_hint", "actionable_task_signal"

    # 4. Bulky / repeated tool-like material -> routable candidates (advisory).
    if block_type in ("tool_result", "assistant_context", "unknown"):
        if has_task_hint and char_len >= large_output_chars:
            return "summarizable_candidate", "large_actionable_tool_block"
        is_large = char_len >= large_output_chars
        is_repeated = occurrences >= min_repeat
        if is_large and is_repeated:
            return "likely_drop_candidate", "large_repeated_tool_block"
        if is_repeated:
            return "likely_drop_candidate", "repeated_tool_block"
        if is_large:
            return "summarizable_candidate", "large_single_tool_block"

    # 5. Everything else: keep by default.
    return "likely_relevant", "default_keep"


def analyze_worker_routing_shadow(
    contents: Iterable[_LLMContent],
    *,
    salt: str,
    large_output_chars: int,
    min_repeat: int,
    top_n: int,
    enabled: bool = True,
) -> WorkerRoutingShadow:
    """Shadow-mode worker-context routing classifier (P0: data collection only).

    Fingerprints each LLM-bound item, assigns a conservative router label, and
    returns aggregate counters + salted hashes for routable candidates. Emits
    NO raw text and never mutates/drops context. ``est_candidate_tokens`` is an
    advisory upper bound on what a *future* router might route away -- not a
    realized saving.
    """
    if not enabled:
        return WorkerRoutingShadow(
            enabled=False,
            item_count=0,
            classified_block_count=0,
            total_occurrences=0,
            must_keep_block_count=0,
            must_keep_occurrence_count=0,
            est_must_keep_tokens=0,
            est_candidate_tokens_total=0,
            est_drop_candidate_tokens=0,
            est_summarizable_candidate_tokens=0,
            label_counts=[],
            reason_counts=[],
            top_candidate_blocks=[],
            notes=["worker-routing shadow analysis disabled via flag"],
        )

    # Aggregate occurrences per fingerprint, picking the most must-keep origin
    # when one block spans several block types.
    agg: dict[str, dict] = {}
    item_count = 0
    for item in contents:
        content = item.content
        if not content:
            continue
        item_count += 1
        h = _salted_hash(content, salt)
        bt = item.block_type
        entry = agg.get(h)
        if entry is None:
            agg[h] = {
                "block_type": bt,
                "char_length": len(content),
                "occurrences": 1,
                "content": content,
            }
        else:
            entry["occurrences"] += 1
            cur = entry["block_type"]
            bt_pri = _TYPE_KEEP_PRIORITY.get(bt, 0)
            cur_pri = _TYPE_KEEP_PRIORITY.get(cur, 0)
            if bt_pri > cur_pri or (bt_pri == cur_pri and bt < cur):
                entry["block_type"] = bt

    # Classify each unique fingerprint and roll up counters.
    label_agg: dict[str, dict] = {}
    reason_agg: dict[tuple[str, str, str], dict] = {}
    candidates: list[RouterCandidateBlock] = []
    must_keep_blocks = 0
    must_keep_occ = 0
    est_must_keep_tokens = 0
    drop_tokens = 0
    summ_tokens = 0

    for h, entry in agg.items():
        bt = entry["block_type"]
        occ = entry["occurrences"]
        char_len = entry["char_length"]
        est = _est_tokens(char_len)
        total_est = est * occ
        label, reason = classify_router_label(
            bt,
            entry["content"],
            occurrences=occ,
            large_output_chars=large_output_chars,
            min_repeat=min_repeat,
        )
        candidate_tokens = total_est if label in _ROUTABLE_LABELS else 0

        la = label_agg.setdefault(
            label,
            {"block_count": 0, "occ": 0, "total_est": 0, "candidate": 0},
        )
        la["block_count"] += 1
        la["occ"] += occ
        la["total_est"] += total_est
        la["candidate"] += candidate_tokens

        ra = reason_agg.setdefault(
            (bt, label, reason),
            {"block_count": 0, "occ": 0, "total_est": 0, "candidate": 0},
        )
        ra["block_count"] += 1
        ra["occ"] += occ
        ra["total_est"] += total_est
        ra["candidate"] += candidate_tokens

        if label == "policy_must_keep":
            must_keep_blocks += 1
            must_keep_occ += occ
            est_must_keep_tokens += total_est
        if label == "likely_drop_candidate":
            drop_tokens += candidate_tokens
        elif label == "summarizable_candidate":
            summ_tokens += candidate_tokens

        if candidate_tokens > 0:
            candidates.append(
                RouterCandidateBlock(
                    block_hash=h,
                    block_type=bt,
                    route_label=label,
                    reason_code=reason,
                    occurrences=occ,
                    char_length=char_len,
                    est_tokens=est,
                    est_candidate_tokens=candidate_tokens,
                )
            )

    # Deterministic ordering: label_counts follow the canonical label order;
    # reason_counts and candidates sort by a stable key.
    label_counts = [
        RouterLabelCount(
            route_label=lbl,
            block_count=label_agg[lbl]["block_count"],
            occurrence_count=label_agg[lbl]["occ"],
            total_est_tokens=label_agg[lbl]["total_est"],
            est_candidate_tokens=label_agg[lbl]["candidate"],
        )
        for lbl in ROUTER_LABELS
        if lbl in label_agg
    ]
    reason_counts = [
        RouterReasonCount(
            block_type=bt,
            route_label=lbl,
            reason_code=reason,
            block_count=v["block_count"],
            occurrence_count=v["occ"],
            total_est_tokens=v["total_est"],
            est_candidate_tokens=v["candidate"],
        )
        for (bt, lbl, reason), v in sorted(reason_agg.items())
    ]
    candidates.sort(
        key=lambda c: (c.est_candidate_tokens, c.occurrences, c.block_hash),
        reverse=True,
    )

    total_occ = sum(e["occurrences"] for e in agg.values())
    notes = [
        "SHADOW MODE P0: classification only -- no context was dropped, summarized, or mutated",
        "route_label/reason_code/block_type are low-cardinality enums; block_hash is a salted SHA-256 fingerprint",
        "est_candidate_tokens is ADVISORY (an upper bound for a FUTURE router), not a realized saving",
        "user/system/skill prompts and safety/acceptance constraints are pinned to policy_must_keep and never routable",
        "classification is conservative: when uncertain, blocks are kept (likely_relevant)",
    ]

    return WorkerRoutingShadow(
        enabled=True,
        item_count=item_count,
        classified_block_count=len(agg),
        total_occurrences=total_occ,
        must_keep_block_count=must_keep_blocks,
        must_keep_occurrence_count=must_keep_occ,
        est_must_keep_tokens=est_must_keep_tokens,
        est_candidate_tokens_total=drop_tokens + summ_tokens,
        est_drop_candidate_tokens=drop_tokens,
        est_summarizable_candidate_tokens=summ_tokens,
        label_counts=label_counts,
        reason_counts=reason_counts,
        top_candidate_blocks=candidates[:top_n],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Parent Aggregation Artifacts — SHADOW MODE (P0 telemetry only)
# ---------------------------------------------------------------------------
# When a parent/orchestrator aggregates results from several workers, the same
# artifact body (a test log, a diff, a file dump, a review summary, ...) is often
# carried into the parent's LLM context once per worker and again in the parent's
# own roll-up -- paying for the same tokens several times. This section collects
# *telemetry only* so a future parent-aggregation dedup can be evaluated offline:
# it groups EXACT artifact bodies by salted content hash, classifies each body
# with a deterministic heuristic kind, and emits low-cardinality metadata +
# counters. It NEVER drops, summarizes, replaces, or mutates any context, and it
# NEVER emits raw artifact text, worker text, tool output, session ids, or
# system prompts.

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


@dataclass
class ArtifactSourceCount:
    """Provenance counter: occurrences of one artifact body from one source."""

    source_type: str
    count: int


@dataclass
class ParentAggregationGroup:
    """One EXACT artifact body observed 2+ times across parent/worker contexts.

    Salted hash + counters only -- never the body text.
    """

    content_hash: str
    artifact_kind: str
    canonical_source_type: str           # dominant origin, chosen deterministically
    occurrences: int
    char_length: int
    est_tokens: int
    est_duplicate_tokens: int            # ADVISORY: (occurrences - 1) * est_tokens
    source_type_counts: list[ArtifactSourceCount]  # provenance: tool_result xN, ...


@dataclass
class ArtifactKindStat:
    """Aggregate over all candidate artifact bodies of one kind."""

    artifact_kind: str
    group_count: int             # distinct bodies of this kind
    occurrence_count: int        # total occurrences of those bodies
    duplicate_group_count: int   # bodies seen >= 2 times
    est_tokens: int              # sum of est tokens for distinct bodies
    est_duplicate_tokens: int    # ADVISORY duplicate tokens for this kind


@dataclass
class ParentAggregationArtifacts:
    """Shadow-mode parent-aggregation artifact report (P0: telemetry only)."""

    enabled: bool
    item_count: int                  # candidate artifact items considered
    artifact_body_count: int         # distinct bodies (groups)
    total_occurrences: int
    duplicate_group_count: int
    est_total_tokens: int            # est tokens for distinct bodies
    est_duplicate_tokens: int        # ADVISORY duplicate-artifact tokens
    by_kind: list[ArtifactKindStat]
    source_type_counts: list[ArtifactSourceCount]   # provenance across candidates
    top_duplicate_groups: list[ParentAggregationGroup]
    notes: list[str] = field(default_factory=list)


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
        f"- Telemetry: {t.events} events, ~{t.tokens_saved} tokens saved, "
        f"coverage {t.coverage_ratio_pct}%",
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
        "--all-sessions",
        action="store_true",
        help="ignore --since-hours; scan all non-archived sessions and active messages",
    )
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
    parser.add_argument(
        "--disable-worker-routing-shadow",
        action="store_true",
        help=(
            "skip the shadow-mode Worker Context Routing classification "
            "(P0 data collection; enabled by default, never prunes context)"
        ),
    )
    parser.add_argument(
        "--disable-parent-aggregation",
        action="store_true",
        help=(
            "skip the shadow-mode Parent Aggregation Artifact telemetry "
            "(P0 telemetry only; enabled by default, never dedups/replaces context)"
        ),
    )
    parser.add_argument(
        "--min-artifact-chars", type=int, default=DEFAULT_MIN_ARTIFACT_CHARS
    )
    args = parser.parse_args()

    if not args.state_db.exists():
        raise SystemExit(f"Hermes state DB not found: {args.state_db}")

    # Harden for unattended cron use: never dump a traceback (which would echo
    # the DB path / SQL); emit only the exception class name and a non-zero code.
    try:
        tool_messages = load_tool_messages(
            args.state_db, since_hours=args.since_hours, all_sessions=args.all_sessions
        )
        llm_contents = load_llm_bound_content(
            args.state_db, since_hours=args.since_hours, all_sessions=args.all_sessions
        )
        heavy_sessions = load_heavy_sessions(
            args.state_db,
            since_hours=args.since_hours,
            salt=args.salt,
            top_n=args.top_n,
            all_sessions=args.all_sessions,
        )
        total_input = total_input_tokens(
            args.state_db, since_hours=args.since_hours, all_sessions=args.all_sessions
        )
        telemetry = parse_telemetry(
            args.telemetry_file,
            since_hours=args.since_hours,
            total_input_tokens=total_input,
            all_sessions=args.all_sessions,
        )
        report = build_report(
            date=args.date,
            since_hours=args.since_hours,
            salt=args.salt,
            tool_messages=tool_messages,
            heavy_sessions=heavy_sessions,
            telemetry=telemetry,
            llm_contents=llm_contents,
            all_sessions=args.all_sessions,
            min_block_chars=args.min_block_chars,
            min_block_repeat=args.min_block_repeat,
            large_output_chars=args.large_output_chars,
            top_n=args.top_n,
            worker_routing_shadow=not args.disable_worker_routing_shadow,
            parent_aggregation_shadow=not args.disable_parent_aggregation,
            min_artifact_chars=args.min_artifact_chars,
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
