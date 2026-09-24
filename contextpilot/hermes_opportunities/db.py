"""Read-only Hermes state-DB loaders.

All loaders open the SQLite DB in read-only mode and return either privacy-safe
dataclasses (``HeavySession``) or in-memory content carriers (``_ToolMessage``,
``_LLMContent``) whose ``content`` is for hashing only and must never be
emitted. Schema is probed defensively so older/newer DBs degrade gracefully.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

from .models import HeavySession, _LLMContent, _ToolMessage
from .privacy import _salted_hash


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


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
