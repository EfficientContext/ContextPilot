"""Trace-derived validation-set builder (read-only, local-only output).

Reads a Hermes local SQLite state DB in read-only mode and exports a *fixed*
JSONL corpus of replayable cases under a local, gitignored directory. The corpus
captures the exact LLM-bound payload (system/skill prompts + ordered messages)
so future accuracy-affecting or runtime-payload-changing changes can be validated
against a stable set instead of an ad-hoc "run once and see".

Privacy contract:

* The DB is opened ``mode=ro`` and never written.
* Raw ``content`` is written ONLY into the local JSONL artifact (default under
  the user's home, and the in-repo convenience dir is gitignored). It is never
  committed and never placed in the privacy-safe manifest.
* Case ids are salted hashes of the session id -- the raw session id is never
  emitted. The manifest carries only a salt *fingerprint*, counters, and enums.
* Sampling is conservative by default (small ``--limit``, a time window) so the
  artifact stays a representative sample, not a full export.

This module deliberately reuses the analyzer's read-only DB helpers and salted
hashing rather than re-implementing them, so the privacy primitives stay in one
place.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

from contextpilot.hermes_opportunities.db import (
    _classify_system_prompt,
    _connect_readonly,
    _message_block_type,
    _window_cutoff,
)
from contextpilot.hermes_opportunities.privacy import (
    _assert_no_forbidden_keys,
    _salt_fingerprint,
    _salted_hash,
)

from .models import (
    DEFAULT_CASE_LIMIT,
    DEFAULT_MIN_INPUT_TOKENS,
    DEFAULT_MIN_MESSAGES,
    DEFAULT_SINCE_HOURS,
    VALIDATION_SET_SCHEMA_VERSION,
    TraceCase,
    TraceMessage,
)


# Manifest dictionaries are passed through the shared privacy guard, which
# rejects forbidden key names such as "system_prompt" or "user_prompt". Keep the
# human meaning but avoid those exact raw-content-shaped key names in reports.
_BLOCK_TYPE_REPORT_KEYS = {
    "system_prompt": "system_ctx",
    "skill_prompt": "skill_ctx",
    "user_prompt": "user_ctx",
    "assistant_context": "assistant_ctx",
    "tool_result": "tool_result",
}


def _report_block_type_key(block_type: str) -> str:
    return _BLOCK_TYPE_REPORT_KEYS.get(block_type, block_type.replace("prompt", "ctx"))

DEFAULT_STATE_DB = Path("/root/.hermes/state.db")
# Default output lives OUTSIDE the repo, under the user's home, so a generated
# corpus can never be committed by accident. The in-repo ``.contextpilot_validation/``
# convenience dir is also gitignored for users who prefer to keep it local-to-repo.
DEFAULT_OUT_DIR = Path.home() / "contextpilot" / "validation_sets"
DEFAULT_SALT = "contextpilot-trace-validation-v1"


def _order_clause(mcols: set[str]) -> str:
    """Pick a deterministic message ordering column, preferring explicit ids."""
    if "id" in mcols:
        return "messages.id ASC"
    if "timestamp" in mcols:
        return "messages.timestamp ASC, messages.rowid ASC"
    return "messages.rowid ASC"


def load_trace_cases(
    db_path: Path,
    *,
    since_hours: int,
    salt: str,
    limit: int,
    all_sessions: bool = False,
    min_input_tokens: int = DEFAULT_MIN_INPUT_TOKENS,
    min_messages: int = DEFAULT_MIN_MESSAGES,
    include_system_prompt: bool = True,
) -> list[TraceCase]:
    """Load up to ``limit`` replayable cases from the Hermes state DB.

    Sessions are filtered by the time window (unless ``all_sessions``), archival
    flag, and ``min_input_tokens``, then ordered by ``input_tokens`` descending so
    the heaviest (most worth validating) sessions are sampled first. Per session
    the optional classified system prompt is emitted first, followed by active
    messages in deterministic order. Content is read in-memory only.
    """
    cutoff = _window_cutoff(since_hours, all_sessions)
    conn = _connect_readonly(db_path)
    try:
        scols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        mcols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        if "id" not in scols:
            return []

        wanted = ["id", "source", "input_tokens"]
        select_cols = [c if c in scols else f"NULL AS {c}" for c in wanted]
        has_sys = include_system_prompt and "system_prompt" in scols
        select_cols.append("system_prompt" if has_sys else "NULL AS system_prompt")

        where: list[str] = []
        params: list[object] = []
        if cutoff is not None and "started_at" in scols:
            where.append("started_at >= ?")
            params.append(cutoff)
        if "archived" in scols:
            where.append("archived = 0")
        if min_input_tokens > 0 and "input_tokens" in scols:
            where.append("input_tokens >= ?")
            params.append(min_input_tokens)
        sql = f"SELECT {', '.join(select_cols)} FROM sessions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        if "input_tokens" in scols:
            sql += " ORDER BY input_tokens DESC"
        session_rows = conn.execute(sql, params).fetchall()

        # Pre-resolve the per-session message query shape once.
        has_content = "content" in mcols
        has_role = "role" in mcols
        has_tool = "tool_name" in mcols
        has_session_fk = "session_id" in mcols
        msg_select = ", ".join(
            [
                "messages.role" if has_role else "NULL AS role",
                "messages.content",
                "messages.tool_name" if has_tool else "NULL AS tool_name",
            ]
        )
        order_by = _order_clause(mcols)

        cases: list[TraceCase] = []
        for sid, source, input_tokens, system_prompt in session_rows:
            if len(cases) >= limit:
                break
            messages: list[TraceMessage] = []

            if has_sys and system_prompt is not None:
                text = str(system_prompt)
                messages.append(
                    TraceMessage(
                        role="system",
                        block_type=_classify_system_prompt(text),
                        content=text,
                    )
                )

            if has_content and has_session_fk:
                mwhere = ["messages.content IS NOT NULL", "messages.session_id = ?"]
                mparams: list[object] = [sid]
                if has_role:
                    mwhere.append(
                        "messages.role IN ('system', 'user', 'assistant', 'tool')"
                    )
                if "active" in mcols:
                    mwhere.append("messages.active = 1")
                msql = (
                    f"SELECT {msg_select} FROM messages "
                    f"WHERE {' AND '.join(mwhere)} ORDER BY {order_by}"
                )
                for role, content, tool_name in conn.execute(msql, mparams):
                    if content is None:
                        continue
                    messages.append(
                        TraceMessage(
                            role=role,
                            block_type=_message_block_type(role, tool_name),
                            content=str(content),
                        )
                    )

            if len(messages) < min_messages:
                continue
            cases.append(
                TraceCase(
                    case_id=_salted_hash(str(sid), salt),
                    source=source,
                    input_tokens=int(input_tokens or 0),
                    message_count=len(messages),
                    messages=messages,
                )
            )
    finally:
        conn.close()
    return cases


def case_to_json(case: TraceCase) -> dict:
    """Serialize a case for the LOCAL JSONL artifact (includes raw content)."""
    return {
        "schema_version": VALIDATION_SET_SCHEMA_VERSION,
        "case_id": case.case_id,
        "source": case.source,
        "input_tokens": case.input_tokens,
        "message_count": case.message_count,
        "messages": [asdict(m) for m in case.messages],
    }


def build_manifest(
    cases: list[TraceCase],
    *,
    date: str,
    salt: str,
    since_hours: int,
    all_sessions: bool,
    min_input_tokens: int,
    include_system_prompt: bool,
    corpus_filename: str,
) -> dict:
    """Build the PRIVACY-SAFE manifest: counters, enums, and a salt fingerprint.

    Never contains raw content; passed through the forbidden-key guard before it
    is returned so a future regression cannot smuggle content in via this path.
    """
    by_source: dict[str, int] = {}
    by_block_type: dict[str, int] = {}
    total_messages = 0
    for case in cases:
        key = case.source or "unknown"
        by_source[key] = by_source.get(key, 0) + 1
        total_messages += case.message_count
        for m in case.messages:
            report_key = _report_block_type_key(m.block_type)
            by_block_type[report_key] = by_block_type.get(report_key, 0) + 1

    manifest = {
        "schema_version": VALIDATION_SET_SCHEMA_VERSION,
        "generated_date": date,
        "salt_fingerprint": _salt_fingerprint(salt),
        "window": "all_sessions" if all_sessions else f"last_{since_hours}h",
        "since_hours": since_hours,
        "all_sessions": all_sessions,
        "min_input_tokens": min_input_tokens,
        "include_system_prompt": include_system_prompt,
        "corpus_file": corpus_filename,
        "case_count": len(cases),
        "total_messages": total_messages,
        "cases_by_source": by_source,
        "messages_by_block_type": by_block_type,
        "privacy_note": (
            "manifest is metadata-only (salted ids, counters, enums); the corpus "
            "JSONL holds raw content and is local-only / gitignored, never committed"
        ),
    }
    _assert_no_forbidden_keys(manifest)
    return manifest


def write_validation_set(
    cases: list[TraceCase], manifest: dict, out_dir: Path, corpus_filename: str
) -> tuple[Path, Path]:
    """Write the local JSONL corpus and its privacy-safe manifest sidecar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / corpus_filename
    manifest_path = out_dir / (corpus_filename + ".manifest.json")
    with corpus_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case_to_json(case), ensure_ascii=False) + "\n")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return corpus_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a trace-derived ContextPilot validation set from a local "
            "Hermes state DB. Raw content is written ONLY to a local/gitignored "
            "JSONL corpus; the manifest is privacy-safe."
        )
    )
    parser.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB)
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--since-hours", type=int, default=DEFAULT_SINCE_HOURS)
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="ignore --since-hours; scan all non-archived sessions",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_CASE_LIMIT,
        help=f"max number of cases to export (default {DEFAULT_CASE_LIMIT})",
    )
    parser.add_argument(
        "--min-input-tokens",
        type=int,
        default=DEFAULT_MIN_INPUT_TOKENS,
        help="only include sessions with at least this many input tokens",
    )
    parser.add_argument(
        "--min-messages",
        type=int,
        default=DEFAULT_MIN_MESSAGES,
        help="drop cases with fewer than this many LLM-bound messages",
    )
    parser.add_argument(
        "--include-system-prompt",
        dest="include_system_prompt",
        action="store_true",
        default=True,
        help="include the session system/skill prompt as the first message (default)",
    )
    parser.add_argument(
        "--no-system-prompt",
        dest="include_system_prompt",
        action="store_false",
        help="exclude session system/skill prompts from the corpus",
    )
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--date", default=dt.date.today().isoformat())
    args = parser.parse_args(argv)

    if not args.state_db.exists():
        raise SystemExit(f"Hermes state DB not found: {args.state_db}")

    out_dir = args.out if args.out is not None else DEFAULT_OUT_DIR
    corpus_filename = f"validation_set_{args.date}.jsonl"

    # Cron-safe: never dump a traceback (which could echo the DB path or SQL);
    # emit only the exception class name and a non-zero exit code.
    try:
        cases = load_trace_cases(
            args.state_db,
            since_hours=args.since_hours,
            salt=args.salt,
            limit=args.limit,
            all_sessions=args.all_sessions,
            min_input_tokens=args.min_input_tokens,
            min_messages=args.min_messages,
            include_system_prompt=args.include_system_prompt,
        )
        manifest = build_manifest(
            cases,
            date=args.date,
            salt=args.salt,
            since_hours=args.since_hours,
            all_sessions=args.all_sessions,
            min_input_tokens=args.min_input_tokens,
            include_system_prompt=args.include_system_prompt,
            corpus_filename=corpus_filename,
        )
        corpus_path, manifest_path = write_validation_set(
            cases, manifest, out_dir, corpus_filename
        )
    except Exception as exc:  # noqa: BLE001 - cron-safe: class name only, no payload
        print(json.dumps({"ok": False, "error": type(exc).__name__}))
        return 1

    # Stdout is privacy-safe: paths + counters only (raw content stays in the file).
    print(
        json.dumps(
            {
                "ok": True,
                "corpus": str(corpus_path),
                "manifest": str(manifest_path),
                "case_count": manifest["case_count"],
                "total_messages": manifest["total_messages"],
            },
            ensure_ascii=False,
        )
    )
    return 0
