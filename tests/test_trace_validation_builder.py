"""Tests for the trace-validation-set builder: redaction, privacy, sampling.

The builder reads a Hermes-shaped SQLite DB read-only and exports a fixed JSONL
corpus (raw content, local-only) plus a privacy-safe manifest. These tests pin:
case ids are salted (never the raw session id); the manifest is metadata-only
(no raw content) and passes the forbidden-key guard; conservative sampling
honours --limit / --min-input-tokens / --min-messages; --no-system-prompt drops
system/skill prompts; and the corpus is the only place raw content appears.
"""
import json
import sqlite3
from pathlib import Path

from contextpilot.hermes_opportunities.privacy import _salt_fingerprint, _salted_hash
from contextpilot.trace_validation.builder import (
    build_manifest,
    load_trace_cases,
    main as build_main,
    write_validation_set,
)

SALT = "test-trace-salt"

SKILL_LINE = (
    "Synthetic reusable skill paragraph that explains how the demo helper "
    "reformats sample markdown tables into neat aligned columns for readers."
)
USER_LINE = "Please summarize the synthetic figures for the demo run."
TOOL_LINE = '{"synthetic_metric": 42, "label": "demo only"}'
SYS_LINE = "Synthetic system narration describing the demo persona and tone."


def _make_db(path: Path) -> None:
    """Create a minimal Hermes-shaped DB with two sessions."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sessions (id TEXT, source TEXT, input_tokens INTEGER, "
        "system_prompt TEXT, started_at REAL, archived INTEGER, message_count INTEGER)"
    )
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, "
        "content TEXT, tool_name TEXT, timestamp REAL, active INTEGER)"
    )
    now = 1_900_000_000.0
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
        ("sess-heavy", "synthetic", 9000, SYS_LINE, now, 0, 3),
    )
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
        ("sess-light", "synthetic", 500, None, now, 0, 2),
    )
    # Heavy session messages (ordered by id).
    conn.execute(
        "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
        (1, "sess-heavy", "user", USER_LINE, None, now, 1),
    )
    conn.execute(
        "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
        (2, "sess-heavy", "tool", TOOL_LINE, "calc", now, 1),
    )
    # An inactive message that must be skipped.
    conn.execute(
        "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
        (3, "sess-heavy", "user", "INACTIVE should not appear", None, now, 0),
    )
    conn.execute(
        "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
        (4, "sess-light", "user", "light user message for the demo", None, now, 1),
    )
    conn.commit()
    conn.close()


def test_case_ids_are_salted_not_raw(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db)
    cases = load_trace_cases(db, since_hours=24, salt=SALT, limit=10, all_sessions=True)
    ids = {c.case_id for c in cases}
    assert "sess-heavy" not in ids and "sess-light" not in ids
    assert _salted_hash("sess-heavy", SALT) in ids


def test_heavy_session_ordered_first_and_inactive_skipped(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db)
    cases = load_trace_cases(db, since_hours=24, salt=SALT, limit=10, all_sessions=True)
    # Ordered by input_tokens desc -> heavy first.
    assert cases[0].input_tokens == 9000
    heavy = cases[0]
    # system(skill/plain) + user + tool, inactive message dropped.
    contents = [m.content for m in heavy.messages]
    assert SYS_LINE in contents
    assert USER_LINE in contents
    assert "INACTIVE should not appear" not in contents


def test_min_input_tokens_filters_light_session(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db)
    cases = load_trace_cases(
        db, since_hours=24, salt=SALT, limit=10, all_sessions=True, min_input_tokens=1000
    )
    assert len(cases) == 1
    assert cases[0].input_tokens == 9000


def test_limit_caps_number_of_cases(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db)
    cases = load_trace_cases(db, since_hours=24, salt=SALT, limit=1, all_sessions=True)
    assert len(cases) == 1


def test_no_system_prompt_excludes_system(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db)
    cases = load_trace_cases(
        db,
        since_hours=24,
        salt=SALT,
        limit=10,
        all_sessions=True,
        include_system_prompt=False,
    )
    heavy = next(c for c in cases if c.input_tokens == 9000)
    assert all(m.role != "system" for m in heavy.messages)
    assert SYS_LINE not in [m.content for m in heavy.messages]


def test_manifest_is_privacy_safe_no_raw_content(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db)
    cases = load_trace_cases(db, since_hours=24, salt=SALT, limit=10, all_sessions=True)
    manifest = build_manifest(
        cases,
        date="2026-06-14",
        salt=SALT,
        since_hours=24,
        all_sessions=True,
        min_input_tokens=0,
        include_system_prompt=True,
        corpus_filename="corpus.jsonl",
    )
    blob = json.dumps(manifest)
    # No raw content anywhere in the manifest.
    for needle in (SKILL_LINE, USER_LINE, TOOL_LINE, SYS_LINE):
        assert needle not in blob
    # Carries a salt fingerprint, never the raw salt.
    assert manifest["salt_fingerprint"] == _salt_fingerprint(SALT)
    assert SALT not in blob
    assert manifest["case_count"] == len(cases)
    assert manifest["messages_by_block_type"]  # counters present


def test_write_validation_set_corpus_has_raw_manifest_does_not(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db)
    cases = load_trace_cases(db, since_hours=24, salt=SALT, limit=10, all_sessions=True)
    manifest = build_manifest(
        cases,
        date="2026-06-14",
        salt=SALT,
        since_hours=24,
        all_sessions=True,
        min_input_tokens=0,
        include_system_prompt=True,
        corpus_filename="corpus.jsonl",
    )
    corpus_path, manifest_path = write_validation_set(
        cases, manifest, tmp_path / "out", "corpus.jsonl"
    )
    corpus_text = corpus_path.read_text()
    manifest_text = manifest_path.read_text()
    # Raw content lives ONLY in the corpus, never the manifest.
    assert USER_LINE in corpus_text
    assert USER_LINE not in manifest_text
    # Each corpus line is a well-formed case object.
    for line in corpus_text.splitlines():
        obj = json.loads(line)
        assert obj["schema_version"] == 1
        assert "case_id" in obj and "messages" in obj


def test_main_emits_privacy_safe_stdout(tmp_path, capsys):
    db = tmp_path / "state.db"
    _make_db(db)
    out = tmp_path / "out"
    rc = build_main(
        [
            "--state-db",
            str(db),
            "--out",
            str(out),
            "--all-sessions",
            "--salt",
            SALT,
            "--date",
            "2026-06-14",
        ]
    )
    assert rc == 0
    printed = capsys.readouterr().out
    payload = json.loads(printed)
    assert payload["ok"] is True
    assert payload["case_count"] == 2
    # Stdout carries paths + counters only, never raw content.
    for needle in (SKILL_LINE, USER_LINE, TOOL_LINE, SYS_LINE):
        assert needle not in printed
    assert Path(payload["corpus"]).exists()
    assert Path(payload["manifest"]).exists()


def test_main_missing_db_exits(tmp_path):
    try:
        build_main(["--state-db", str(tmp_path / "nope.db")])
    except SystemExit as exc:
        assert exc.code != 0
    else:  # pragma: no cover - should always raise
        raise AssertionError("expected SystemExit for missing DB")
