import importlib.util
import json
import sqlite3
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_hermes_context_opportunities.py"
)
spec = importlib.util.spec_from_file_location(
    "analyze_hermes_context_opportunities", MODULE_PATH
)
analyzer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analyzer
spec.loader.exec_module(analyzer)


FAR_FUTURE = 4102444800.0  # 2100-01-01, always inside a generous test window
WIDE_WINDOW = 24 * 365 * 100


def _make_db(path: Path, messages, *, sessions=None):
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            api_call_count INTEGER DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0,
            system_prompt TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_name TEXT,
            reasoning TEXT,
            timestamp REAL NOT NULL
        )
        """
    )
    # tuple layout: (id, source, _placeholder, tool_call_count, message_count,
    #                input_tokens, output_tokens, api_call_count, system_prompt)
    for s in sessions or [
        ("raw-session-id", "discord", None, 4, 6, 1000, 200, 3, "SECRET SYSTEM PROMPT")
    ]:
        conn.execute(
            """
            INSERT INTO sessions (
                id, source, started_at, tool_call_count, message_count,
                input_tokens, output_tokens, api_call_count, archived, system_prompt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (s[0], s[1], FAR_FUTURE, s[3], s[4], s[5], s[6], s[7], s[8]),
        )
    for role, content, tool_name in messages:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_name, reasoning, timestamp)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("raw-session-id", role, content, tool_name, "PRIVATE REASONING", FAR_FUTURE),
        )
    conn.commit()
    conn.close()


def _analyze(db, tmp_path, telemetry=None, salt="test-salt"):
    tool_messages = analyzer.load_tool_messages(db, since_hours=WIDE_WINDOW)
    heavy = analyzer.load_heavy_sessions(
        db, since_hours=WIDE_WINDOW, salt=salt, top_n=20
    )
    total_input = sum(h.input_tokens for h in heavy)
    tel = analyzer.parse_telemetry(
        telemetry if telemetry is not None else tmp_path / "none.jsonl",
        since_hours=WIDE_WINDOW,
        total_input_tokens=total_input,
    )
    report = analyzer.build_report(
        date="2100-01-01",
        since_hours=24,
        salt=salt,
        tool_messages=tool_messages,
        heavy_sessions=heavy,
        telemetry=tel,
        min_block_repeat=2,
    )
    return report


def test_no_raw_content_leaks_in_reports(tmp_path):
    db = tmp_path / "state.db"
    secret = "TOP-SECRET-TOOL-OUTPUT-PAYLOAD-DO-NOT-LEAK " * 10
    _make_db(
        db,
        [
            ("tool", secret, "Bash"),
            ("tool", secret, "Bash"),
            ("user", "DO NOT READ ME USER TEXT", None),
        ],
    )
    report = _analyze(db, tmp_path)
    json_path, md_path = analyzer.write_report(report, tmp_path / "out")

    blob = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    # Raw content, prompts, reasoning, and raw session ids must never appear.
    assert "TOP-SECRET-TOOL-OUTPUT-PAYLOAD" not in blob
    assert "DO NOT READ ME" not in blob
    assert "SECRET SYSTEM PROMPT" not in blob
    assert "PRIVATE REASONING" not in blob
    assert "raw-session-id" not in blob
    # But the duplicate was still detected via hashing.
    assert report.duplicate_tool_output_groups == 1
    assert report.heavy_sessions[0].session_hash != "raw-session-id"


def test_exact_duplicate_tool_outputs_counted(tmp_path):
    db = tmp_path / "state.db"
    payload = "identical output line one\nidentical output line two\n" * 3
    _make_db(
        db,
        [
            ("tool", payload, "Read"),
            ("tool", payload, "Read"),
            ("tool", payload, "Read"),
        ],
    )
    report = _analyze(db, tmp_path)
    assert report.duplicate_tool_output_groups == 1
    group = report.exact_duplicate_groups[0]
    assert group.occurrences == 3
    assert group.tool_name == "Read"
    # Two of the three sends are pure waste.
    assert group.est_wasted_tokens == group.est_tokens * 2
    assert report.duplicate_tool_output_wasted_tokens == group.est_wasted_tokens


def test_near_or_different_content_not_exact_duplicate(tmp_path):
    db = tmp_path / "state.db"
    base = "the quick brown fox jumps over the lazy dog " * 5
    near = base + "X"  # one char different -> different hash
    other = "completely unrelated tool output content here " * 5
    _make_db(
        db,
        [
            ("tool", base, "Bash"),
            ("tool", near, "Bash"),
            ("tool", other, "Bash"),
        ],
    )
    report = _analyze(db, tmp_path)
    # No two outputs are byte-identical -> zero exact-duplicate groups.
    assert report.duplicate_tool_output_groups == 0
    assert report.duplicate_tool_output_wasted_tokens == 0


def test_malformed_telemetry_tolerated(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db, [("tool", "some output", "Bash")])
    tel = tmp_path / "telemetry.jsonl"
    tel.write_text(
        "\n".join(
            [
                json.dumps({"ts": FAR_FUTURE, "chars_saved": 400, "tokens_saved": 100}),
                json.dumps({"ts": FAR_FUTURE, "chars_saved": 200}),  # missing tokens_saved
                "this is not json at all",
                json.dumps([1, 2, 3]),  # not a dict
                json.dumps({"ts": FAR_FUTURE, "note": "no counters here"}),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = _analyze(db, tmp_path, telemetry=tel)
    t = report.telemetry
    # Two valid records aggregated; second infers tokens from chars (200//4=50).
    assert t.events == 2
    assert t.chars_saved == 600
    assert t.tokens_saved == 150
    # Non-json, non-dict, and missing-counter lines are skipped, not fatal.
    assert t.malformed_records_skipped == 3
    assert t.coverage_ratio_pct > 0


def test_repeated_blocks_and_large_outputs(tmp_path):
    db = tmp_path / "state.db"
    shared = "this shared boilerplate block is long enough to be fingerprinted"
    big = "x" * 9000
    _make_db(
        db,
        [
            ("tool", shared + "\nunique tail alpha that is also sufficiently long here", "Bash"),
            ("tool", shared + "\nunique tail beta that is also sufficiently long here", "Bash"),
            ("tool", big, "Read"),
        ],
    )
    report = _analyze(db, tmp_path)
    assert any(b.occurrences >= 2 for b in report.repeated_blocks)
    read_stat = next(s for s in report.large_tool_outputs_by_tool if s.tool_name == "Read")
    assert read_stat.large_output_count == 1


def test_missing_telemetry_file_is_safe(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db, [("tool", "out", "Bash")])
    report = _analyze(db, tmp_path, telemetry=tmp_path / "nope.jsonl")
    assert report.telemetry.events == 0
    assert report.telemetry.malformed_records_skipped == 0
