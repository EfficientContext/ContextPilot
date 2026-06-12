import importlib.util
import json
import sqlite3
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hermes_contextpilot_monitor.py"
spec = importlib.util.spec_from_file_location("hermes_contextpilot_monitor", MODULE_PATH)
monitor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = monitor
spec.loader.exec_module(monitor)


def _make_db(path: Path):
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
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            estimated_cost_usd REAL,
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
            reasoning TEXT,
            timestamp REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO sessions (
            id, source, started_at, message_count, tool_call_count,
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            reasoning_tokens, estimated_cost_usd, api_call_count, archived,
            system_prompt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "raw-session-id",
            "discord",
            4102444800.0,  # 2100-01-01, always inside test window
            4,
            2,
            1000,
            200,
            50,
            10,
            25,
            0.0123,
            3,
            0,
            "SECRET SYSTEM PROMPT",
        ),
    )
    conn.execute(
        """
        INSERT INTO messages (session_id, role, content, reasoning, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("raw-session-id", "user", "DO NOT READ ME", "PRIVATE", 4102444800.0),
    )
    conn.commit()
    conn.close()


def test_monitor_reads_metadata_only_and_hashes_session_ids(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db)
    log = tmp_path / "gateway.log"
    log.write_text(
        "2026-01-01 INFO [ContextPilot] Turn 2: saved 400 chars (~100 tokens) | cumulative: 400 chars (~100 tokens)\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "reports"

    metrics = monitor.load_session_metrics(db, since_hours=24 * 365 * 100, salt="test")
    report = monitor.build_report(
        metrics,
        date="2100-01-01",
        since_hours=24,
        log_stats=monitor.parse_contextpilot_savings(log, since_hours=24),
    )
    json_path, md_path = monitor.write_report(report, out_dir)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    md = md_path.read_text(encoding="utf-8")
    assert data["session_count"] == 1
    assert data["contextpilot_tokens_saved"] == 100
    assert data["estimated_input_token_reduction_pct"] > 0
    assert "raw-session-id" not in md
    assert "DO NOT READ ME" not in md
    assert "SECRET SYSTEM PROMPT" not in md
    assert data["top_token_sessions"][0]["session_hash"] != "raw-session-id"


def _write_telemetry(path, records):
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


def test_parse_telemetry_aggregates_recent_records(tmp_path):
    tel = tmp_path / "telemetry.jsonl"
    far_future = 4102444800.0  # 2100-01-01
    _write_telemetry(
        tel,
        [
            {"ts": far_future, "type": "turn", "session": "s1", "turn": 1,
             "chars_saved": 400, "tokens_saved": 100},
            {"ts": far_future, "type": "turn", "session": "s1", "turn": 2,
             "chars_saved": 200, "tokens_saved": 50},
            # Stale record far in the past must be excluded by the window.
            {"ts": 1000.0, "type": "turn", "session": "s0", "turn": 1,
             "chars_saved": 999999, "tokens_saved": 999999},
            "this is not json",
        ],
    )

    events, chars, tokens = monitor.parse_contextpilot_telemetry(tel, since_hours=24)
    assert events == 2
    assert chars == 600
    assert tokens == 150


def test_parse_telemetry_missing_file_is_safe(tmp_path):
    assert monitor.parse_contextpilot_telemetry(tmp_path / "nope.jsonl", since_hours=24) == (0, 0, 0)


def test_build_report_prefers_telemetry_over_logs(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db)
    metrics = monitor.load_session_metrics(db, since_hours=24 * 365 * 100, salt="test")

    report = monitor.build_report(
        metrics,
        date="2100-01-01",
        since_hours=24,
        log_stats=(5, 4000, 1000),
        telemetry_stats=(2, 600, 150),
    )
    # Telemetry is authoritative when present; logs are not summed on top.
    assert report.contextpilot_tokens_saved == 150
    assert report.contextpilot_chars_saved == 600
    assert report.contextpilot_telemetry_events == 2
    assert report.contextpilot_log_events == 5
    assert report.contextpilot_savings_source == "telemetry"


def test_build_report_falls_back_to_logs_without_telemetry(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db)
    metrics = monitor.load_session_metrics(db, since_hours=24 * 365 * 100, salt="test")

    report = monitor.build_report(
        metrics,
        date="2100-01-01",
        since_hours=24,
        log_stats=(5, 4000, 1000),
        telemetry_stats=(0, 0, 0),
    )
    assert report.contextpilot_tokens_saved == 1000
    assert report.contextpilot_savings_source == "gateway-log"
