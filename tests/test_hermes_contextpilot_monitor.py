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
