import importlib.util
import json
import sys
import time
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "contextpilot_savings.py"
spec = importlib.util.spec_from_file_location("contextpilot_savings", MODULE_PATH)
savings = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = savings
spec.loader.exec_module(savings)


def _write_jsonl(path, records):
    lines = []
    for r in records:
        lines.append(r if isinstance(r, str) else json.dumps(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_missing_file_is_safe_and_prompts_user():
    summary = savings.summarize_telemetry(Path("/no/such/telemetry.jsonl"), since_hours=24)
    assert summary["file_exists"] is False
    assert summary["events"] == 0
    assert summary["tokens_saved"] == 0
    text = savings.render_text(summary)
    assert "No ContextPilot telemetry found" in text
    assert "Restart Hermes" in text
    assert "Enable ContextPilot" in text


def test_time_window_filtering(tmp_path):
    tel = tmp_path / "telemetry.jsonl"
    now = time.time()
    _write_jsonl(
        tel,
        [
            {"ts": now - 3600, "type": "turn", "chars_saved": 400, "tokens_saved": 100},
            {"ts": now - 7200, "type": "turn", "chars_saved": 200, "tokens_saved": 50},
            # ~2 days ago: outside a 24h window.
            {"ts": now - 48 * 3600, "type": "turn", "chars_saved": 999999, "tokens_saved": 999999},
            # Unknown timestamps are counted in all-time mode only; for a time
            # window we skip them because we cannot prove they are in-window.
            {"type": "turn", "chars_saved": 100, "tokens_saved": 25},
        ],
    )
    summary = savings.summarize_telemetry(tel, since_hours=24)
    assert summary["events"] == 2
    assert summary["chars_saved"] == 600
    assert summary["tokens_saved"] == 150
    assert summary["avg_tokens_per_event"] == 75.0
    assert summary["window_start_iso"] is not None
    assert summary["skipped_lines"] == 1


def test_all_time_mode_includes_old_records(tmp_path):
    tel = tmp_path / "telemetry.jsonl"
    now = time.time()
    _write_jsonl(
        tel,
        [
            {"ts": now - 3600, "type": "turn", "chars_saved": 400, "tokens_saved": 100},
            {"ts": now - 48 * 3600, "type": "turn", "chars_saved": 200, "tokens_saved": 50},
            # No timestamp at all should still count in all-time mode.
            {"type": "turn", "chars_saved": 40, "tokens_saved": 10},
        ],
    )
    summary = savings.summarize_telemetry(tel, since_hours=None)
    assert summary["all_time"] is True
    assert summary["since_hours"] is None
    assert summary["window_start_iso"] is None
    assert summary["events"] == 3
    assert summary["chars_saved"] == 640
    assert summary["tokens_saved"] == 160


def test_malformed_lines_skipped_and_counted(tmp_path):
    tel = tmp_path / "telemetry.jsonl"
    now = time.time()
    _write_jsonl(
        tel,
        [
            {"ts": now, "type": "turn", "chars_saved": 400, "tokens_saved": 100},
            "this is not json",
            "[1, 2, 3]",  # valid json, but not a dict
            {"ts": now, "type": "turn", "chars_saved": "not a number"},  # bad field
            {"ts": now, "type": "turn", "chars_saved": -1, "tokens_saved": 0},
            {"ts": now, "type": "turn", "chars_saved": 10, "tokens_saved": -1},
            "",  # blank line ignored, not counted as skipped
            {"ts": now, "type": "turn", "chars_saved": 100},  # tokens derived from chars
        ],
    )
    summary = savings.summarize_telemetry(tel, since_hours=None)
    assert summary["events"] == 2
    assert summary["chars_saved"] == 500
    # 100 (explicit) + 100//4 (derived) = 125
    assert summary["tokens_saved"] == 125
    assert summary["skipped_lines"] == 5


def test_json_output_schema_and_no_raw_content(tmp_path, capsys):
    tel = tmp_path / "telemetry.jsonl"
    now = time.time()
    # Telemetry should never carry content, but prove the summary cannot leak it
    # even if a stray field appears in a record.
    _write_jsonl(
        tel,
        [
            {
                "ts": now,
                "type": "turn",
                "chars_saved": 400,
                "tokens_saved": 100,
                "content": "SECRET CONVERSATION TEXT",
                "system_prompt": "SECRET SYSTEM PROMPT",
            },
        ],
    )
    rc = savings.main(["--telemetry-file", str(tel), "--all-time", "--format", "json"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)

    expected_keys = {
        "telemetry_file",
        "file_exists",
        "all_time",
        "since_hours",
        "window_start_iso",
        "events",
        "chars_saved",
        "tokens_saved",
        "avg_tokens_per_event",
        "skipped_lines",
    }
    assert set(data.keys()) == expected_keys
    assert data["events"] == 1
    assert data["tokens_saved"] == 100
    assert "SECRET CONVERSATION TEXT" not in out
    assert "SECRET SYSTEM PROMPT" not in out


def test_text_output_renders_savings(tmp_path, capsys):
    tel = tmp_path / "telemetry.jsonl"
    now = time.time()
    _write_jsonl(
        tel,
        [{"ts": now, "type": "turn", "chars_saved": 400, "tokens_saved": 100}],
    )
    rc = savings.main(["--telemetry-file", str(tel), "--since-hours", "24"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ContextPilot token savings (last 24h)" in out
    assert "Telemetry tokens saved" in out
    assert str(tel) in out


def test_no_events_in_window_message(tmp_path, capsys):
    tel = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        tel,
        [{"ts": 1000.0, "type": "turn", "chars_saved": 400, "tokens_saved": 100}],
    )
    rc = savings.main(["--telemetry-file", str(tel), "--since-hours", "24"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "No ContextPilot savings recorded" in out
    assert "--all-time" in out
