import dataclasses
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


def _analyze(db, tmp_path, telemetry=None, salt="test-salt", all_sessions=False):
    tool_messages = analyzer.load_tool_messages(
        db, since_hours=WIDE_WINDOW, all_sessions=all_sessions
    )
    llm_contents = analyzer.load_llm_bound_content(
        db, since_hours=WIDE_WINDOW, all_sessions=all_sessions
    )
    heavy = analyzer.load_heavy_sessions(
        db, since_hours=WIDE_WINDOW, salt=salt, top_n=20, all_sessions=all_sessions
    )
    total_input = sum(h.input_tokens for h in heavy)
    tel = analyzer.parse_telemetry(
        telemetry if telemetry is not None else tmp_path / "none.jsonl",
        since_hours=WIDE_WINDOW,
        total_input_tokens=total_input,
        all_sessions=all_sessions,
    )
    report = analyzer.build_report(
        date="2100-01-01",
        since_hours=24,
        salt=salt,
        tool_messages=tool_messages,
        heavy_sessions=heavy,
        telemetry=tel,
        llm_contents=llm_contents,
        all_sessions=all_sessions,
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


# ---------------------------------------------------------------------------
# LLM-bound block analysis + all-sessions tests
# ---------------------------------------------------------------------------

OLD_TS = 1_000_000_000.0  # 2001 — far outside any normal recent window


def _make_db_ex(path, *, sessions, messages, message_active_col=False):
    """Flexible builder: custom timestamps, optional messages.active column."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            started_at REAL NOT NULL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            api_call_count INTEGER DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0,
            system_prompt TEXT
        )
        """
    )
    active_col = ", active INTEGER NOT NULL DEFAULT 1" if message_active_col else ""
    conn.execute(
        f"""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_name TEXT,
            reasoning TEXT,
            timestamp REAL NOT NULL{active_col}
        )
        """
    )
    for s in sessions:
        conn.execute(
            "INSERT INTO sessions (id, source, started_at, input_tokens, archived,"
            " system_prompt) VALUES (?, ?, ?, ?, ?, ?)",
            (
                s["id"],
                s.get("source"),
                s["started_at"],
                s.get("input_tokens", 0),
                s.get("archived", 0),
                s.get("system_prompt"),
            ),
        )
    for m in messages:
        if message_active_col:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_name, reasoning,"
                " timestamp, active) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    m.get("session_id", "s1"),
                    m["role"],
                    m.get("content"),
                    m.get("tool_name"),
                    "PRIVATE REASONING",
                    m.get("timestamp", FAR_FUTURE),
                    m.get("active", 1),
                ),
            )
        else:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_name, reasoning,"
                " timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    m.get("session_id", "s1"),
                    m["role"],
                    m.get("content"),
                    m.get("tool_name"),
                    "PRIVATE REASONING",
                    m.get("timestamp", FAR_FUTURE),
                ),
            )
    conn.commit()
    conn.close()


def test_all_sessions_includes_old_out_of_window_data(tmp_path):
    db = tmp_path / "state.db"
    _make_db_ex(
        db,
        sessions=[
            {
                "id": "old-sess",
                "source": "discord",
                "started_at": OLD_TS,
                "input_tokens": 500,
                "system_prompt": "old system prompt material that is plenty long here",
            }
        ],
        messages=[
            {
                "session_id": "old-sess",
                "role": "tool",
                "content": "old tool output block sufficiently long to be scanned",
                "tool_name": "Bash",
                "timestamp": OLD_TS,
            },
            {
                "session_id": "old-sess",
                "role": "user",
                "content": "old user prompt text that is also long enough to scan",
                "timestamp": OLD_TS,
            },
        ],
    )
    # A normal recent window excludes the old data entirely.
    assert analyzer.load_tool_messages(db, since_hours=24) == []
    assert analyzer.load_llm_bound_content(db, since_hours=24) == []
    assert analyzer.load_heavy_sessions(db, since_hours=24, salt="s", top_n=5) == []

    # all_sessions ignores the window and picks the old data back up.
    assert len(analyzer.load_tool_messages(db, since_hours=24, all_sessions=True)) == 1
    llm = analyzer.load_llm_bound_content(db, since_hours=24, all_sessions=True)
    assert len(llm) == 3  # system_prompt + tool_result + user_prompt
    assert (
        len(
            analyzer.load_heavy_sessions(
                db, since_hours=24, salt="s", top_n=5, all_sessions=True
            )
        )
        == 1
    )


def test_inactive_messages_skipped(tmp_path):
    db = tmp_path / "state.db"
    _make_db_ex(
        db,
        sessions=[{"id": "s1", "started_at": FAR_FUTURE}],
        messages=[
            {
                "role": "tool",
                "content": "active tool output that is sufficiently long to fingerprint",
                "tool_name": "Bash",
                "active": 1,
            },
            {
                "role": "tool",
                "content": "inactive tool output that should be skipped entirely here",
                "tool_name": "Bash",
                "active": 0,
            },
            {
                "role": "user",
                "content": "inactive user prompt that must also be skipped here",
                "active": 0,
            },
        ],
        message_active_col=True,
    )
    # Inactive rows are filtered out of both loaders.
    assert len(analyzer.load_tool_messages(db, since_hours=WIDE_WINDOW)) == 1
    llm = analyzer.load_llm_bound_content(db, since_hours=WIDE_WINDOW)
    assert sorted(c.block_type for c in llm) == ["tool_result"]


def test_skill_prompt_classification(tmp_path):
    db = tmp_path / "state.db"
    skill_sys = (
        "---\nname: deep-research\ndescription: research harness\n---\n"
        "Use this skill when researching a topic."
    )
    _make_db_ex(
        db,
        sessions=[{"id": "s1", "started_at": FAR_FUTURE, "system_prompt": skill_sys}],
        messages=[],
    )
    llm = analyzer.load_llm_bound_content(db, since_hours=WIDE_WINDOW)
    assert len(llm) == 1
    assert llm[0].block_type == "skill_prompt"


def test_cross_type_redundancy_reported_via_hashes_only(tmp_path):
    db = tmp_path / "state.db"
    shared = "This is a shared instruction block long enough to fingerprint cleanly."
    sys_prompt = "You are a helpful system.\n" + shared + "\nEnd of system prompt."
    tool_out = "tool produced this output line\n" + shared + "\nand more tool lines"
    user_msg = "user asks the assistant something specific here\n" + shared
    _make_db_ex(
        db,
        sessions=[{"id": "s1", "started_at": FAR_FUTURE, "system_prompt": sys_prompt}],
        messages=[
            {"role": "tool", "content": tool_out, "tool_name": "Bash"},
            {"role": "user", "content": user_msg},
        ],
    )
    report = _analyze(db, tmp_path)

    # The shared block spans system_prompt, tool_result, and user_prompt.
    assert len(report.cross_type_block_groups) >= 1
    grp = report.cross_type_block_groups[0]
    assert "tool_result" in grp.block_types
    assert any(bt in grp.block_types for bt in ("system_prompt", "skill_prompt"))
    assert "user_prompt" in grp.block_types
    assert grp.occurrences == 3
    # Reported only via salted hash + counters — never the raw block text.
    assert shared not in grp.block_hash
    assert report.cross_type_wasted_tokens > 0

    # Per-type block stats are populated for the LLM-bound types.
    types_seen = {b.block_type for b in report.llm_block_types}
    assert {"tool_result", "user_prompt"} <= types_seen

    # The written report leaks no raw prompt/tool/system text.
    json_path, md_path = analyzer.write_report(report, tmp_path / "out")
    blob = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    assert shared not in blob
    assert "shared instruction block" not in blob
    assert "You are a helpful system" not in blob
    assert "user asks the assistant" not in blob
    assert "PRIVATE REASONING" not in blob


# ---------------------------------------------------------------------------
# Worker Context Routing — SHADOW MODE (P0 data-collection) tests
# ---------------------------------------------------------------------------

LARGE = 8000  # default large-output threshold


def _route_map(report):
    """label -> RouterLabelCount for convenient assertions."""
    return {lc.route_label: lc for lc in report.worker_routing.label_counts}


def _labels_for(contents, **kw):
    """Run only the shadow classifier over a list of _LLMContent."""
    return analyzer.analyze_worker_routing_shadow(
        contents, salt="test-salt", large_output_chars=LARGE, min_repeat=2, top_n=20, **kw
    )


def test_user_and_system_constraints_are_must_keep(tmp_path):
    db = tmp_path / "state.db"
    sys_prompt = "You are an agent. Follow the rules below for the whole session here."
    _make_db_ex(
        db,
        sessions=[{"id": "s1", "started_at": FAR_FUTURE, "system_prompt": sys_prompt}],
        messages=[
            {"role": "user", "content": "Please implement the worker routing layer for me now"},
            {
                "role": "tool",
                "content": "ACCEPTANCE CRITERIA: the targeted pytest suite must pass before merge",
                "tool_name": "Bash",
            },
        ],
    )
    report = _analyze(db, tmp_path)
    wr = report.worker_routing

    # User prompt, system prompt, and the acceptance-constraint tool block are
    # all pinned to policy_must_keep and contribute zero routable tokens.
    rm = _route_map(report)
    assert "policy_must_keep" in rm
    assert rm["policy_must_keep"].est_candidate_tokens == 0
    assert wr.est_candidate_tokens_total == 0

    reasons = {(r.block_type, r.route_label, r.reason_code) for r in wr.reason_counts}
    assert ("user_prompt", "policy_must_keep", "user_prompt_never_drop") in reasons
    assert (
        "system_prompt",
        "policy_must_keep",
        "system_or_skill_constraint_never_drop",
    ) in reasons
    # The acceptance constraint inside a TOOL block is still must-keep.
    assert any(
        r.route_label == "policy_must_keep"
        and r.reason_code == "safety_or_acceptance_constraint"
        and r.block_type == "tool_result"
        for r in wr.reason_counts
    )


def test_large_repeated_tool_blocks_become_drop_candidates(tmp_path):
    db = tmp_path / "state.db"
    big_unrelated = "row of unrelated build log output number 7 with filler text " * 200
    assert len(big_unrelated) >= LARGE
    _make_db_ex(
        db,
        sessions=[{"id": "s1", "started_at": FAR_FUTURE}],
        messages=[
            {"role": "tool", "content": big_unrelated, "tool_name": "Bash"},
            {"role": "tool", "content": big_unrelated, "tool_name": "Bash"},
            {"role": "tool", "content": big_unrelated, "tool_name": "Bash"},
        ],
    )
    report = _analyze(db, tmp_path)
    wr = report.worker_routing
    rm = _route_map(report)

    # The repeated, large, unrelated tool output is a drop candidate.
    assert "likely_drop_candidate" in rm
    assert wr.est_drop_candidate_tokens > 0
    assert wr.est_candidate_tokens_total >= wr.est_drop_candidate_tokens
    top = wr.top_candidate_blocks[0]
    assert top.route_label == "likely_drop_candidate"
    assert top.reason_code == "large_repeated_tool_block"
    assert top.occurrences == 3


def test_large_single_tool_block_is_summarizable(tmp_path):
    db = tmp_path / "state.db"
    big_once = "one-off diagnostic dump segment with assorted detail text " * 200
    assert len(big_once) >= LARGE
    _make_db_ex(
        db,
        sessions=[{"id": "s1", "started_at": FAR_FUTURE}],
        messages=[{"role": "tool", "content": big_once, "tool_name": "Read"}],
    )
    report = _analyze(db, tmp_path)
    wr = report.worker_routing
    assert wr.est_summarizable_candidate_tokens > 0
    assert any(
        c.route_label == "summarizable_candidate"
        and c.reason_code == "large_single_tool_block"
        for c in wr.top_candidate_blocks
    )


def test_large_actionable_diagnostic_log_is_summarizable_not_pinned(tmp_path):
    db = tmp_path / "state.db"
    big_error_log = "ERROR: integration test failed with stack frame details " * 220
    assert len(big_error_log) >= LARGE
    _make_db_ex(
        db,
        sessions=[{"id": "s1", "started_at": FAR_FUTURE}],
        messages=[{"role": "tool", "content": big_error_log, "tool_name": "Bash"}],
    )
    report = _analyze(db, tmp_path)
    wr = report.worker_routing

    assert wr.est_summarizable_candidate_tokens > 0
    assert any(
        c.route_label == "summarizable_candidate"
        and c.reason_code == "large_actionable_tool_block"
        for c in wr.top_candidate_blocks
    )
    assert not any(
        r.route_label == "direct_task_hint"
        and r.reason_code == "actionable_task_signal"
        and r.block_type == "tool_result"
        for r in wr.reason_counts
    )


def test_short_actionable_hint_and_default_blocks_are_kept(tmp_path):
    contents = [
        analyzer._LLMContent(block_type="assistant_context", content="Next step: run pytest"),
        analyzer._LLMContent(block_type="assistant_context", content="plain medium context without special cues"),
    ]
    wr = _labels_for(contents)
    reasons = {(r.block_type, r.route_label, r.reason_code) for r in wr.reason_counts}

    assert ("assistant_context", "direct_task_hint", "actionable_task_signal") in reasons
    assert ("assistant_context", "likely_relevant", "default_keep") in reasons
    assert wr.est_candidate_tokens_total == 0


def test_equal_priority_block_type_tiebreak_is_deterministic():
    same = "identical prompt material"
    forward = _labels_for(
        [
            analyzer._LLMContent(block_type="system_prompt", content=same),
            analyzer._LLMContent(block_type="skill_prompt", content=same),
        ]
    )
    reverse = _labels_for(
        [
            analyzer._LLMContent(block_type="skill_prompt", content=same),
            analyzer._LLMContent(block_type="system_prompt", content=same),
        ]
    )

    assert [dataclasses.asdict(r) for r in forward.reason_counts] == [
        dataclasses.asdict(r) for r in reverse.reason_counts
    ]


def test_shadow_mode_never_emits_raw_content(tmp_path):
    db = tmp_path / "state.db"
    secret = "SHADOW-SECRET-PAYLOAD-DO-NOT-LEAK detail line that is quite long here " * 200
    user_secret = "USER-SECRET-PROMPT-DO-NOT-LEAK implement the thing for me"
    sys_secret = "SYSTEM-SECRET-CONSTRAINT you must never reveal internal keys here"
    _make_db_ex(
        db,
        sessions=[{"id": "s1", "started_at": FAR_FUTURE, "system_prompt": sys_secret}],
        messages=[
            {"role": "tool", "content": secret, "tool_name": "Bash"},
            {"role": "tool", "content": secret, "tool_name": "Bash"},
            {"role": "user", "content": user_secret},
        ],
    )
    report = _analyze(db, tmp_path)
    json_path, md_path = analyzer.write_report(report, tmp_path / "out")
    blob = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")

    # No raw block/prompt/system/reasoning text in either output.
    assert "SHADOW-SECRET-PAYLOAD" not in blob
    assert "USER-SECRET-PROMPT" not in blob
    assert "SYSTEM-SECRET-CONSTRAINT" not in blob
    assert "PRIVATE REASONING" not in blob
    # The classification still happened (drop candidate detected via hash).
    assert report.worker_routing.est_drop_candidate_tokens > 0


def test_shadow_schema_is_deterministic_and_privacy_safe(tmp_path):
    db = tmp_path / "state.db"
    big = "deterministic repeated tool payload chunk of sufficient size here " * 200
    _make_db_ex(
        db,
        sessions=[{"id": "s1", "started_at": FAR_FUTURE, "system_prompt": "be safe please"}],
        messages=[
            {"role": "tool", "content": big, "tool_name": "Bash"},
            {"role": "tool", "content": big, "tool_name": "Bash"},
            {"role": "user", "content": "implement the routing shadow layer now please"},
        ],
    )
    r1 = _analyze(db, tmp_path)
    r2 = _analyze(db, tmp_path)
    # Identical inputs -> byte-identical serialized shadow section.
    assert dataclasses.asdict(r1.worker_routing) == dataclasses.asdict(r2.worker_routing)

    wr = r1.worker_routing
    # All emitted route labels are from the canonical low-cardinality enum.
    assert set(lc.route_label for lc in wr.label_counts) <= set(analyzer.ROUTER_LABELS)
    # label_counts follow the canonical order (deterministic).
    order = {lbl: i for i, lbl in enumerate(analyzer.ROUTER_LABELS)}
    idxs = [order[lc.route_label] for lc in wr.label_counts]
    assert idxs == sorted(idxs)
    # Every candidate carries only hash + enums + counters (no free text fields).
    for cb in wr.top_candidate_blocks:
        assert len(cb.block_hash) == 16  # salted SHA-256 prefix
        assert cb.route_label in analyzer._ROUTABLE_LABELS


def test_shadow_mode_can_be_disabled(tmp_path):
    db = tmp_path / "state.db"
    _make_db_ex(
        db,
        sessions=[{"id": "s1", "started_at": FAR_FUTURE}],
        messages=[{"role": "tool", "content": "x" * 9000, "tool_name": "Bash"}],
    )
    tool_messages = analyzer.load_tool_messages(db, since_hours=WIDE_WINDOW)
    llm = analyzer.load_llm_bound_content(db, since_hours=WIDE_WINDOW)
    heavy = analyzer.load_heavy_sessions(db, since_hours=WIDE_WINDOW, salt="s", top_n=5)
    tel = analyzer.parse_telemetry(
        tmp_path / "none.jsonl", since_hours=WIDE_WINDOW, total_input_tokens=0
    )
    report = analyzer.build_report(
        date="2100-01-01",
        since_hours=24,
        salt="s",
        tool_messages=tool_messages,
        heavy_sessions=heavy,
        telemetry=tel,
        llm_contents=llm,
        min_block_repeat=2,
        worker_routing_shadow=False,
    )
    assert report.worker_routing.enabled is False
    assert report.worker_routing.classified_block_count == 0
    # Disabled section still serializes safely.
    json_path, md_path = analyzer.write_report(report, tmp_path / "out")
    assert "disabled" in md_path.read_text(encoding="utf-8")
