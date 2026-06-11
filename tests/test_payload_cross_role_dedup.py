"""Cross-role exact-block dedup within a single LLM-bound payload.

This is the safest concrete ContextPilot optimization: inside ONE OpenAI chat
payload, an exact repeated block that already appears in an earlier message
(system / skill prompt, user, assistant, or tool result) is replaced — in the
*later* message only — by a short reference back to the earlier copy. The LLM
has already seen one full copy in the same request, so no information is lost.

Hard safety contract these tests lock in:

1. Exact repeated blocks ACROSS DIFFERENT ROLES are deduped. The first
   (earliest, in document order) occurrence keeps its full text; later
   occurrences are shortened to a reference pointing "above".
2. References point to an EARLIER block in the SAME payload (never forward,
   never the first occurrence).
3. A one-character-different / near-duplicate block is NEVER collapsed — its
   unique text survives verbatim.
4. Genuinely different content is left byte-for-byte intact.
5. No raw block/message content is ever written to telemetry.
"""
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "contextpilot" / "dedup" / "block_dedup.py"
_spec = importlib.util.spec_from_file_location("contextpilot_block_dedup_xrole", MODULE_PATH)
block_dedup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(block_dedup)

dedup_chat_completions = block_dedup.dedup_chat_completions

REFERENCE_MARKER = "identical to earlier"


def _instruction_block(prefix: str = "always follow safety rule", n: int = 30) -> str:
    """A deterministic multi-line instruction/skill block that chunks cleanly."""
    return "\n".join(
        f"INSTRUCTION LINE {i}: {prefix} number {i} carefully and verbatim every time"
        for i in range(n)
    )


def _cross_role_payload(sys_block, user_block, tool_block, asst_block) -> dict:
    """A single chat payload where the same instruction block can recur by role."""
    return {
        "messages": [
            {"role": "system", "content": "You are a coding agent.\n" + sys_block + "\nEnd system."},
            {"role": "user", "content": "Please remember the rules:\n" + user_block + "\nThanks."},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "Read"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "File header\n" + tool_block + "\nFooter."},
            {"role": "assistant", "content": "Acknowledging the rules:\n" + asst_block + "\nDone."},
        ]
    }


def test_repeated_block_across_roles_is_deduped_first_copy_kept():
    block = _instruction_block()
    body = _cross_role_payload(block, block, block, block)
    original_system = body["messages"][0]["content"]

    result = dedup_chat_completions(body)

    system_after = body["messages"][0]["content"]
    user_after = body["messages"][1]["content"]
    tool_after = body["messages"][3]["content"]
    asst_after = body["messages"][4]["content"]

    # Real savings, attributed to deduped blocks.
    assert result.chars_saved > 0
    assert result.blocks_deduped > 0

    # First (earliest) occurrence — the system prompt — is left fully intact.
    assert system_after == original_system

    # Every later role that repeats the block is shortened and references "above".
    assert len(user_after) < len(body["messages"][1]["content"]) or REFERENCE_MARKER in user_after
    assert REFERENCE_MARKER in user_after, "user-role duplicate must be deduped"
    assert REFERENCE_MARKER in tool_after, "tool-role duplicate must be deduped"
    assert REFERENCE_MARKER in asst_after, "assistant-role duplicate must be deduped"


def test_reference_points_backward_only_never_first_occurrence():
    block = _instruction_block()
    body = _cross_role_payload(block, block, block, block)
    dedup_chat_completions(body)

    # The system message is first; it must never become a reference to itself or
    # to anything later in the payload.
    assert REFERENCE_MARKER not in body["messages"][0]["content"]


def test_near_duplicate_block_survives_verbatim():
    block = _instruction_block()
    # One unique line differs in the user copy — a one-line delta.
    lines = block.split("\n")
    lines[15] = "INSTRUCTION LINE 15: UNIQUE_DELTA_MARKER_qwerty brand new never-seen directive"
    edited = "\n".join(lines)

    body = _cross_role_payload(block, edited, block, block)
    dedup_chat_completions(body)

    user_after = body["messages"][1]["content"]
    # The changed line MUST survive verbatim — never hidden behind a reference.
    assert "UNIQUE_DELTA_MARKER_qwerty" in user_after


def test_single_char_difference_is_not_collapsed():
    block = _instruction_block()
    idx = len(block) // 2
    mutated = block[:idx] + ("Z" if block[idx] != "Z" else "Q") + block[idx + 1:]

    body = _cross_role_payload(block, mutated, block, block)
    dedup_chat_completions(body)

    user_after = body["messages"][1]["content"]
    mutated_line = mutated.split("\n")[block[:idx].count("\n")]
    assert mutated_line in user_after


def test_genuinely_different_content_left_intact():
    block = _instruction_block()
    other = "\n".join(
        f"UNRELATED ROW {i}: a completely different paragraph about widgets and gears {i}"
        for i in range(30)
    )
    body = _cross_role_payload(block, other, other, other)
    user_before = body["messages"][1]["content"]
    tool_before = body["messages"][3]["content"]

    result = dedup_chat_completions(body)

    assert result.chars_saved == 0
    assert result.blocks_deduped == 0
    assert body["messages"][1]["content"] == user_before
    assert body["messages"][3]["content"] == tool_before


def test_no_raw_block_content_in_plugin_telemetry(monkeypatch, tmp_path):
    """End-to-end through the Hermes engine: telemetry stays metadata-only."""
    import sys
    import types

    # Minimal fake Hermes surface so __init__.py imports cleanly.
    agent_pkg = types.ModuleType("agent")
    context_engine_mod = types.ModuleType("agent.context_engine")
    context_compressor_mod = types.ModuleType("agent.context_compressor")

    class FakeContextEngine:
        threshold_percent = 0.75

        def get_status(self):
            return {}

    class FakeContextCompressor(FakeContextEngine):
        def __init__(self, **kwargs):
            self.threshold_tokens = 0
            self.context_length = 0
            self.protect_first_n = 3
            self.protect_last_n = 6
            self.compression_count = 0

        def on_session_start(self, session_id, **kwargs):
            return None

    context_engine_mod.ContextEngine = FakeContextEngine
    context_compressor_mod.ContextCompressor = FakeContextCompressor
    agent_pkg.context_engine = context_engine_mod
    agent_pkg.context_compressor = context_compressor_mod
    monkeypatch.setitem(sys.modules, "agent", agent_pkg)
    monkeypatch.setitem(sys.modules, "agent.context_engine", context_engine_mod)
    monkeypatch.setitem(sys.modules, "agent.context_compressor", context_compressor_mod)

    run_agent_mod = types.ModuleType("run_agent")

    class FakeAIAgent:
        @staticmethod
        def _sanitize_api_messages(messages):
            return messages

    run_agent_mod.AIAgent = FakeAIAgent
    monkeypatch.setitem(sys.modules, "run_agent", run_agent_mod)

    module_path = REPO_ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location("contextpilot_plugin_xrole", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "_check_reorder", lambda: False)
    monkeypatch.setattr(module, "_CONTEXTPILOT_AVAILABLE", False)

    telemetry = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("CONTEXTPILOT_TELEMETRY_FILE", str(telemetry))

    block = _instruction_block()
    secret_line = "INSTRUCTION LINE 0: always follow safety rule number 0 carefully and verbatim every time"
    assert secret_line in block

    engine = module.ContextPilotEngine()
    engine.on_session_start("session-XR", model="test-model")
    body = _cross_role_payload(block, block, block, block)
    _out, stats = engine.optimize_api_messages(body["messages"])

    assert stats["chars_saved"] > 0

    assert telemetry.exists()
    raw = telemetry.read_text(encoding="utf-8")
    # No raw block/message content may ever leak into telemetry.
    assert secret_line not in raw
    assert "INSTRUCTION LINE" not in raw
    for record in (json.loads(l) for l in raw.splitlines() if l.strip()):
        forbidden = {"content", "messages", "prompt", "system_prompt", "text", "tool_calls"}
        assert forbidden.isdisjoint(record.keys())
