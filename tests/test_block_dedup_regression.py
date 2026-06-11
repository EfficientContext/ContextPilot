"""Regression coverage for the block-dedup behavioral contract.

These tests lock in the guarantees the Hermes integration depends on:

1. Exact-identical tool-result chunks are replaced by short references.
2. Edited / near-duplicate content keeps the changed (delta) text verbatim and
   is NOT collapsed wholesale into an "identical" reference.
3. Genuinely different content (e.g. a different file in the same repo) is never
   claimed identical — nothing is deduped and the payload is left untouched.

The hashing is exact-content based; these tests guard against any future change
that weakens it (e.g. fuzzy matching that would hide new, unique content behind
a reference).
"""
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "contextpilot" / "dedup" / "block_dedup.py"
_spec = importlib.util.spec_from_file_location("contextpilot_block_dedup_test", MODULE_PATH)
block_dedup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(block_dedup)

dedup_chat_completions = block_dedup.dedup_chat_completions


def _file_content(prefix: str = "compute_value", n: int = 80) -> str:
    """Deterministic multi-line tool-result body that chunks into many blocks."""
    return "\n".join(
        f"{i:3d}| def function_number_{i}(): return {prefix}({i}) + base_offset_value"
        for i in range(n)
    )


def _two_tool_results(content_a: str, content_b: str) -> dict:
    """Two Read tool results in a single chat-completions body."""
    return {
        "messages": [
            {"role": "user", "content": "read the file"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "Read"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": content_a},
            {"role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "Read"}}]},
            {"role": "tool", "tool_call_id": "c2", "content": content_b},
        ]
    }


def test_exact_duplicate_tool_result_is_replaced_by_reference():
    content = _file_content()
    body = _two_tool_results(content, content)

    result = dedup_chat_completions(body)

    first = body["messages"][2]["content"]
    second = body["messages"][4]["content"]

    # Savings actually happened and were attributed to deduped blocks.
    assert result.chars_saved > 0
    assert result.blocks_deduped > 0

    # The first occurrence is untouched; the second is shortened and points back.
    assert first == content
    assert len(second) < len(content)
    assert "identical to earlier" in second


def test_edited_near_duplicate_preserves_delta_verbatim():
    content = _file_content()
    lines = content.split("\n")
    # Edit a single line in the middle — a realistic same-file edit between turns.
    delta = "40| TOTALLY_UNIQUE_EDITED_LINE_MARKER xyzzy brand new content not seen before"
    lines[40] = delta
    edited = "\n".join(lines)

    body = _two_tool_results(content, edited)
    result = dedup_chat_completions(body)

    second = body["messages"][4]["content"]

    # The unique edited text MUST survive verbatim — it must never be hidden
    # behind an "identical" reference.
    assert "TOTALLY_UNIQUE_EDITED_LINE_MARKER" in second
    assert delta in second

    # Identical surrounding blocks are still deduped (so this is not a no-op),
    # but the result is not collapsed into a single wholesale "identical" marker.
    assert result.blocks_deduped > 0
    assert result.blocks_deduped < result.blocks_total


def test_different_file_same_repo_is_not_claimed_identical():
    content = _file_content("compute_value")
    other = "\n".join(
        f"{i:3d}| class Widget_{i}: pass  # unrelated module, distinct content line"
        for i in range(80)
    )
    body = _two_tool_results(content, other)

    result = dedup_chat_completions(body)

    # No shared blocks -> nothing deduped and both payloads left byte-for-byte intact.
    assert result.chars_saved == 0
    assert result.blocks_deduped == 0
    assert body["messages"][2]["content"] == content
    assert body["messages"][4]["content"] == other


def test_single_changed_char_breaks_block_match():
    """A one-character change must produce a different hash (no fuzzy collapse)."""
    content = _file_content()
    # Flip exactly one character deep inside the body.
    idx = len(content) // 2
    mutated = content[:idx] + ("Z" if content[idx] != "Z" else "Q") + content[idx + 1:]

    body = _two_tool_results(content, mutated)
    dedup_chat_completions(body)

    second = body["messages"][4]["content"]
    # The block containing the mutation is preserved verbatim (not referenced away).
    mutated_line = mutated.split("\n")[content[:idx].count("\n")]
    assert mutated_line in second
