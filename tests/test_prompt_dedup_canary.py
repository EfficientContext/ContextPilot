"""Tests for the default-off prompt-dedup canary (runtime prompt mutation).

The canary is the only place ContextPilot may actually rewrite prompt text. These
tests pin the safety gate: default off (no mutation), canary touches ONLY
same_type_skill_prompt_only duplicates (first occurrence kept), never touches
system/cross-type/user/assistant/tool content, honours the safety denylist and
the escape-hatch kill switch, never grows the payload, and emits metadata-only
telemetry with no raw prompt text.
"""
import json

from contextpilot.hermes_opportunities import (
    CANARY_DEDUP_CLASS,
    PROMPT_DEDUP_DISABLE_ENV,
    PROMPT_DEDUP_MODE_ENV,
    apply_prompt_dedup_canary,
    build_canary_telemetry_record,
    resolve_prompt_dedup_mode,
    _LLMContent,
)
from contextpilot.hermes_opportunities.prompt_dedup_canary import (
    _reference_string,
    _salted_hash,
)

SALT = "test-salt"
MIN = 40

# A benign skill block, comfortably longer than the reference placeholder so a
# replacement actually saves characters, and free of any denylist keyword.
LONG_SKILL = (
    "Example reusable skill paragraph describing how the helper reformats "
    "markdown tables into neat aligned columns for the reader."
)
SYS_BLOCK = (
    "Plain system narration paragraph describing the assistant persona and the "
    "general tone it should adopt across replies."
)


def _ref_len() -> int:
    return len(_reference_string("skill_prompt", _salted_hash(LONG_SKILL, SALT)))


# ---------------------------------------------------------------------------
# Mode resolution + escape hatch
# ---------------------------------------------------------------------------


def test_mode_defaults_to_off(monkeypatch):
    monkeypatch.delenv(PROMPT_DEDUP_MODE_ENV, raising=False)
    monkeypatch.delenv(PROMPT_DEDUP_DISABLE_ENV, raising=False)
    assert resolve_prompt_dedup_mode() == "off"


def test_mode_reads_env_values():
    assert resolve_prompt_dedup_mode({PROMPT_DEDUP_MODE_ENV: "shadow"}) == "shadow"
    assert resolve_prompt_dedup_mode({PROMPT_DEDUP_MODE_ENV: "CANARY"}) == "canary"
    # Unknown / garbage values fall back to the safe default.
    assert resolve_prompt_dedup_mode({PROMPT_DEDUP_MODE_ENV: "aggressive"}) == "off"


def test_disable_env_is_a_kill_switch():
    env = {PROMPT_DEDUP_MODE_ENV: "canary", PROMPT_DEDUP_DISABLE_ENV: "1"}
    assert resolve_prompt_dedup_mode(env) == "off"


# ---------------------------------------------------------------------------
# Default off never mutates
# ---------------------------------------------------------------------------


def test_default_off_does_not_change_payload(monkeypatch):
    monkeypatch.delenv(PROMPT_DEDUP_MODE_ENV, raising=False)
    monkeypatch.delenv(PROMPT_DEDUP_DISABLE_ENV, raising=False)
    contents = [_LLMContent("skill_prompt", f"{LONG_SKILL}\n{LONG_SKILL}\n{LONG_SKILL}")]
    before = contents[0].content
    result = apply_prompt_dedup_canary(contents, salt=SALT, min_block_chars=MIN)
    assert result.mode == "off"
    assert result.mutated is False
    assert result.blocks_replaced == 0
    assert result.chars_saved == 0
    assert contents[0].content == before  # payload byte-identical


def test_disable_env_blocks_mutation_even_with_canary_set(monkeypatch):
    monkeypatch.setenv(PROMPT_DEDUP_MODE_ENV, "canary")
    monkeypatch.setenv(PROMPT_DEDUP_DISABLE_ENV, "true")
    contents = [_LLMContent("skill_prompt", f"{LONG_SKILL}\n{LONG_SKILL}")]
    before = contents[0].content
    result = apply_prompt_dedup_canary(contents, salt=SALT, min_block_chars=MIN)
    assert result.mode == "off"
    assert contents[0].content == before


# ---------------------------------------------------------------------------
# Canary replaces only same_type_skill_prompt_only duplicates
# ---------------------------------------------------------------------------


def test_canary_replaces_later_skill_duplicates_keeps_first():
    contents = [_LLMContent("skill_prompt", f"{LONG_SKILL}\n{LONG_SKILL}\n{LONG_SKILL}")]
    result = apply_prompt_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    lines = contents[0].content.split("\n")
    assert lines[0] == LONG_SKILL  # first occurrence kept verbatim
    assert lines[1] != LONG_SKILL and lines[2] != LONG_SKILL  # later ones replaced
    assert lines[1] == lines[2]  # deterministic reference string
    assert result.mode == "canary"
    assert result.mutated is True
    assert result.blocks_replaced == 2
    assert result.prompt_dedup_class == CANARY_DEDUP_CLASS
    assert result.chars_saved == 2 * (len(LONG_SKILL) - _ref_len())


def test_canary_replacement_carries_no_raw_content():
    contents = [_LLMContent("skill_prompt", f"{LONG_SKILL}\n{LONG_SKILL}")]
    apply_prompt_dedup_canary(contents, salt=SALT, min_block_chars=MIN, mode="canary")
    ref_line = contents[0].content.split("\n")[1]
    # The reference holds only a type enum + salted hash, never the block text.
    assert "skill_prompt" in ref_line
    assert LONG_SKILL not in ref_line


def test_canary_replicates_across_two_skill_items():
    # Same block in two separate skill_prompt items: first item keeps it, the
    # occurrence in the second item is replaced.
    a = _LLMContent("skill_prompt", LONG_SKILL)
    b = _LLMContent("skill_prompt", LONG_SKILL)
    result = apply_prompt_dedup_canary(
        [a, b], salt=SALT, min_block_chars=MIN, mode="canary"
    )
    assert a.content == LONG_SKILL
    assert b.content != LONG_SKILL
    assert result.blocks_replaced == 1


# ---------------------------------------------------------------------------
# Canary must NOT touch other classes / roles
# ---------------------------------------------------------------------------


def test_canary_leaves_system_only_duplicates_untouched():
    contents = [_LLMContent("system_prompt", f"{SYS_BLOCK}\n{SYS_BLOCK}\n{SYS_BLOCK}")]
    before = contents[0].content
    result = apply_prompt_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    assert contents[0].content == before
    assert result.blocks_replaced == 0
    assert result.candidate_block_count == 0


def test_canary_leaves_cross_type_duplicates_untouched():
    # The same block appears in BOTH a system and a skill prompt -> cross-type,
    # never eligible for the skill-only canary.
    skill = _LLMContent("skill_prompt", f"{LONG_SKILL}\n{LONG_SKILL}")
    system = _LLMContent("system_prompt", LONG_SKILL)
    skill_before = skill.content
    result = apply_prompt_dedup_canary(
        [skill, system], salt=SALT, min_block_chars=MIN, mode="canary"
    )
    assert skill.content == skill_before
    assert result.blocks_replaced == 0
    assert result.candidate_block_count == 0


def test_canary_leaves_user_assistant_tool_untouched():
    contents = [
        _LLMContent("user_prompt", f"{LONG_SKILL}\n{LONG_SKILL}"),
        _LLMContent("assistant_context", f"{LONG_SKILL}\n{LONG_SKILL}"),
        _LLMContent("tool_result", f"{LONG_SKILL}\n{LONG_SKILL}"),
    ]
    befores = [c.content for c in contents]
    result = apply_prompt_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    assert [c.content for c in contents] == befores
    assert result.blocks_replaced == 0
    # Non system/skill items are not even scanned for candidates.
    assert result.item_count == 0


# ---------------------------------------------------------------------------
# Safety denylist + payload-growth guard
# ---------------------------------------------------------------------------


def test_denylisted_blocks_are_not_replaced():
    danger = (
        "You must always follow this required safety rule precisely and never "
        "skip it under any circumstances whatsoever here."
    )
    contents = [_LLMContent("skill_prompt", f"{danger}\n{danger}\n{danger}")]
    before = contents[0].content
    result = apply_prompt_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    assert contents[0].content == before  # untouched
    assert result.blocks_replaced == 0
    assert result.denylisted_block_count == 1


def test_canary_never_grows_payload_for_short_duplicates():
    # A duplicate shorter than the reference placeholder would grow if replaced,
    # so it is left alone.
    short = "Short but over forty chars skill helper line."
    assert len(short) < _ref_len()
    contents = [_LLMContent("skill_prompt", f"{short}\n{short}\n{short}")]
    before = contents[0].content
    result = apply_prompt_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    assert contents[0].content == before
    assert result.blocks_replaced == 0
    assert result.chars_saved == 0


# ---------------------------------------------------------------------------
# Shadow mode measures but does not mutate
# ---------------------------------------------------------------------------


def test_shadow_measures_candidates_without_mutating():
    contents = [_LLMContent("skill_prompt", f"{LONG_SKILL}\n{LONG_SKILL}\n{LONG_SKILL}")]
    before = contents[0].content
    result = apply_prompt_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="shadow"
    )
    assert contents[0].content == before  # never mutated
    assert result.mode == "shadow"
    assert result.mutated is False
    assert result.blocks_replaced == 0
    assert result.candidate_block_count == 1
    assert result.candidate_chars == 2 * len(LONG_SKILL)


# ---------------------------------------------------------------------------
# Telemetry: metadata-only, savings only when a real mutation happened
# ---------------------------------------------------------------------------


def test_telemetry_records_no_savings_when_off():
    contents = [_LLMContent("skill_prompt", f"{LONG_SKILL}\n{LONG_SKILL}")]
    result = apply_prompt_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="off"
    )
    record = build_canary_telemetry_record(result)
    assert record["prompt_dedup_mode"] == "off"
    assert record["prompt_dedup_chars_saved"] == 0
    assert record["prompt_dedup_blocks_replaced"] == 0
    assert record["chars_saved"] == 0


def test_telemetry_records_no_savings_in_shadow():
    contents = [_LLMContent("skill_prompt", f"{LONG_SKILL}\n{LONG_SKILL}")]
    result = apply_prompt_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="shadow"
    )
    record = build_canary_telemetry_record(result)
    assert record["prompt_dedup_mode"] == "shadow"
    # Shadow contributes nothing to the realized chars_saved total.
    assert record["chars_saved"] == 0
    assert record["prompt_dedup_chars_saved"] == 0


def test_telemetry_records_realized_savings_in_canary():
    contents = [_LLMContent("skill_prompt", f"{LONG_SKILL}\n{LONG_SKILL}\n{LONG_SKILL}")]
    result = apply_prompt_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    record = build_canary_telemetry_record(result)
    expected = 2 * (len(LONG_SKILL) - _ref_len())
    assert record["prompt_dedup_mode"] == "canary"
    assert record["prompt_dedup_class"] == CANARY_DEDUP_CLASS
    assert record["prompt_dedup_blocks_replaced"] == 2
    assert record["prompt_dedup_chars_saved"] == expected
    # The aggregate counter includes prompt dedup only because a mutation occurred.
    assert record["chars_saved"] == expected


def test_telemetry_is_metadata_only_no_prompt_text():
    contents = [_LLMContent("skill_prompt", f"{LONG_SKILL}\n{LONG_SKILL}")]
    result = apply_prompt_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    record = build_canary_telemetry_record(result)
    blob = json.dumps(record)
    assert LONG_SKILL not in blob
    # Only low-cardinality enums + integer counters are present.
    assert set(record) == {
        "prompt_dedup_mode",
        "prompt_dedup_class",
        "prompt_dedup_blocks_replaced",
        "prompt_dedup_chars_saved",
        "chars_saved",
    }
    for key in (
        "prompt_dedup_blocks_replaced",
        "prompt_dedup_chars_saved",
        "chars_saved",
    ):
        assert isinstance(record[key], int)
