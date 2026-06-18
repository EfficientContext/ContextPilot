"""RED-phase tests for the provenance-aware tool-artifact reuse canary.

This canary is the second (and, like the prompt-dedup canary, default-OFF)
runtime mutation path. Where the prompt-dedup canary rewrites *skill_prompt*
lines, this one dedups whole **artifact bodies** carried by ``tool_result`` and
``assistant_context`` items: it keeps the FIRST full artifact body verbatim and
replaces a later EXACT duplicate body (regardless of which of the two artifact
block types it appears in -- provenance-aware) with a deterministic, strictly
shorter reference string that points back at the canonical body via a salted
hash.

The safety contract these tests pin:

1. ``off`` (default) and ``shadow`` never mutate the payload; only ``canary``
   replaces a later exact-duplicate artifact body, always keeping the first one.
2. Only ``tool_result`` / ``assistant_context`` artifact bodies are mutable;
   ``system_prompt`` / ``user_prompt`` / ``skill_prompt`` (and any other
   non-artifact content) are protected and survive byte-identical.
3. A reference is valid only if it resolves to an EARLIER canonical full body in
   the same payload; the runner/validation gate fails a dangling reference or a
   protected-content mutation.
4. Telemetry / the runner report carry REALIZED ``chars_saved`` only when an
   actual mutation happened, and never any raw artifact content.

Production is implemented in ``contextpilot.hermes_opportunities.artifact_dedup_canary``.
Fixtures are synthetic only.
"""
import json

from contextpilot.hermes_opportunities.artifact_dedup_canary import (
    ARTIFACT_DEDUP_CANARY_REFERENCE_TEMPLATE,
    ARTIFACT_DEDUP_CLASS,
    ARTIFACT_DEDUP_DISABLE_ENV,
    ARTIFACT_DEDUP_MODE_ENV,
    MUTABLE_ARTIFACT_BLOCK_TYPES,
    ArtifactDedupCanaryResult,
    ArtifactSpanLink,
    apply_artifact_dedup_canary,
    build_artifact_canary_telemetry_record,
    dangling_artifact_references,
    resolve_artifact_dedup_mode,
    _artifact_reference_string,
    _segment_fenced_blocks,
)
from contextpilot.hermes_opportunities.models import _LLMContent
from contextpilot.hermes_opportunities.privacy import _salted_hash
from contextpilot.trace_validation.runner import (
    ARTIFACT_INVARIANT_NAMES,
    assert_report_privacy_safe,
    render_markdown,
    report_to_dict,
    run_artifact_validation,
)

SALT = "test-artifact-salt"
MIN = 40

# A synthetic artifact body comfortably longer than the reference placeholder so
# a replacement actually shrinks the payload. Free of secrets/raw user text.
LONG_ARTIFACT = (
    "Synthetic tool artifact body: rows=128 cols=8 checksum=alpha-bravo-charlie "
    "delta-echo. Summary of the synthetic computation produced purely for this "
    "test fixture and nothing else, padded to comfortably exceed the reference."
)
# A non-artifact protected block (system narration); duplicated to prove the
# canary never touches it.
SYS_BLOCK = (
    "Synthetic system narration describing the assistant persona and the general "
    "tone it should adopt across replies in this fixture."
)
# Just over min_block_chars but shorter than any reference placeholder, so a
# replacement would GROW the payload and must be skipped.
SHORT_ARTIFACT = "Short synthetic artifact body just over forty chars."
LONG_FENCE_BLOCK = (
    "```log\n"
    "synthetic provenance artifact line 001: worker output checksum=alpha\n"
    "synthetic provenance artifact line 002: worker output checksum=bravo\n"
    "synthetic provenance artifact line 003: worker output checksum=charlie\n"
    "```"
)
FENCED_PARENT_ARTIFACT = (
    "Parent aggregation summary before first artifact.\n"
    f"{LONG_FENCE_BLOCK}\n"
    "Short prose between artifacts must survive byte-identical.\n"
    f"{LONG_FENCE_BLOCK}\n"
    "Parent aggregation summary after duplicate artifact."
)
SOURCE_SPAN_BLOCK = (
    "worker-span-line-001 provenance payload alpha bravo charlie\n"
    "worker-span-line-002 provenance payload delta echo foxtrot\n"
    "worker-span-line-003 provenance payload golf hotel india"
)
SOURCE_SPAN_TOOL = (
    "tool preamble stays canonical\n"
    f"{SOURCE_SPAN_BLOCK}\n"
    "tool epilogue stays canonical"
)
SOURCE_SPAN_PARENT = (
    "parent summary before copied worker span\n"
    f"{SOURCE_SPAN_BLOCK}\n"
    "parent summary after copied worker span"
)


def _ref(body: str, *, canonical_type: str = "tool_result") -> str:
    """The deterministic reference string a canary would emit for ``body``."""
    return _artifact_reference_string(canonical_type, _salted_hash(body, SALT))


def _ref_len(canonical_type: str = "tool_result") -> int:
    return len(_ref(LONG_ARTIFACT, canonical_type=canonical_type))


def _block_ref(block: str, *, canonical_type: str = "tool_result#block") -> str:
    return _artifact_reference_string(canonical_type, _salted_hash(block, SALT))


def _source_span_link() -> ArtifactSpanLink:
    src_start = SOURCE_SPAN_TOOL.index(SOURCE_SPAN_BLOCK)
    tgt_start = SOURCE_SPAN_PARENT.index(SOURCE_SPAN_BLOCK)
    return ArtifactSpanLink(
        source_index=0,
        source_start=src_start,
        source_end=src_start + len(SOURCE_SPAN_BLOCK),
        target_index=1,
        target_start=tgt_start,
        target_end=tgt_start + len(SOURCE_SPAN_BLOCK),
    )


def _span_ref(span: str, *, canonical_type: str = "tool_result#span") -> str:
    return _artifact_reference_string(canonical_type, _salted_hash(span, SALT))


# ---------------------------------------------------------------------------
# Mode resolution + escape hatch (default OFF)
# ---------------------------------------------------------------------------


def test_mode_defaults_to_off(monkeypatch):
    monkeypatch.delenv(ARTIFACT_DEDUP_MODE_ENV, raising=False)
    monkeypatch.delenv(ARTIFACT_DEDUP_DISABLE_ENV, raising=False)
    assert resolve_artifact_dedup_mode() == "off"


def test_mode_reads_env_values():
    assert resolve_artifact_dedup_mode({ARTIFACT_DEDUP_MODE_ENV: "shadow"}) == "shadow"
    assert resolve_artifact_dedup_mode({ARTIFACT_DEDUP_MODE_ENV: "CANARY"}) == "canary"
    # Unknown / garbage values fall back to the safe default.
    assert resolve_artifact_dedup_mode({ARTIFACT_DEDUP_MODE_ENV: "aggressive"}) == "off"


def test_disable_env_is_a_kill_switch():
    env = {ARTIFACT_DEDUP_MODE_ENV: "canary", ARTIFACT_DEDUP_DISABLE_ENV: "1"}
    assert resolve_artifact_dedup_mode(env) == "off"


# ---------------------------------------------------------------------------
# (1) off (default) and shadow never mutate
# ---------------------------------------------------------------------------


def test_default_off_does_not_change_payload(monkeypatch):
    monkeypatch.delenv(ARTIFACT_DEDUP_MODE_ENV, raising=False)
    monkeypatch.delenv(ARTIFACT_DEDUP_DISABLE_ENV, raising=False)
    contents = [
        _LLMContent("tool_result", LONG_ARTIFACT),
        _LLMContent("tool_result", LONG_ARTIFACT),
    ]
    before = [c.content for c in contents]
    result = apply_artifact_dedup_canary(contents, salt=SALT, min_block_chars=MIN)
    assert result.mode == "off"
    assert result.mutated is False
    assert result.blocks_replaced == 0
    assert result.chars_saved == 0
    assert result.item_count == 0  # off does not even scan
    assert [c.content for c in contents] == before  # byte-identical


def test_disable_env_blocks_mutation_even_with_canary_set(monkeypatch):
    monkeypatch.setenv(ARTIFACT_DEDUP_MODE_ENV, "canary")
    monkeypatch.setenv(ARTIFACT_DEDUP_DISABLE_ENV, "true")
    contents = [
        _LLMContent("tool_result", LONG_ARTIFACT),
        _LLMContent("tool_result", LONG_ARTIFACT),
    ]
    before = [c.content for c in contents]
    result = apply_artifact_dedup_canary(contents, salt=SALT, min_block_chars=MIN)
    assert result.mode == "off"
    assert [c.content for c in contents] == before


def test_shadow_measures_duplicates_without_mutating():
    contents = [
        _LLMContent("tool_result", LONG_ARTIFACT),
        _LLMContent("tool_result", LONG_ARTIFACT),
    ]
    before = [c.content for c in contents]
    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="shadow"
    )
    assert [c.content for c in contents] == before  # never mutated
    assert result.mode == "shadow"
    assert result.mutated is False
    assert result.blocks_replaced == 0
    assert result.chars_saved == 0
    # Advisory: one eligible duplicate group; later occurrence chars measured.
    assert result.candidate_group_count == 1
    assert result.candidate_chars == len(LONG_ARTIFACT)


# ---------------------------------------------------------------------------
# (1) canary keeps the first full body, replaces later exact duplicates
# ---------------------------------------------------------------------------


def test_canary_keeps_first_full_and_replaces_later_duplicate():
    contents = [
        _LLMContent("tool_result", LONG_ARTIFACT),
        _LLMContent("tool_result", LONG_ARTIFACT),
    ]
    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    assert contents[0].content == LONG_ARTIFACT  # first kept verbatim
    assert contents[1].content == _ref(LONG_ARTIFACT)  # later replaced
    assert result.mode == "canary"
    assert result.mutated is True
    assert result.artifact_dedup_class == ARTIFACT_DEDUP_CLASS
    assert result.blocks_replaced == 1
    assert result.chars_saved == len(LONG_ARTIFACT) - _ref_len()


def test_canary_three_occurrences_keeps_first_replaces_two():
    contents = [
        _LLMContent("tool_result", LONG_ARTIFACT),
        _LLMContent("assistant_context", LONG_ARTIFACT),
        _LLMContent("tool_result", LONG_ARTIFACT),
    ]
    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    assert contents[0].content == LONG_ARTIFACT  # canonical kept
    assert contents[1].content == _ref(LONG_ARTIFACT)  # later dup replaced
    assert contents[2].content == _ref(LONG_ARTIFACT)
    assert result.blocks_replaced == 2
    assert result.chars_saved == 2 * (len(LONG_ARTIFACT) - _ref_len())


def test_canary_dedups_across_artifact_types_provenance_canonical_is_first():
    # tool_result first, assistant_context second: the duplicate spans both
    # artifact types. The first (tool_result) is canonical; the reference left in
    # the assistant_context records that canonical provenance.
    contents = [
        _LLMContent("tool_result", LONG_ARTIFACT),
        _LLMContent("assistant_context", LONG_ARTIFACT),
    ]
    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    assert contents[0].content == LONG_ARTIFACT
    # Reference records the canonical (first) provenance == tool_result.
    assert contents[1].content == _ref(LONG_ARTIFACT, canonical_type="tool_result")
    assert "tool_result" in contents[1].content
    assert "assistant_context" not in contents[1].content
    assert result.blocks_replaced == 1


def test_segment_fenced_blocks_round_trips_and_marks_closed_fences():
    segments = _segment_fenced_blocks(FENCED_PARENT_ARTIFACT)
    assert "".join(text for _kind, text in segments) == FENCED_PARENT_ARTIFACT
    assert [kind for kind, _text in segments].count("fence") == 2


def test_canary_replaces_later_exact_duplicate_fenced_block_inside_artifact_body():
    contents = [_LLMContent("assistant_context", FENCED_PARENT_ARTIFACT)]

    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )

    assert contents[0].content.count(LONG_FENCE_BLOCK) == 1
    expected_ref = _block_ref(LONG_FENCE_BLOCK, canonical_type="assistant_context#block")
    assert expected_ref in contents[0].content
    assert "Short prose between artifacts must survive byte-identical." in contents[0].content
    assert result.blocks_replaced == 1
    assert result.chars_saved == len(LONG_FENCE_BLOCK) - len(expected_ref)
    assert dangling_artifact_references(contents, salt=SALT) == []


def test_canary_replaces_duplicate_fenced_block_across_artifact_types():
    first = f"tool output wrapper\n{LONG_FENCE_BLOCK}\nend"
    second = f"assistant rollup wrapper\n{LONG_FENCE_BLOCK}\nend"
    contents = [
        _LLMContent("tool_result", first),
        _LLMContent("assistant_context", second),
    ]

    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )

    assert contents[0].content == first
    assert LONG_FENCE_BLOCK not in contents[1].content
    assert _block_ref(LONG_FENCE_BLOCK, canonical_type="tool_result#block") in contents[1].content
    assert result.blocks_replaced == 1


def test_whole_body_canonical_is_not_registered_before_internal_block_rewrite():
    contents = [
        _LLMContent("assistant_context", FENCED_PARENT_ARTIFACT),
        _LLMContent("assistant_context", FENCED_PARENT_ARTIFACT),
    ]

    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )

    assert result.blocks_replaced >= 2
    # A later reference must never point at the pre-mutation whole-body hash after
    # the first body was internally rewritten; every emitted reference resolves
    # to an earlier full fenced block/body still present in the payload.
    assert dangling_artifact_references(contents, salt=SALT) == []


def test_unterminated_fence_is_treated_as_prose_and_not_mutated():
    body = "prefix\n```log\n" + ("unterminated synthetic artifact line\n" * 8)
    contents = [_LLMContent("tool_result", body + body)]
    before = contents[0].content

    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )

    assert contents[0].content == before
    assert result.blocks_replaced == 0


# ---------------------------------------------------------------------------
# Level 2: declared source-span provenance (metadata-driven, not discovery)
# ---------------------------------------------------------------------------


def test_source_span_canary_replaces_declared_parent_span_only():
    contents = [
        _LLMContent("tool_result", SOURCE_SPAN_TOOL),
        _LLMContent("assistant_context", SOURCE_SPAN_PARENT),
    ]
    link = _source_span_link()

    result = apply_artifact_dedup_canary(
        contents,
        salt=SALT,
        min_block_chars=MIN,
        mode="canary",
        span_links=[link],
    )

    expected_ref = _span_ref(SOURCE_SPAN_BLOCK)
    assert contents[0].content == SOURCE_SPAN_TOOL
    assert contents[1].content == SOURCE_SPAN_PARENT.replace(SOURCE_SPAN_BLOCK, expected_ref)
    assert result.span_blocks_replaced == 1
    assert result.span_chars_saved == len(SOURCE_SPAN_BLOCK) - len(expected_ref)
    assert result.blocks_replaced == 1
    assert result.chars_saved == result.span_chars_saved
    assert dangling_artifact_references(contents, salt=SALT, span_links=[link]) == []


def test_source_span_shadow_measures_without_mutating():
    contents = [
        _LLMContent("tool_result", SOURCE_SPAN_TOOL),
        _LLMContent("assistant_context", SOURCE_SPAN_PARENT),
    ]
    before = [c.content for c in contents]

    result = apply_artifact_dedup_canary(
        contents,
        salt=SALT,
        min_block_chars=MIN,
        mode="shadow",
        span_links=[_source_span_link()],
    )

    assert [c.content for c in contents] == before
    assert result.span_blocks_replaced == 0
    assert result.span_candidate_count == 1
    assert result.span_candidate_chars == len(SOURCE_SPAN_BLOCK)


def test_source_span_mismatch_or_forward_link_is_not_mutated():
    mismatch_parent = SOURCE_SPAN_PARENT.replace("alpha", "ALPHA", 1)
    contents = [
        _LLMContent("tool_result", SOURCE_SPAN_TOOL),
        _LLMContent("assistant_context", mismatch_parent),
    ]
    result = apply_artifact_dedup_canary(
        contents,
        salt=SALT,
        min_block_chars=MIN,
        mode="canary",
        span_links=[_source_span_link()],
    )
    assert contents[1].content == mismatch_parent
    assert result.span_blocks_replaced == 0

    forward = ArtifactSpanLink(1, 0, len(SOURCE_SPAN_BLOCK), 0, 0, len(SOURCE_SPAN_BLOCK))
    before = [c.content for c in contents]
    result = apply_artifact_dedup_canary(
        contents,
        salt=SALT,
        min_block_chars=MIN,
        mode="canary",
        span_links=[forward],
    )
    assert [c.content for c in contents] == before
    assert result.span_blocks_replaced == 0


def test_source_span_rejects_protected_or_inline_scope():
    inline_parent = SOURCE_SPAN_PARENT.replace("\n" + SOURCE_SPAN_BLOCK + "\n", SOURCE_SPAN_BLOCK)
    contents = [
        _LLMContent("tool_result", SOURCE_SPAN_TOOL),
        _LLMContent("assistant_context", inline_parent),
        _LLMContent("user_prompt", SOURCE_SPAN_PARENT),
    ]
    inline_start = inline_parent.index(SOURCE_SPAN_BLOCK)
    links = [
        ArtifactSpanLink(0, SOURCE_SPAN_TOOL.index(SOURCE_SPAN_BLOCK), SOURCE_SPAN_TOOL.index(SOURCE_SPAN_BLOCK) + len(SOURCE_SPAN_BLOCK), 1, inline_start, inline_start + len(SOURCE_SPAN_BLOCK)),
        ArtifactSpanLink(0, SOURCE_SPAN_TOOL.index(SOURCE_SPAN_BLOCK), SOURCE_SPAN_TOOL.index(SOURCE_SPAN_BLOCK) + len(SOURCE_SPAN_BLOCK), 2, SOURCE_SPAN_PARENT.index(SOURCE_SPAN_BLOCK), SOURCE_SPAN_PARENT.index(SOURCE_SPAN_BLOCK) + len(SOURCE_SPAN_BLOCK)),
    ]
    before = [c.content for c in contents]

    result = apply_artifact_dedup_canary(
        contents,
        salt=SALT,
        min_block_chars=MIN,
        mode="canary",
        span_links=links,
    )

    assert [c.content for c in contents] == before
    assert result.span_blocks_replaced == 0


def test_canary_reference_carries_no_raw_artifact_body():
    contents = [
        _LLMContent("tool_result", LONG_ARTIFACT),
        _LLMContent("tool_result", LONG_ARTIFACT),
    ]
    apply_artifact_dedup_canary(contents, salt=SALT, min_block_chars=MIN, mode="canary")
    ref_line = contents[1].content
    # Only a low-cardinality provenance enum + salted hash, never the body.
    assert "tool_result" in ref_line
    assert _salted_hash(LONG_ARTIFACT, SALT) in ref_line
    assert LONG_ARTIFACT not in ref_line
    assert len(ref_line) < len(LONG_ARTIFACT)  # always shrinks


def test_canary_never_grows_payload_for_short_duplicate():
    assert len(SHORT_ARTIFACT) >= MIN
    assert len(SHORT_ARTIFACT) < _ref_len()  # reference would be longer
    contents = [
        _LLMContent("tool_result", SHORT_ARTIFACT),
        _LLMContent("tool_result", SHORT_ARTIFACT),
    ]
    before = [c.content for c in contents]
    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    assert [c.content for c in contents] == before  # left alone (would grow)
    assert result.blocks_replaced == 0
    assert result.chars_saved == 0


# ---------------------------------------------------------------------------
# (2) only tool_result / assistant_context are mutable; others protected
# ---------------------------------------------------------------------------


def test_mutable_set_is_exactly_the_two_artifact_types():
    assert set(MUTABLE_ARTIFACT_BLOCK_TYPES) == {"tool_result", "assistant_context"}


def test_canary_leaves_non_artifact_block_types_untouched():
    # Duplicate bodies in every PROTECTED block type must survive byte-identical
    # and must not even be scanned as artifact candidates.
    contents = [
        _LLMContent("system_prompt", SYS_BLOCK),
        _LLMContent("system_prompt", SYS_BLOCK),
        _LLMContent("user_prompt", LONG_ARTIFACT),
        _LLMContent("user_prompt", LONG_ARTIFACT),
        _LLMContent("skill_prompt", LONG_ARTIFACT),
        _LLMContent("skill_prompt", LONG_ARTIFACT),
        _LLMContent("unknown", LONG_ARTIFACT),
        _LLMContent("unknown", LONG_ARTIFACT),
    ]
    before = [c.content for c in contents]
    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    assert [c.content for c in contents] == before
    assert result.blocks_replaced == 0
    assert result.item_count == 0  # no artifact items present to scan


def test_canary_does_not_dedup_artifact_against_protected_duplicate():
    # The same body appears in a protected user_prompt AND a tool_result. The
    # artifact occurrence has no EARLIER artifact canonical, so nothing is
    # replaced (a reference may only point at an artifact canonical body).
    contents = [
        _LLMContent("user_prompt", LONG_ARTIFACT),
        _LLMContent("tool_result", LONG_ARTIFACT),
    ]
    before = [c.content for c in contents]
    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    assert [c.content for c in contents] == before
    assert result.blocks_replaced == 0


def test_assistant_context_is_a_mutable_artifact():
    contents = [
        _LLMContent("assistant_context", LONG_ARTIFACT),
        _LLMContent("assistant_context", LONG_ARTIFACT),
    ]
    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    assert contents[0].content == LONG_ARTIFACT
    assert contents[1].content == _ref(LONG_ARTIFACT, canonical_type="assistant_context")
    assert result.blocks_replaced == 1


# ---------------------------------------------------------------------------
# (3) reference resolution: must point at an earlier canonical full body
# ---------------------------------------------------------------------------


def test_canary_output_has_no_dangling_references():
    contents = [
        _LLMContent("tool_result", LONG_ARTIFACT),
        _LLMContent("assistant_context", LONG_ARTIFACT),
        _LLMContent("tool_result", LONG_ARTIFACT),
    ]
    apply_artifact_dedup_canary(contents, salt=SALT, min_block_chars=MIN, mode="canary")
    # Every reference the canary emitted resolves to an earlier canonical body.
    assert dangling_artifact_references(contents, salt=SALT) == []


def test_resolution_accepts_reference_after_its_canonical_body():
    contents = [
        _LLMContent("tool_result", LONG_ARTIFACT),  # canonical full body
        _LLMContent("tool_result", _ref(LONG_ARTIFACT)),  # resolves to index 0
    ]
    assert dangling_artifact_references(contents, salt=SALT) == []


def test_resolution_flags_reference_with_no_earlier_canonical():
    # A reference whose canonical body never appears earlier is dangling.
    contents = [
        _LLMContent("tool_result", _ref(LONG_ARTIFACT)),
    ]
    assert dangling_artifact_references(contents, salt=SALT) == [0]


def test_resolution_flags_reference_before_its_canonical_body():
    # The canonical body exists, but only AFTER the reference -> still dangling.
    contents = [
        _LLMContent("tool_result", _ref(LONG_ARTIFACT)),
        _LLMContent("tool_result", LONG_ARTIFACT),
    ]
    assert dangling_artifact_references(contents, salt=SALT) == [0]


# ---------------------------------------------------------------------------
# (4) telemetry: realized chars_saved only on actual mutation, no raw content
# ---------------------------------------------------------------------------


def test_telemetry_records_no_savings_when_off():
    contents = [
        _LLMContent("tool_result", LONG_ARTIFACT),
        _LLMContent("tool_result", LONG_ARTIFACT),
    ]
    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="off"
    )
    record = build_artifact_canary_telemetry_record(result)
    assert record["artifact_dedup_mode"] == "off"
    assert record["artifact_dedup_blocks_replaced"] == 0
    assert record["artifact_dedup_chars_saved"] == 0
    assert record["chars_saved"] == 0


def test_telemetry_records_no_savings_in_shadow():
    contents = [
        _LLMContent("tool_result", LONG_ARTIFACT),
        _LLMContent("tool_result", LONG_ARTIFACT),
    ]
    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="shadow"
    )
    record = build_artifact_canary_telemetry_record(result)
    assert record["artifact_dedup_mode"] == "shadow"
    # Shadow contributes nothing to the realized chars_saved total.
    assert record["artifact_dedup_chars_saved"] == 0
    assert record["chars_saved"] == 0


def test_telemetry_records_realized_savings_in_canary():
    contents = [
        _LLMContent("tool_result", LONG_ARTIFACT),
        _LLMContent("tool_result", LONG_ARTIFACT),
        _LLMContent("assistant_context", LONG_ARTIFACT),
    ]
    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    record = build_artifact_canary_telemetry_record(result)
    expected = 2 * (len(LONG_ARTIFACT) - _ref_len())
    assert record["artifact_dedup_mode"] == "canary"
    assert record["artifact_dedup_class"] == ARTIFACT_DEDUP_CLASS
    assert record["artifact_dedup_blocks_replaced"] == 2
    assert record["artifact_dedup_chars_saved"] == expected
    assert record["chars_saved"] == expected


def test_telemetry_is_metadata_only_no_artifact_text():
    contents = [
        _LLMContent("tool_result", LONG_ARTIFACT),
        _LLMContent("tool_result", LONG_ARTIFACT),
    ]
    result = apply_artifact_dedup_canary(
        contents, salt=SALT, min_block_chars=MIN, mode="canary"
    )
    record = build_artifact_canary_telemetry_record(result)
    blob = json.dumps(record)
    assert LONG_ARTIFACT not in blob
    assert set(record) == {
        "artifact_dedup_mode",
        "artifact_dedup_class",
        "artifact_dedup_blocks_replaced",
        "artifact_span_blocks_replaced",
        "artifact_span_chars_saved",
        "artifact_dedup_chars_saved",
        "chars_saved",
    }
    for key in (
        "artifact_dedup_blocks_replaced",
        "artifact_span_blocks_replaced",
        "artifact_span_chars_saved",
        "artifact_dedup_chars_saved",
        "chars_saved",
    ):
        assert isinstance(record[key], int)


# ---------------------------------------------------------------------------
# (3)+(4) runner/validation gate over synthetic artifact cases
# ---------------------------------------------------------------------------


def _artifact_cases() -> list[dict]:
    """Two synthetic cases, each with an exact-duplicate artifact body."""
    return [
        {
            "case_id": "syn-art-1",
            "source": "synthetic",
            "messages": [
                {"role": "system", "block_type": "system_prompt", "content": SYS_BLOCK},
                {"role": "tool", "block_type": "tool_result", "content": LONG_ARTIFACT},
                {
                    "role": "user",
                    "block_type": "user_prompt",
                    "content": "Synthetic user question referencing the artifact above.",
                },
                # exact duplicate of the earlier tool_result -> replaced by ref.
                {"role": "tool", "block_type": "tool_result", "content": LONG_ARTIFACT},
            ],
        },
        {
            "case_id": "syn-art-2",
            "source": "synthetic",
            "messages": [
                {
                    "role": "assistant",
                    "block_type": "assistant_context",
                    "content": LONG_ARTIFACT,
                },
                # cross-type duplicate -> canonical is the assistant_context above.
                {"role": "tool", "block_type": "tool_result", "content": LONG_ARTIFACT},
            ],
        },
    ]


def test_artifact_runner_passes_on_synthetic_duplicate_artifacts():
    report = run_artifact_validation(
        _artifact_cases(),
        baseline_mode="off",
        candidate_mode="canary",
        salt=SALT,
        min_block_chars=MIN,
        date="2026-06-15",
    )
    assert report.passed is True
    assert report.failed_cases == 0
    assert report.invariant_names == ARTIFACT_INVARIANT_NAMES
    assert report.total_blocks_replaced == 2  # one duplicate per case
    assert report.total_chars_saved > 0
    assert any(c.mutated for c in report.cases)
    # No tokenizer backend configured by default.
    assert report.tokenizer_status == "unavailable"
    assert report.total_actual_tokens_saved is None


def test_artifact_runner_shadow_passes_without_realized_savings():
    report = run_artifact_validation(
        _artifact_cases(),
        baseline_mode="off",
        candidate_mode="shadow",
        salt=SALT,
        min_block_chars=MIN,
        date="2026-06-15",
    )
    assert report.passed is True
    assert report.total_blocks_replaced == 0
    assert report.total_chars_saved == 0
    assert all(not c.mutated for c in report.cases)


def test_artifact_runner_report_is_privacy_safe():
    report = run_artifact_validation(
        _artifact_cases(),
        baseline_mode="off",
        candidate_mode="canary",
        salt=SALT,
        min_block_chars=MIN,
        date="2026-06-15",
    )
    report_dict = report_to_dict(report)
    raw_needles = [
        LONG_ARTIFACT,
        SYS_BLOCK,
        "Synthetic user question referencing the artifact above.",
    ]
    assert_report_privacy_safe(report_dict, raw_needles)
    blob = json.dumps(report_dict, ensure_ascii=False)
    md = render_markdown(report)
    for needle in raw_needles:
        assert needle not in blob
        assert needle not in md


def test_artifact_runner_fails_on_dangling_reference():
    # A candidate that replaces the FIRST (canonical) artifact body with a
    # reference leaves a dangling reference: nothing earlier resolves it.
    def bad_dangling(messages, *, mode, salt, min_block_chars):
        out = [dict(m) for m in messages]
        if mode == "bad":
            for m in out:
                if m["block_type"] in MUTABLE_ARTIFACT_BLOCK_TYPES:
                    m["content"] = _artifact_reference_string(
                        m["block_type"], _salted_hash("no-such-canonical", salt)
                    )
                    break
            result = ArtifactDedupCanaryResult(
                mode="canary",
                artifact_dedup_class=ARTIFACT_DEDUP_CLASS,
                mutated=True,
                item_count=2,
                candidate_group_count=1,
                candidate_chars=len(LONG_ARTIFACT),
                blocks_replaced=1,
                chars_saved=len(LONG_ARTIFACT) - _ref_len(),
            )
            return out, result
        result = ArtifactDedupCanaryResult(
            mode="off",
            artifact_dedup_class=ARTIFACT_DEDUP_CLASS,
            mutated=False,
            item_count=0,
            candidate_group_count=0,
            candidate_chars=0,
            blocks_replaced=0,
            chars_saved=0,
        )
        return out, result

    report = run_artifact_validation(
        _artifact_cases()[:1],
        baseline_mode="off",
        candidate_mode="bad",
        salt=SALT,
        min_block_chars=MIN,
        date="2026-06-15",
        optimize_fn=bad_dangling,
    )
    assert report.passed is False
    assert report.failed_cases == 1
    assert "artifact_reference_resolvable" in report.cases[0].failed_invariants


def test_artifact_runner_fails_on_protected_mutation():
    # A candidate that rewrites a protected (non-artifact) user_prompt must fail
    # the protected-content and mutation-scope invariants.
    def bad_protected(messages, *, mode, salt, min_block_chars):
        out = [dict(m) for m in messages]
        if mode == "bad":
            for m in out:
                if m["block_type"] == "user_prompt":
                    m["content"] = "[dropped]"
                    break
            result = ArtifactDedupCanaryResult(
                mode="canary",
                artifact_dedup_class=ARTIFACT_DEDUP_CLASS,
                mutated=True,
                item_count=2,
                candidate_group_count=0,
                candidate_chars=0,
                blocks_replaced=1,
                chars_saved=1,
            )
            return out, result
        result = ArtifactDedupCanaryResult(
            mode="off",
            artifact_dedup_class=ARTIFACT_DEDUP_CLASS,
            mutated=False,
            item_count=0,
            candidate_group_count=0,
            candidate_chars=0,
            blocks_replaced=0,
            chars_saved=0,
        )
        return out, result

    report = run_artifact_validation(
        _artifact_cases()[:1],
        baseline_mode="off",
        candidate_mode="bad",
        salt=SALT,
        min_block_chars=MIN,
        date="2026-06-15",
        optimize_fn=bad_protected,
    )
    assert report.passed is False
    failed = report.cases[0].failed_invariants
    assert "protected_content_preserved" in failed
    assert "artifact_mutation_scope_allowed" in failed


def test_reference_template_is_low_cardinality_placeholder_only():
    # The template carries only <type>/<hash> placeholders -- never content.
    assert "<type>" in ARTIFACT_DEDUP_CANARY_REFERENCE_TEMPLATE
    assert "<hash>" in ARTIFACT_DEDUP_CANARY_REFERENCE_TEMPLATE
