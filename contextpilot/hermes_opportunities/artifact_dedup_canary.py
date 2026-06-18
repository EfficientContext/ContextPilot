"""Default-OFF provenance-aware tool-artifact reuse canary.

This is the *second* runtime mutation path in ContextPilot and, like
:mod:`.prompt_dedup_canary`, it is default-OFF and narrowly scoped. Where the
prompt-dedup canary rewrites duplicate ``skill_prompt`` *lines*, this canary
dedups whole **artifact bodies** carried by ``tool_result`` and
``assistant_context`` items: it keeps the FIRST full artifact body verbatim and
replaces a later EXACT duplicate body -- regardless of which of the two mutable
artifact block types it appears in (provenance-aware) -- with a deterministic,
strictly shorter reference string that records the canonical body's provenance
(type + salted hash).

Risk gate (all conditions must hold before a single character is changed):

* Mode must be ``canary``. The mode is read from
  ``CONTEXTPILOT_ARTIFACT_DEDUP_MODE`` (``off`` | ``shadow`` | ``canary``) and
  defaults to ``off``. ``off`` and ``shadow`` never mutate the payload.
* The escape-hatch env ``CONTEXTPILOT_ARTIFACT_DEDUP_DISABLE`` (any truthy
  value) forces ``off`` regardless of the mode var -- an immediate kill switch.
* Only ``tool_result`` / ``assistant_context`` artifact bodies are mutable.
  ``system_prompt`` / ``user_prompt`` / ``skill_prompt`` / ``unknown`` (and any
  other non-artifact content) are protected and never scanned or rewritten.
* Only EXACT duplicate full bodies are eligible. The first occurrence (within
  the mutable artifact types) is the canonical body and is always kept verbatim;
  only later exact occurrences are replaced, and only when the deterministic
  reference string is strictly shorter than the body it replaces (never grows
  the payload).
* A reference may only point at an EARLIER canonical full body in the same
  payload; :func:`dangling_artifact_references` exposes a check used by the
  validation gate to reject a dangling reference.

The reference string carries only a low-cardinality artifact-type enum and a
salted body hash -- never raw artifact content. Telemetry is metadata-only:
mode/class enums and integer counters; no artifact text and no realized-savings
claim unless an actual mutation occurred.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

from .models import _LLMContent
from .privacy import _assert_no_forbidden_keys, _salted_hash

# Environment controls. ``off`` is the default and the safe state.
ARTIFACT_DEDUP_MODE_ENV = "CONTEXTPILOT_ARTIFACT_DEDUP_MODE"
ARTIFACT_DEDUP_DISABLE_ENV = "CONTEXTPILOT_ARTIFACT_DEDUP_DISABLE"
ARTIFACT_DEDUP_MODES = ("off", "shadow", "canary")
DEFAULT_ARTIFACT_DEDUP_MODE = "off"

# The only block types whose artifact bodies this canary may dedup. Both are
# mutable; the duplicate may span them (provenance-aware, cross-type).
MUTABLE_ARTIFACT_BLOCK_TYPES = ("tool_result", "assistant_context")

# The only duplicate class this canary acts on: an exact-duplicate full artifact
# body across the mutable artifact types.
ARTIFACT_DEDUP_CLASS = "same_payload_exact_artifact_body"

# Deterministic placeholder left in place of a later duplicate body. ``<type>``
# is the CANONICAL (first) body's provenance and ``<hash>`` its salted
# fingerprint -- both low-cardinality, never raw artifact content.
ARTIFACT_DEDUP_CANARY_REFERENCE_TEMPLATE = (
    "[ContextPilot artifact dedup: duplicate <type> artifact body omitted; "
    "ref=<type>:<hash>]"
)

# Fixed head of the reference string (everything before the first placeholder),
# used to recognize a reference line without re-rendering it.
_REF_HEAD = ARTIFACT_DEDUP_CANARY_REFERENCE_TEMPLATE.split("<type>", 1)[0]


@dataclass
class ArtifactDedupCanaryResult:
    """Metadata-only outcome of an artifact-dedup canary pass. No raw text, ever.

    ``chars_saved`` / ``blocks_replaced`` are REALIZED figures and are non-zero
    only when ``mode == 'canary'`` and an actual replacement occurred. The
    ``candidate_*`` fields are advisory (what a canary *would* replace) and are
    populated in ``shadow`` mode for visibility without mutating anything.
    """

    mode: str                      # off | shadow | canary
    artifact_dedup_class: str      # always ARTIFACT_DEDUP_CLASS
    mutated: bool                  # True only if a real replacement happened
    item_count: int                # mutable artifact items scanned
    candidate_group_count: int     # eligible exact-duplicate body groups
    candidate_chars: int           # advisory chars later occurrences occupy
    blocks_replaced: int           # REALIZED replacements (canary only)
    chars_saved: int               # REALIZED chars saved (canary only)
    notes: list[str] = field(default_factory=list)


def _truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() not in ("", "0", "false", "no", "off")


def resolve_artifact_dedup_mode(env: dict | None = None) -> str:
    """Resolve the active artifact-dedup mode, defaulting to the safe ``off``.

    Unknown values fall back to ``off``. The escape-hatch disable variable, when
    truthy, forces ``off`` regardless of the mode variable.
    """
    source = os.environ if env is None else env
    if _truthy(source.get(ARTIFACT_DEDUP_DISABLE_ENV)):
        return "off"
    raw = (
        source.get(ARTIFACT_DEDUP_MODE_ENV) or DEFAULT_ARTIFACT_DEDUP_MODE
    ).strip().lower()
    return raw if raw in ARTIFACT_DEDUP_MODES else DEFAULT_ARTIFACT_DEDUP_MODE


def _artifact_reference_string(canonical_type: str, body_hash: str) -> str:
    """Render the reference that points at a canonical artifact body.

    Carries only the canonical provenance enum and the salted hash -- never the
    artifact body itself.
    """
    return ARTIFACT_DEDUP_CANARY_REFERENCE_TEMPLATE.replace(
        "<type>", canonical_type
    ).replace("<hash>", body_hash)


def _parse_artifact_reference(line: str) -> str | None:
    """Return the salted hash a reference line encodes, or ``None`` if not one."""
    if not (line.startswith(_REF_HEAD) and "ref=" in line and line.endswith("]")):
        return None
    after = line.rsplit("ref=", 1)[1][:-1]  # strip the trailing ']'
    _type, sep, body_hash = after.partition(":")
    if not sep or not body_hash:
        return None
    return body_hash


def _segment_fenced_blocks(body: str) -> list[tuple[str, str]]:
    """Split ``body`` into reversible prose/fence segments.

    Only closed triple-backtick fences are marked as ``"fence"``. Unterminated
    fences are deliberately treated as prose so the canary never guesses a block
    boundary. Concatenating the segment text always reproduces ``body`` exactly.
    """
    segments: list[tuple[str, str]] = []
    pos = 0
    n = len(body)
    while pos < n:
        start = body.find("```", pos)
        if start == -1:
            if pos < n:
                segments.append(("prose", body[pos:]))
            break
        close = body.find("```", start + 3)
        if close == -1:
            if pos < n:
                segments.append(("prose", body[pos:]))
            break
        if start > pos:
            segments.append(("prose", body[pos:start]))
        end = close + 3
        segments.append(("fence", body[start:end]))
        pos = end
    return segments


def _scan_fenced_block_candidates(
    contents: list[_LLMContent], *, salt: str, min_block_chars: int
) -> tuple[int, int]:
    """Advisory duplicate count for exact fenced sub-artifacts."""
    agg: dict[str, dict] = {}
    for item in contents:
        if item.block_type not in MUTABLE_ARTIFACT_BLOCK_TYPES:
            continue
        # Whole-body references are not canonical sources for sub-blocks.
        if _parse_artifact_reference(item.content) is not None:
            continue
        for kind, text in _segment_fenced_blocks(item.content):
            if kind != "fence" or len(text) < min_block_chars:
                continue
            if _parse_artifact_reference(text) is not None:
                continue
            h = _salted_hash(text, salt)
            entry = agg.get(h)
            if entry is None:
                agg[h] = {"canonical_type": f"{item.block_type}#block", "char_length": len(text), "occ": 1}
            else:
                entry["occ"] += 1
    return _eligible_groups(agg)


def _scan_artifacts(
    contents: list[_LLMContent], *, salt: str, min_block_chars: int
) -> tuple[dict[str, dict], int]:
    """Fingerprint mutable artifact bodies in order.

    Returns ``(agg, item_count)`` where ``agg`` maps a body hash to
    ``{canonical_type, char_length, occ}`` (``canonical_type`` is the FIRST
    occurrence's provenance) and ``item_count`` is the number of mutable
    artifact items seen.
    """
    agg: dict[str, dict] = {}
    item_count = 0
    for item in contents:
        if item.block_type not in MUTABLE_ARTIFACT_BLOCK_TYPES:
            continue
        item_count += 1
        body = item.content
        if len(body) < min_block_chars:
            continue
        # A reference left by an earlier pass is not itself a canonical body.
        if _parse_artifact_reference(body) is not None:
            continue
        h = _salted_hash(body, salt)
        entry = agg.get(h)
        if entry is None:
            agg[h] = {
                "canonical_type": item.block_type,
                "char_length": len(body),
                "occ": 1,
            }
        else:
            entry["occ"] += 1
    return agg, item_count


def _eligible_groups(agg: dict[str, dict]) -> tuple[int, int]:
    """Advisory measurement of duplicate body groups that would actually shrink.

    Returns ``(candidate_group_count, candidate_chars)`` where ``candidate_chars``
    is the chars the later (replaceable) occurrences currently occupy.
    """
    group_count = 0
    candidate_chars = 0
    for h, entry in agg.items():
        if entry["occ"] < 2:
            continue  # not a duplicate -> nothing to replace
        ref = _artifact_reference_string(entry["canonical_type"], h)
        if len(ref) >= entry["char_length"]:
            continue  # replacement would grow the payload -> skip
        group_count += 1
        candidate_chars += (entry["occ"] - 1) * entry["char_length"]
    return group_count, candidate_chars


def apply_artifact_dedup_canary(
    contents: Iterable[_LLMContent],
    *,
    salt: str,
    min_block_chars: int,
    mode: str | None = None,
    env: dict | None = None,
) -> ArtifactDedupCanaryResult:
    """Run the artifact-dedup canary over LLM-bound content.

    ``contents`` are the in-memory ``_LLMContent`` items bound for the LLM. In
    ``canary`` mode this MUTATES the ``content`` of eligible mutable artifact
    items in place (keeping the first canonical body, replacing later exact
    duplicates with a deterministic, strictly shorter reference). In ``off`` and
    ``shadow`` modes nothing is mutated.

    ``mode`` overrides the resolved environment mode (used by tests); otherwise
    the mode comes from :func:`resolve_artifact_dedup_mode`.
    """
    items = list(contents)
    resolved = mode if mode is not None else resolve_artifact_dedup_mode(env)
    if resolved not in ARTIFACT_DEDUP_MODES:
        resolved = DEFAULT_ARTIFACT_DEDUP_MODE

    if resolved == "off":
        # Safe default: no scan, no candidates, no savings.
        return ArtifactDedupCanaryResult(
            mode="off",
            artifact_dedup_class=ARTIFACT_DEDUP_CLASS,
            mutated=False,
            item_count=0,
            candidate_group_count=0,
            candidate_chars=0,
            blocks_replaced=0,
            chars_saved=0,
            notes=["artifact-dedup canary off (default): payload unchanged"],
        )

    agg, item_count = _scan_artifacts(items, salt=salt, min_block_chars=min_block_chars)
    candidate_group_count, candidate_chars = _eligible_groups(agg)
    block_group_count, block_candidate_chars = _scan_fenced_block_candidates(
        items, salt=salt, min_block_chars=min_block_chars
    )
    candidate_group_count += block_group_count
    candidate_chars += block_candidate_chars

    if resolved == "shadow":
        # Measure what a canary would replace, but never touch the payload.
        return ArtifactDedupCanaryResult(
            mode="shadow",
            artifact_dedup_class=ARTIFACT_DEDUP_CLASS,
            mutated=False,
            item_count=item_count,
            candidate_group_count=candidate_group_count,
            candidate_chars=candidate_chars,
            blocks_replaced=0,
            chars_saved=0,
            notes=["artifact-dedup canary shadow: candidates measured, payload unchanged"],
        )

    # --- canary: the ONLY branch that mutates LLM-bound payload ---------------
    blocks_replaced = 0
    chars_saved = 0
    # hash -> canonical provenance type of the first (kept) occurrence.
    canonical: dict[str, str] = {}
    # hash -> canonical provenance type for exact fenced sub-artifacts.
    block_canonical: dict[str, str] = {}
    for item in items:
        if item.block_type not in MUTABLE_ARTIFACT_BLOCK_TYPES:
            continue  # protected content is never touched
        body = item.content
        if len(body) < min_block_chars:
            continue
        if _parse_artifact_reference(body) is not None:
            continue
        h = _salted_hash(body, salt)
        already_has_whole_canonical = h in canonical
        if already_has_whole_canonical:
            # Later exact duplicate whole body: reference the EARLIER canonical
            # body's provenance and do not also scan sub-blocks (no double count).
            ref = _artifact_reference_string(canonical[h], h)
            if len(ref) < len(body):  # only when it actually shrinks the payload
                item.content = ref
                blocks_replaced += 1
                chars_saved += len(body) - len(ref)
                continue

        # If the whole body is not replaced, opportunistically dedup exact
        # duplicate fenced sub-artifacts within/across mutable artifact bodies.
        segments = _segment_fenced_blocks(body)
        if not any(kind == "fence" for kind, _text in segments):
            if not already_has_whole_canonical:
                canonical[h] = item.block_type  # keep the first canonical body verbatim
            continue
        new_segments: list[str] = []
        changed = False
        for kind, text in segments:
            if kind != "fence" or len(text) < min_block_chars:
                new_segments.append(text)
                continue
            if _parse_artifact_reference(text) is not None:
                new_segments.append(text)
                continue
            bh = _salted_hash(text, salt)
            if bh not in block_canonical:
                block_canonical[bh] = f"{item.block_type}#block"
                new_segments.append(text)
                continue
            ref = _artifact_reference_string(block_canonical[bh], bh)
            if len(ref) < len(text):
                new_segments.append(ref)
                blocks_replaced += 1
                chars_saved += len(text) - len(ref)
                changed = True
            else:
                new_segments.append(text)
        if changed:
            item.content = "".join(new_segments)
            # Register only the post-mutation whole body as canonical. Registering
            # the original pre-mutation hash would let a later whole-body
            # reference point to a body no longer present in the payload.
            canonical[_salted_hash(item.content, salt)] = item.block_type
        elif not already_has_whole_canonical:
            canonical[h] = item.block_type  # keep the first canonical body verbatim

    return ArtifactDedupCanaryResult(
        mode="canary",
        artifact_dedup_class=ARTIFACT_DEDUP_CLASS,
        mutated=blocks_replaced > 0,
        item_count=item_count,
        candidate_group_count=candidate_group_count,
        candidate_chars=candidate_chars,
        blocks_replaced=blocks_replaced,
        chars_saved=chars_saved,
        notes=["artifact-dedup canary active: exact duplicate artifact bodies only"],
    )


def dangling_artifact_references(
    contents: Iterable[_LLMContent], *, salt: str
) -> list[int]:
    """Return indices of artifact references that do not resolve to an earlier body.

    A reference is valid only if an EARLIER mutable artifact item carries the
    full canonical body whose salted hash matches the reference. A reference with
    no such earlier canonical body (or one that only appears later) is dangling.
    """
    seen_full: set[str] = set()  # hashes of earlier full canonical artifact bodies/blocks
    dangling: list[int] = []
    for idx, item in enumerate(contents):
        body = item.content
        ref_hash = _parse_artifact_reference(body)
        if ref_hash is not None:
            if ref_hash not in seen_full:
                dangling.append(idx)
            continue
        if item.block_type not in MUTABLE_ARTIFACT_BLOCK_TYPES:
            continue

        # Whole artifact body can satisfy whole-body references.
        seen_full.add(_salted_hash(body, salt))

        # Within a body, ordering matters: an earlier fenced block can satisfy a
        # later reference segment in the same body, but a later block cannot.
        for kind, text in _segment_fenced_blocks(body):
            if kind != "fence":
                # References may also appear as standalone prose lines after a
                # fenced block replacement. Embedded prose around the line stays
                # protected; only exact standalone reference lines are accepted.
                for line in text.splitlines():
                    seg_ref = _parse_artifact_reference(line.strip())
                    if seg_ref is not None and seg_ref not in seen_full:
                        dangling.append(idx)
                continue
            seg_ref = _parse_artifact_reference(text)
            if seg_ref is not None:
                if seg_ref not in seen_full:
                    dangling.append(idx)
                continue
            seen_full.add(_salted_hash(text, salt))
    return dangling


def build_artifact_canary_telemetry_record(result: ArtifactDedupCanaryResult) -> dict:
    """Build a metadata-only telemetry record for an artifact-dedup canary pass.

    The aggregate ``chars_saved`` counter gains the artifact-dedup contribution
    ONLY when a real mutation occurred (canary). ``off``/``shadow`` contribute 0
    to the total while still reporting the separated ``artifact_dedup_*`` fields.
    Contains only mode/class enums and integer counters -- never artifact text.
    """
    realized = result.chars_saved if result.mutated else 0
    record = {
        "artifact_dedup_mode": result.mode,
        "artifact_dedup_class": result.artifact_dedup_class,
        "artifact_dedup_blocks_replaced": result.blocks_replaced if result.mutated else 0,
        # Separated field: always present, mirrors the realized artifact-dedup save.
        "artifact_dedup_chars_saved": realized,
        # Aggregate total: includes artifact dedup only when a mutation occurred.
        "chars_saved": realized,
    }
    _assert_no_forbidden_keys(record)
    return record
