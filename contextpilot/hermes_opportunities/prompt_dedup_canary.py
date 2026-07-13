"""Default-OFF prompt-dedup canary (the only runtime prompt mutation path).

Everything else in this package is measurement/shadow/simulation only. This
module is the single, narrowly-scoped place where ContextPilot may *actually*
replace prompt text bound for the LLM -- and only when an operator has opted in
via an environment variable, and only for the lowest-risk duplicate class.

Risk gate (all conditions must hold before a single character is changed):

* Mode must be ``canary``. The mode is read from
  ``CONTEXTPILOT_PROMPT_DEDUP_MODE`` (``off`` | ``shadow`` | ``canary``) and
  defaults to ``off``. ``off`` and ``shadow`` never mutate the payload.
* The escape-hatch env ``CONTEXTPILOT_PROMPT_DEDUP_DISABLE`` (any truthy value)
  forces ``off`` regardless of the mode var -- an immediate kill switch.
* Only the ``same_type_skill_prompt_only`` class is eligible: an EXACT duplicate
  block whose every occurrence is inside ``skill_prompt`` content. Duplicates
  confined to ``system_prompt`` and cross-type ``system_prompt``/``skill_prompt``
  blocks are NEVER replaced.
* Only ``skill_prompt`` items are ever rewritten. ``system_prompt``,
  ``user_prompt``, ``assistant_context`` and ``tool_result`` content is never
  touched.
* The first occurrence of each duplicate is kept verbatim; only later exact
  occurrences are replaced, and only when the deterministic reference string is
  strictly shorter than the line it replaces (never grows the payload).
* Any block matching the safety denylist (instruction / safety / security /
  tool / auth / secret / must / never / always / required / ...) is left
  unchanged even in canary mode.

The reference string carries only a low-cardinality prompt-type enum and a
salted block hash -- never raw prompt content. Telemetry is metadata-only:
mode/class enums and integer counters; no prompt text and no realized-savings
claim unless an actual mutation occurred.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

from .models import PROMPT_DUPLICATE_BLOCK_TYPES, _LLMContent
from .privacy import _assert_no_forbidden_keys, _salted_hash

# Environment controls. ``off`` is the default and the safe state.
PROMPT_DEDUP_MODE_ENV = "CONTEXTPILOT_PROMPT_DEDUP_MODE"
PROMPT_DEDUP_DISABLE_ENV = "CONTEXTPILOT_PROMPT_DEDUP_DISABLE"
PROMPT_DEDUP_MODES = ("off", "shadow", "canary")
DEFAULT_PROMPT_DEDUP_MODE = "off"

# The only duplicate class this canary will ever act on.
CANARY_DEDUP_CLASS = "same_type_skill_prompt_only"

# Deterministic placeholder left in place of a later duplicate occurrence. Unlike
# the A/B simulation template ("... omitted in simulation ..."), this string is
# really emitted into the payload, so it is labelled as a real ContextPilot
# replacement. ``<type>`` / ``<hash>`` are low-cardinality only.
PROMPT_DEDUP_CANARY_REFERENCE_TEMPLATE = (
    "[ContextPilot dedup: duplicate skill_prompt block omitted; ref=<type>:<hash>]"
)

# Safety denylist. If any of these case-insensitive substrings appears in a
# duplicate block, the block is left untouched even in canary mode. The list is
# deliberately broad: it is always safe to skip a replacement, never safe to
# silently rewrite a hard instruction / safety / security / auth / secret line.
SAFETY_DENYLIST = (
    "instruction",
    "instructions",
    "safety",
    "security",
    "secure",
    "tool",
    "auth",
    "authenticate",
    "authentication",
    "authorization",
    "secret",
    "credential",
    "password",
    "api key",
    "token",
    "must",
    "never",
    "always",
    "required",
    "require",
    "do not",
    "don't",
    "important",
    "critical",
    "mandatory",
    "forbidden",
    "permission",
    "sensitive",
    "confidential",
    "policy",
    "verify",
)


@dataclass
class PromptDedupCanaryResult:
    """Metadata-only outcome of a canary pass. No raw prompt text, ever.

    ``chars_saved`` / ``blocks_replaced`` are REALIZED figures and are non-zero
    only when ``mode == 'canary'`` and an actual replacement occurred. The
    ``candidate_*`` fields are advisory (what a canary *would* replace) and are
    populated in ``shadow`` mode for visibility without mutating anything.
    """

    mode: str                      # off | shadow | canary
    prompt_dedup_class: str        # always CANARY_DEDUP_CLASS
    mutated: bool                  # True only if a real replacement happened
    item_count: int                # system/skill prompt items scanned
    skill_item_count: int          # skill_prompt items among them
    candidate_block_count: int     # eligible skill-only duplicate groups
    candidate_chars: int           # advisory chars later occurrences occupy
    blocks_replaced: int           # REALIZED replacements (canary only)
    chars_saved: int               # REALIZED chars saved (canary only)
    denylisted_block_count: int    # skill-only duplicate groups skipped by denylist
    notes: list[str] = field(default_factory=list)


def _truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() not in ("", "0", "false", "no", "off")


def resolve_prompt_dedup_mode(env: dict | None = None) -> str:
    """Resolve the active prompt-dedup mode, defaulting to the safe ``off``.

    Unknown values fall back to ``off``. The escape-hatch disable variable, when
    truthy, forces ``off`` regardless of the mode variable.
    """
    source = os.environ if env is None else env
    if _truthy(source.get(PROMPT_DEDUP_DISABLE_ENV)):
        return "off"
    raw = (source.get(PROMPT_DEDUP_MODE_ENV) or DEFAULT_PROMPT_DEDUP_MODE).strip().lower()
    return raw if raw in PROMPT_DEDUP_MODES else DEFAULT_PROMPT_DEDUP_MODE


def _is_denied(block: str) -> bool:
    """Conservative safety gate: any denylist substring blocks replacement."""
    low = block.lower()
    return any(keyword in low for keyword in SAFETY_DENYLIST)


def _reference_string(canonical_type: str, block_hash: str) -> str:
    return PROMPT_DEDUP_CANARY_REFERENCE_TEMPLATE.replace(
        "<type>", canonical_type
    ).replace("<hash>", block_hash)


def _build_eligibility(
    contents: list[_LLMContent], *, salt: str, min_block_chars: int
) -> tuple[dict[str, str], int, int, int]:
    """Fingerprint system/skill blocks and return the canary-eligible hashes.

    Returns ``(eligible, candidate_chars, denylisted, item_count)`` where
    ``eligible`` maps a block hash to its reference string. A hash is eligible
    only when it is an EXACT duplicate (occurs 2+ times), every occurrence is in
    ``skill_prompt`` content (same_type_skill_prompt_only), and the block does
    not match the safety denylist.
    """
    # hash -> {char_length, types: {block_type: occ}, denied}
    agg: dict[str, dict] = {}
    item_count = 0
    for item in contents:
        bt = item.block_type
        if bt not in PROMPT_DUPLICATE_BLOCK_TYPES:
            continue
        item_count += 1
        for raw_line in item.content.split("\n"):
            block = raw_line.strip()
            if len(block) < min_block_chars:
                continue
            h = _salted_hash(block, salt)
            entry = agg.get(h)
            if entry is None:
                agg[h] = {
                    "char_length": len(block),
                    "types": {bt: 1},
                    "denied": _is_denied(block),
                }
            else:
                entry["types"][bt] = entry["types"].get(bt, 0) + 1

    eligible: dict[str, str] = {}
    candidate_chars = 0
    denylisted = 0
    for h, entry in agg.items():
        types = entry["types"]
        occ = sum(types.values())
        if occ < 2:
            continue  # not a duplicate -> nothing to replace
        # same_type_skill_prompt_only: every occurrence is a skill prompt block.
        if set(types) != {"skill_prompt"}:
            continue
        if entry["denied"]:
            denylisted += 1
            continue
        eligible[h] = _reference_string("skill_prompt", h)
        # Advisory: chars the later (replaceable) occurrences currently occupy.
        candidate_chars += (occ - 1) * entry["char_length"]
    return eligible, candidate_chars, denylisted, item_count


def apply_prompt_dedup_canary(
    contents: Iterable[_LLMContent],
    *,
    salt: str,
    min_block_chars: int,
    mode: str | None = None,
    env: dict | None = None,
) -> PromptDedupCanaryResult:
    """Run the prompt-dedup canary over LLM-bound content.

    ``contents`` are the in-memory ``_LLMContent`` items bound for the LLM. In
    ``canary`` mode this MUTATES the ``content`` of eligible ``skill_prompt``
    items in place (keeping the first occurrence, replacing later exact
    duplicates with a deterministic reference string). In ``off`` and ``shadow``
    modes nothing is mutated.

    ``mode`` overrides the resolved environment mode (used by tests); otherwise
    the mode comes from :func:`resolve_prompt_dedup_mode`.
    """
    items = list(contents)
    resolved = mode if mode is not None else resolve_prompt_dedup_mode(env)
    if resolved not in PROMPT_DEDUP_MODES:
        resolved = DEFAULT_PROMPT_DEDUP_MODE

    skill_item_count = sum(1 for it in items if it.block_type == "skill_prompt")

    if resolved == "off":
        # Safe default: no scan, no candidates, no savings.
        return PromptDedupCanaryResult(
            mode="off",
            prompt_dedup_class=CANARY_DEDUP_CLASS,
            mutated=False,
            item_count=0,
            skill_item_count=skill_item_count,
            candidate_block_count=0,
            candidate_chars=0,
            blocks_replaced=0,
            chars_saved=0,
            denylisted_block_count=0,
            notes=["prompt-dedup canary off (default): payload unchanged"],
        )

    eligible, candidate_chars, denylisted, item_count = _build_eligibility(
        items, salt=salt, min_block_chars=min_block_chars
    )

    if resolved == "shadow":
        # Measure what a canary would replace, but never touch the payload.
        return PromptDedupCanaryResult(
            mode="shadow",
            prompt_dedup_class=CANARY_DEDUP_CLASS,
            mutated=False,
            item_count=item_count,
            skill_item_count=skill_item_count,
            candidate_block_count=len(eligible),
            candidate_chars=candidate_chars,
            blocks_replaced=0,
            chars_saved=0,
            denylisted_block_count=denylisted,
            notes=["prompt-dedup canary shadow: candidates measured, payload unchanged"],
        )

    # --- canary: the ONLY branch that mutates LLM-bound payload ---------------
    blocks_replaced = 0
    chars_saved = 0
    consumed: set[str] = set()  # hashes whose first (kept) occurrence was seen
    for item in items:
        if item.block_type != "skill_prompt":
            continue  # never touch system/user/assistant/tool content
        if not eligible:
            break
        new_lines: list[str] = []
        changed = False
        for raw_line in item.content.split("\n"):
            block = raw_line.strip()
            if len(block) < min_block_chars:
                new_lines.append(raw_line)
                continue
            h = _salted_hash(block, salt)
            ref = eligible.get(h)
            if ref is None:
                new_lines.append(raw_line)
                continue
            if h not in consumed:
                consumed.add(h)  # keep the first occurrence verbatim
                new_lines.append(raw_line)
                continue
            # Later exact duplicate: replace only when it actually shrinks the line.
            if len(ref) < len(raw_line):
                new_lines.append(ref)
                blocks_replaced += 1
                chars_saved += len(raw_line) - len(ref)
                changed = True
            else:
                new_lines.append(raw_line)
        if changed:
            item.content = "\n".join(new_lines)

    notes = ["prompt-dedup canary active: same_type_skill_prompt_only duplicates only"]
    if denylisted:
        notes.append(f"{denylisted} skill-only duplicate group(s) skipped by safety denylist")
    return PromptDedupCanaryResult(
        mode="canary",
        prompt_dedup_class=CANARY_DEDUP_CLASS,
        mutated=blocks_replaced > 0,
        item_count=item_count,
        skill_item_count=skill_item_count,
        candidate_block_count=len(eligible),
        candidate_chars=candidate_chars,
        blocks_replaced=blocks_replaced,
        chars_saved=chars_saved,
        denylisted_block_count=denylisted,
        notes=notes,
    )


def build_canary_telemetry_record(result: PromptDedupCanaryResult) -> dict:
    """Build a metadata-only telemetry record for a canary pass.

    The aggregate ``chars_saved`` counter gains the prompt-dedup contribution
    ONLY when a real mutation occurred (canary). ``off``/``shadow`` contribute 0
    to the total while still reporting the separated ``prompt_dedup_*`` fields.
    Contains only mode/class enums and integer counters -- never prompt text.
    """
    realized = result.chars_saved if result.mutated else 0
    record = {
        "prompt_dedup_mode": result.mode,
        "prompt_dedup_class": result.prompt_dedup_class,
        "prompt_dedup_blocks_replaced": result.blocks_replaced if result.mutated else 0,
        # Separated field: always present, mirrors the realized prompt-dedup save.
        "prompt_dedup_chars_saved": realized,
        # Aggregate total: includes prompt dedup only when a mutation occurred.
        "chars_saved": realized,
    }
    _assert_no_forbidden_keys(record)
    return record
