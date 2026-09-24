"""Prompt dedup A/B simulation harness (OFFLINE simulation + measurement only).

This is the evidence gate to evaluate *before* any canary prompt replacement.
It scans ONLY ``system_prompt`` / ``skill_prompt`` LLM-bound blocks, fingerprints
exact duplicate blocks, and simulates -- in accounting only -- keeping the first
occurrence of each duplicate while replacing every later occurrence with a
deterministic reference placeholder.

Hard guarantees:

* It never mutates the DB, runtime state, or any emitted prompt; it produces no
  side effects beyond the privacy-safe report dataclasses below.
* It emits salted hashes / counters / low-cardinality enums only -- never raw
  prompt text and never the reference placeholder filled with real content.
* Char and token deltas are SIMULATED candidate figures, explicitly NOT realized
  savings. ContextPilot performs no canonicalization or replacement at runtime.
* Exact token figures appear only when an explicitly configured tokenizer backend
  is available (opt-in, off by default); otherwise the status is ``unavailable``
  and no actual-token fields are populated.
"""
from __future__ import annotations

from typing import Iterable

from .models import (
    PROMPT_DEDUP_AB_CLASSES,
    PROMPT_DEDUP_AB_REFERENCE_TEMPLATE,
    PROMPT_DUPLICATE_BLOCK_TYPES,
    PromptDedupABClass,
    PromptDedupABSimulation,
    _LLMContent,
)
from .privacy import _salted_hash
from .tokenizer import TokenizerBackend

# Per-class risk label + advisory note. The skill-only class is the lowest-risk
# first canary candidate; the other classes are reported but flagged higher risk.
_CLASS_META = {
    "same_type_skill_prompt_only": (
        "low",
        "first canary candidate: exact duplicate blocks within skill prompts only",
    ),
    "same_type_system_prompt_only": (
        "high",
        "higher risk: exact duplicate blocks within system prompts only",
    ),
    "cross_type_system_skill": (
        "high",
        "higher risk: exact duplicate blocks shared across system and skill prompts",
    ),
}


def _classify_group(types: dict[str, int]) -> str | None:
    """Map a duplicate group's prompt-type spread to a candidate class."""
    present = set(types)
    if present == {"skill_prompt"}:
        return "same_type_skill_prompt_only"
    if present == {"system_prompt"}:
        return "same_type_system_prompt_only"
    if present == {"system_prompt", "skill_prompt"}:
        return "cross_type_system_skill"
    return None  # only system/skill are scanned; anything else is ignored


def _canonical_type(types: dict[str, int]) -> str:
    """Deterministically pick the canonical prompt type for the reference string.

    Dominant by occurrence count; ties broken by sorted type name so the choice
    is stable across runs and inputs.
    """
    return sorted(types.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _reference_string(canonical_type: str, block_hash: str) -> str:
    return PROMPT_DEDUP_AB_REFERENCE_TEMPLATE.replace("<type>", canonical_type).replace(
        "<hash>", block_hash
    )


def simulate_prompt_dedup_ab(
    contents: Iterable[_LLMContent],
    *,
    salt: str,
    min_block_chars: int,
    tokenizer: TokenizerBackend | None = None,
    enabled: bool = True,
) -> PromptDedupABSimulation:
    """Simulate prompt-dedup replacement over system/skill prompt blocks.

    Restricted to ``system_prompt`` / ``skill_prompt`` items. Every fingerprintable
    line is counted (intra- and inter-prompt), and any fingerprint seen 2+ times
    is a duplicate group. Each group is assigned to exactly one candidate class and
    simulated independently: the first occurrence is kept full, every later
    occurrence is replaced (in accounting only) by the deterministic reference
    string ``[Prompt duplicate omitted in simulation; canonical=<type>:<hash>]``.

    Returns a privacy-safe :class:`PromptDedupABSimulation` -- hashes, counters,
    and enums only. No DB/runtime/payload is touched.
    """
    scanned = list(PROMPT_DUPLICATE_BLOCK_TYPES)
    tok_status = "available" if tokenizer is not None else "unavailable"
    tok_backend = tokenizer.name if tokenizer is not None else None

    if not enabled:
        return PromptDedupABSimulation(
            enabled=False,
            item_count=0,
            scanned_block_types=scanned,
            tokenizer_status="unavailable",
            tokenizer_backend=None,
            reference_string_template=PROMPT_DEDUP_AB_REFERENCE_TEMPLATE,
            classes=[],
            notes=["prompt-dedup A/B simulation disabled"],
        )

    # block_hash -> {char_length, text (in-memory only), types: {block_type: occ}}
    agg: dict[str, dict] = {}
    item_count = 0
    for item in contents:
        bt = item.block_type
        if bt not in PROMPT_DUPLICATE_BLOCK_TYPES:
            continue
        item_count += 1
        for line in item.content.splitlines():
            block = line.strip()
            if len(block) < min_block_chars:
                continue
            h = _salted_hash(block, salt)
            entry = agg.get(h)
            if entry is None:
                # ``text`` is held in-memory only for exact token counting; it is
                # never written to the report (no dataclass field carries it).
                agg[h] = {"char_length": len(block), "text": block, "types": {bt: 1}}
            else:
                entry["types"][bt] = entry["types"].get(bt, 0) + 1

    # Per-class running totals.
    acc: dict[str, dict] = {
        cls: {
            "groups": 0,
            "repl_occ": 0,
            "chars_before": 0,
            "chars_after": 0,
            "tok_before": 0,
            "tok_after": 0,
        }
        for cls in PROMPT_DEDUP_AB_CLASSES
    }

    for h, entry in agg.items():
        types = entry["types"]
        occ = sum(types.values())
        if occ < 2:
            continue  # not a duplicate -> no replacement candidate
        cls = _classify_group(types)
        if cls is None:
            continue
        char_len = entry["char_length"]
        ref = _reference_string(_canonical_type(types), h)
        ref_len = len(ref)

        a = acc[cls]
        a["groups"] += 1
        a["repl_occ"] += occ - 1
        a["chars_before"] += occ * char_len
        # Keep first occurrence full; later occurrences become the reference str.
        a["chars_after"] += char_len + (occ - 1) * ref_len
        if tokenizer is not None:
            tb = tokenizer.count(entry["text"])
            tr = tokenizer.count(ref)
            a["tok_before"] += occ * tb
            a["tok_after"] += tb + (occ - 1) * tr

    classes: list[PromptDedupABClass] = []
    for cls in PROMPT_DEDUP_AB_CLASSES:
        a = acc[cls]
        risk, note = _CLASS_META[cls]
        if tokenizer is not None:
            tok_before = a["tok_before"]
            tok_after = a["tok_after"]
            tok_delta = tok_before - tok_after
        else:
            tok_before = tok_after = tok_delta = None
        classes.append(
            PromptDedupABClass(
                candidate_class=cls,
                risk_label=risk,
                candidate_group_count=a["groups"],
                replacement_occurrence_count=a["repl_occ"],
                chars_before=a["chars_before"],
                chars_after_simulated=a["chars_after"],
                chars_delta_simulated=a["chars_before"] - a["chars_after"],
                tokenizer_status=tok_status,
                actual_tokens_before=tok_before,
                actual_tokens_after=tok_after,
                actual_tokens_delta=tok_delta,
                note=note,
            )
        )

    notes = [
        "OFFLINE SIMULATION + MEASUREMENT ONLY: no DB/runtime/prompt is mutated; "
        "ContextPilot performs no replacement or canonicalization",
        "char/token deltas are SIMULATED candidate figures, NOT realized savings",
        "this A/B evidence is the gate to evaluate before any canary prompt replacement",
        "same_type_skill_prompt_only is the lowest-risk first canary candidate; "
        "system-only and cross-type classes are higher risk",
        "chars_delta_simulated is signed: negative means a short duplicate would grow "
        "if replaced by the reference placeholder",
    ]
    if tokenizer is None:
        notes.append(
            "actual-token measurement unavailable (no exact tokenizer backend configured); "
            "no actual-token fields are reported"
        )
    if item_count == 0:
        notes.append("no system/skill prompt items observed in the selected window")

    return PromptDedupABSimulation(
        enabled=True,
        item_count=item_count,
        scanned_block_types=scanned,
        tokenizer_status=tok_status,
        tokenizer_backend=tok_backend,
        reference_string_template=PROMPT_DEDUP_AB_REFERENCE_TEMPLATE,
        classes=classes,
        notes=notes,
    )
