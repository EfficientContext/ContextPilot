"""Content-aware redundancy detection (reporting/measurement only).

These detectors fingerprint in-memory message/tool/LLM-bound text with salted
hashes and emit privacy-safe dataclasses (hashes + counters + low-cardinality
enums). They measure redundancy opportunities; they never drop, summarize, or
mutate any context.
"""
from __future__ import annotations

from typing import Iterable

from .models import (
    PROMPT_DUPLICATE_BLOCK_TYPES,
    BlockTypeStat,
    CrossTypeBlockGroup,
    DuplicateToolOutput,
    PromptDuplicateBlock,
    PromptDuplicateShadow,
    PromptDuplicateTypeCount,
    RepeatedBlock,
    ToolSizeStat,
    TypeCount,
    _est_tokens,
    _LLMContent,
    _ToolMessage,
)
from .privacy import _salted_hash


def detect_exact_duplicate_tool_outputs(
    messages: Iterable[_ToolMessage], *, salt: str, top_n: int
) -> list[DuplicateToolOutput]:
    groups: dict[str, dict] = {}
    for msg in messages:
        content = msg.content
        if not content:
            continue
        h = _salted_hash(content, salt)
        g = groups.get(h)
        if g is None:
            groups[h] = {
                "tool_name": msg.tool_name,
                "occurrences": 1,
                "char_length": len(content),
            }
        else:
            g["occurrences"] += 1
            if g["tool_name"] != msg.tool_name:
                g["tool_name"] = None  # mixed tools produced identical output

    dups: list[DuplicateToolOutput] = []
    for h, g in groups.items():
        if g["occurrences"] < 2:
            continue
        est = _est_tokens(g["char_length"])
        dups.append(
            DuplicateToolOutput(
                content_hash=h,
                tool_name=g["tool_name"],
                occurrences=g["occurrences"],
                char_length=g["char_length"],
                est_tokens=est,
                est_wasted_tokens=est * (g["occurrences"] - 1),
            )
        )
    dups.sort(key=lambda d: d.est_wasted_tokens, reverse=True)
    return dups[:top_n]


def detect_repeated_blocks(
    messages: Iterable[_ToolMessage],
    *,
    salt: str,
    min_block_chars: int,
    min_repeat: int,
    top_n: int,
) -> list[RepeatedBlock]:
    counts: dict[str, dict] = {}
    for msg in messages:
        seen_in_msg: set[str] = set()
        for line in msg.content.splitlines():
            block = line.strip()
            if len(block) < min_block_chars:
                continue
            h = _salted_hash(block, salt)
            # Count cross-message recurrence; collapse repeats within one
            # message so a single noisy output cannot dominate.
            if h in seen_in_msg:
                continue
            seen_in_msg.add(h)
            c = counts.get(h)
            if c is None:
                counts[h] = {"occurrences": 1, "char_length": len(block)}
            else:
                c["occurrences"] += 1

    blocks: list[RepeatedBlock] = []
    for h, c in counts.items():
        if c["occurrences"] < min_repeat:
            continue
        est = _est_tokens(c["char_length"])
        blocks.append(
            RepeatedBlock(
                block_hash=h,
                occurrences=c["occurrences"],
                char_length=c["char_length"],
                est_tokens=est,
                est_wasted_tokens=est * (c["occurrences"] - 1),
            )
        )
    blocks.sort(key=lambda b: b.est_wasted_tokens, reverse=True)
    return blocks[:top_n]


def summarize_tool_sizes(
    messages: Iterable[_ToolMessage], *, large_output_chars: int, top_n: int
) -> list[ToolSizeStat]:
    agg: dict[str, dict] = {}
    for msg in messages:
        name = msg.tool_name or "(unknown)"
        length = len(msg.content)
        a = agg.get(name)
        if a is None:
            agg[name] = {
                "output_count": 1,
                "total_chars": length,
                "max_chars": length,
                "large_output_count": 1 if length >= large_output_chars else 0,
            }
        else:
            a["output_count"] += 1
            a["total_chars"] += length
            a["max_chars"] = max(a["max_chars"], length)
            if length >= large_output_chars:
                a["large_output_count"] += 1

    stats: list[ToolSizeStat] = []
    for name, a in agg.items():
        stats.append(
            ToolSizeStat(
                tool_name=name,
                output_count=a["output_count"],
                total_chars=a["total_chars"],
                max_chars=a["max_chars"],
                avg_chars=a["total_chars"] // a["output_count"],
                total_est_tokens=_est_tokens(a["total_chars"]),
                large_output_count=a["large_output_count"],
            )
        )
    stats.sort(key=lambda s: s.total_chars, reverse=True)
    return stats[:top_n]


def detect_prompt_duplicate_blocks(
    contents: Iterable[_LLMContent],
    *,
    salt: str,
    min_block_chars: int,
    top_n: int,
    enabled: bool = True,
) -> PromptDuplicateShadow:
    """Advisory scan for EXACT duplicate blocks in system/skill prompt text.

    Restricted to ``system_prompt`` / ``skill_prompt`` items. Counts every block
    instance (intra- and inter-prompt) so a block literally present multiple
    times in the static prompt payload is detected. A "duplicate" is any block
    fingerprint observed 2+ times.

    SHADOW/ADVISORY ONLY: output is salted hashes + counters + block-type enums;
    char figures are ACTUAL duplicated chars and the token figure is an ADVISORY
    chars/4 estimate. This never rewrites or dedups prompts and must never be
    reported as a realized saving.
    """
    scanned = list(PROMPT_DUPLICATE_BLOCK_TYPES)
    if not enabled:
        return PromptDuplicateShadow(
            enabled=False,
            item_count=0,
            scanned_block_types=scanned,
            duplicate_group_count=0,
            total_duplicate_occurrences=0,
            total_chars_duplicated=0,
            advisory_est_duplicate_tokens_chars_div_4=0,
            by_block_type=[],
            top_duplicate_blocks=[],
            notes=["prompt-duplicate shadow disabled"],
        )

    # block_hash -> {char_length, types: {block_type: occ}}
    agg: dict[str, dict] = {}
    item_counts: dict[str, int] = {}
    for item in contents:
        bt = item.block_type
        if bt not in PROMPT_DUPLICATE_BLOCK_TYPES:
            continue
        item_counts[bt] = item_counts.get(bt, 0) + 1
        # Count every fingerprintable line (no intra-item dedup) so repeated
        # blocks within a single prompt are surfaced too.
        for line in item.content.splitlines():
            block = line.strip()
            if len(block) < min_block_chars:
                continue
            h = _salted_hash(block, salt)
            entry = agg.get(h)
            if entry is None:
                agg[h] = {"char_length": len(block), "types": {bt: 1}}
            else:
                entry["types"][bt] = entry["types"].get(bt, 0) + 1

    dup_blocks: list[PromptDuplicateBlock] = []
    per_type: dict[str, dict] = {}
    total_chars_dup = 0
    total_dup_occ = 0
    for h, entry in agg.items():
        types = entry["types"]
        occ = sum(types.values())
        if occ < 2:
            continue  # not a duplicate
        char_len = entry["char_length"]
        chars_dup = (occ - 1) * char_len
        total_chars_dup += chars_dup
        total_dup_occ += occ
        dup_blocks.append(
            PromptDuplicateBlock(
                block_hash=h,
                block_types=sorted(types.keys()),
                occurrences=occ,
                char_length=char_len,
                chars_duplicated=chars_dup,
                advisory_est_duplicate_tokens_chars_div_4=_est_tokens(chars_dup),
            )
        )
        for bt, type_occ in types.items():
            t = per_type.setdefault(
                bt, {"blocks": 0, "occ": 0, "chars_dup": 0}
            )
            t["blocks"] += 1
            t["occ"] += type_occ
            # Attribute duplicated chars within this type (occ-1 of the in-type
            # instances are duplicates); cross-type-only blocks contribute 0 here.
            t["chars_dup"] += max(type_occ - 1, 0) * char_len

    by_block_type = [
        PromptDuplicateTypeCount(
            block_type=bt,
            duplicate_block_count=per_type.get(bt, {}).get("blocks", 0),
            occurrence_count=per_type.get(bt, {}).get("occ", 0),
            chars_duplicated=per_type.get(bt, {}).get("chars_dup", 0),
        )
        for bt in scanned
    ]

    dup_blocks.sort(key=lambda b: b.chars_duplicated, reverse=True)
    notes: list[str] = []
    if not item_counts:
        notes.append("no system/skill prompt items observed in the selected window")
    return PromptDuplicateShadow(
        enabled=True,
        item_count=sum(item_counts.values()),
        scanned_block_types=scanned,
        duplicate_group_count=len(dup_blocks),
        total_duplicate_occurrences=total_dup_occ,
        total_chars_duplicated=total_chars_dup,
        advisory_est_duplicate_tokens_chars_div_4=_est_tokens(total_chars_dup),
        by_block_type=by_block_type,
        top_duplicate_blocks=dup_blocks[:top_n],
        notes=notes,
    )


def _iter_blocks(content: str, min_block_chars: int) -> Iterable[str]:
    """Yield the distinct fingerprintable lines of one item (deduped in-item)."""
    seen: set[str] = set()
    for line in content.splitlines():
        block = line.strip()
        if len(block) < min_block_chars:
            continue
        if block in seen:
            continue
        seen.add(block)
        yield block


def analyze_llm_bound_blocks(
    contents: Iterable[_LLMContent],
    *,
    salt: str,
    min_block_chars: int,
    min_repeat: int,
    top_n: int,
) -> tuple[list[BlockTypeStat], list[CrossTypeBlockGroup]]:
    """Fingerprint LLM-bound blocks and report redundancy.

    Returns (per-type stats, cross-type repeated block groups). All output is
    salted hashes / counters / block-type enums -- no raw text.
    """
    # block_hash -> {char_length, types: {block_type: occ}}
    agg: dict[str, dict] = {}
    # block_type -> source item count
    item_counts: dict[str, int] = {}

    for item in contents:
        bt = item.block_type
        item_counts[bt] = item_counts.get(bt, 0) + 1
        for block in _iter_blocks(item.content, min_block_chars):
            h = _salted_hash(block, salt)
            entry = agg.get(h)
            if entry is None:
                agg[h] = {"char_length": len(block), "types": {bt: 1}}
            else:
                entry["types"][bt] = entry["types"].get(bt, 0) + 1

    # --- per block-type aggregate redundancy ------------------------------
    per_type: dict[str, dict] = {}
    for entry in agg.values():
        est = _est_tokens(entry["char_length"])
        for bt, occ in entry["types"].items():
            t = per_type.setdefault(
                bt,
                {
                    "block_count": 0,
                    "unique": 0,
                    "repeated": 0,
                    "redundant_tokens": 0,
                },
            )
            t["block_count"] += occ
            t["unique"] += 1
            if occ >= min_repeat:
                t["repeated"] += 1
                t["redundant_tokens"] += est * (occ - 1)

    block_type_stats: list[BlockTypeStat] = []
    for bt in sorted(set(per_type) | set(item_counts)):
        t = per_type.get(
            bt, {"block_count": 0, "unique": 0, "repeated": 0, "redundant_tokens": 0}
        )
        block_type_stats.append(
            BlockTypeStat(
                block_type=bt,
                item_count=item_counts.get(bt, 0),
                block_count=t["block_count"],
                unique_block_count=t["unique"],
                repeated_block_count=t["repeated"],
                est_redundant_tokens=t["redundant_tokens"],
            )
        )

    # --- cross-type repeated blocks ---------------------------------------
    cross: list[CrossTypeBlockGroup] = []
    for h, entry in agg.items():
        types = entry["types"]
        if len(types) < 2:
            continue
        total_occ = sum(types.values())
        est = _est_tokens(entry["char_length"])
        cross.append(
            CrossTypeBlockGroup(
                block_hash=h,
                block_types=sorted(types.keys()),
                type_occurrences=[
                    TypeCount(block_type=bt, count=occ)
                    for bt, occ in sorted(types.items())
                ],
                occurrences=total_occ,
                char_length=entry["char_length"],
                est_tokens=est,
                est_wasted_tokens=est * (total_occ - 1),
            )
        )
    cross.sort(key=lambda g: g.est_wasted_tokens, reverse=True)
    return block_type_stats, cross[:top_n]
