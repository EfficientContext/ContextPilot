"""Content-aware redundancy detection (reporting/measurement only).

These detectors fingerprint in-memory message/tool/LLM-bound text with salted
hashes and emit privacy-safe dataclasses (hashes + counters + low-cardinality
enums). They measure redundancy opportunities; they never drop, summarize, or
mutate any context.
"""
from __future__ import annotations

from typing import Iterable

from .models import (
    BlockTypeStat,
    CrossTypeBlockGroup,
    DuplicateToolOutput,
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
