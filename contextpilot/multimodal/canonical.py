"""
Canonical frame prefixes for video RAG.

Reordering retrieved frames so that shared frames come first (see
:mod:`contextpilot.multimodal.api`) turns *set* overlap into *prefix* overlap.
On engines whose prefix cache is content-hashed that is enough. On hybrid
linear-attention models (Qwen3.5 / Qwen3.8 / GLM-5.3-Flash) it is not: their
radix cache can only restore a recurrent state at recorded checkpoints, which
exist at a previous request's end and at branch points recorded by an earlier
miss. A prefix boundary that occurs only once is never reused, so per-request
reordering — where every pair of questions shares a slightly different number
of leading frames — yields no KV reuse at all.

This module removes that variance. For a group of requests over the same video
it picks a **core** set of frames (the ones most requests retrieved anyway),
pins them to the front of every prompt in one fixed (chronological) order, and
appends each request's remaining frames afterwards. Every prompt in the group
then shares one identical boundary, so the core is computed twice and reused by
every later request.

``include_missing=True`` also adds core frames a given request did not retrieve.
Those frames cost prefill only until the core is cached, after which they are
free, and they keep the boundary identical for every request in the group.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

from .blocks import ImageBlock
from .prompt import true_order


def select_core(
    block_lists: Sequence[Sequence[ImageBlock]],
    *,
    min_share: float = 0.5,
    core_size: Optional[int] = None,
    min_core: int = 1,
) -> List[ImageBlock]:
    """Pick the frames that should be pinned to the front for this group.

    A frame is a core candidate when at least ``min_share`` of the requests
    retrieved it (and at least two of them, since a frame only one request
    needs cannot be shared). ``core_size`` caps the core (most frequent first, ties broken
    chronologically). The returned blocks are in chronological order.
    """
    if not block_lists:
        return []
    n = len(block_lists)
    counts: Counter = Counter()
    pool: Dict[str, ImageBlock] = {}
    for blocks in block_lists:
        for key in {b.key for b in blocks}:
            counts[key] += 1
        for b in blocks:
            pool.setdefault(b.key, b)

    # A frame only helps the prefix if more than one request needs it, so the
    # share threshold is floored at 2 whenever the group has several requests.
    threshold = max(1, math.ceil(min_share * n))
    if n >= 2:
        threshold = max(2, threshold)
    keys = [k for k, c in counts.items() if c >= threshold]
    if len(keys) < min_core:
        keys = [k for k, _ in counts.most_common(min_core)]
    if core_size is not None and len(keys) > core_size:
        keys.sort(key=lambda k: (-counts[k], _order_of(pool[k])))
        keys = keys[:core_size]
    return true_order([pool[k] for k in keys])


def _order_of(b: ImageBlock) -> float:
    return b.order if b.order is not None else float("inf")


def plan_canonical(
    blocks: Sequence[ImageBlock],
    core: Sequence[ImageBlock],
    *,
    include_missing: bool = True,
    tail_chronological: bool = True,
    budget: Optional[int] = None,
) -> List[ImageBlock]:
    """Display order for one request: canonical core first, then its own frames.

    Args:
        blocks: the frames retrieved for this request, best first.
        core: the group's canonical core (from :func:`select_core`).
        include_missing: also show core frames this request did not retrieve,
            so every request in the group shares one identical prefix.
        tail_chronological: order the non-core remainder chronologically
            (the alternative is to keep the retrieval order).
        budget: cap on the number of frames shown. Pinning core frames the
            request did not retrieve would otherwise make the prompt longer
            than the baseline; with a budget the lowest-ranked own frames are
            dropped instead, so prompt length is unchanged and the saving is
            real. The core is never truncated (that would break the shared
            boundary), so the budget is a floor of ``len(core)``.
    """
    core_keys = {b.key for b in core}
    by_key = {b.key: b for b in blocks}
    prefix: List[ImageBlock] = []
    for cb in core:
        if cb.key in by_key:
            prefix.append(by_key[cb.key])
        elif include_missing:
            prefix.append(cb)
    seen = {b.key for b in prefix}
    tail = [b for b in blocks if b.key not in seen and b.key not in core_keys]
    if budget is not None:
        # blocks are in retrieval order (best first), so keep the head
        tail = tail[: max(0, budget - len(prefix))]
    if tail_chronological:
        tail = true_order(tail)
    return prefix + tail


def plan_canonical_batch(
    all_blocks: Sequence[Sequence[ImageBlock]],
    group_keys: Sequence[Hashable],
    *,
    min_share: float = 0.5,
    core_size: Optional[int] = None,
    include_missing: bool = True,
    tail_chronological: bool = True,
    budget: Optional[int] = None,
) -> Tuple[List[List[ImageBlock]], Dict[Hashable, List[ImageBlock]]]:
    """Plan a whole batch: one canonical core per group (e.g. per video).

    Returns ``(displayed_batch, cores)`` where ``displayed_batch[i]``
    corresponds to ``all_blocks[i]`` (the input order is preserved; use
    :func:`group_execution_order` to decide what to send when).
    """
    if len(all_blocks) != len(group_keys):
        raise ValueError(
            f"all_blocks ({len(all_blocks)}) and group_keys "
            f"({len(group_keys)}) must have the same length."
        )
    grouped: Dict[Hashable, List[Sequence[ImageBlock]]] = {}
    for blocks, g in zip(all_blocks, group_keys):
        grouped.setdefault(g, []).append(blocks)

    cores = {
        g: select_core(lists, min_share=min_share, core_size=core_size)
        for g, lists in grouped.items()
    }
    displayed = [
        plan_canonical(
            blocks, cores[g],
            include_missing=include_missing,
            tail_chronological=tail_chronological,
            budget=budget,
        )
        for blocks, g in zip(all_blocks, group_keys)
    ]
    return displayed, cores


def group_execution_order(group_keys: Sequence[Hashable]) -> List[int]:
    """Send requests of the same group consecutively (cache-aware order).

    Groups keep the order of first appearance, and requests keep their order
    within a group, so the result is deterministic.
    """
    order: Dict[Hashable, List[int]] = {}
    for i, g in enumerate(group_keys):
        order.setdefault(g, []).append(i)
    return [i for idxs in order.values() for i in idxs]
