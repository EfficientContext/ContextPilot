"""
High-level multimodal API: reorder image / video-frame blocks for KV-cache
prefix sharing and return ready-to-send OpenAI chat messages.

    import contextpilot as cp
    from contextpilot.multimodal import video_frames_to_blocks, optimize_multimodal

    blocks = video_frames_to_blocks(frame_paths, timestamps, video_id="vid42")
    retrieved = [blocks[i] for i in retriever.top_k(question)]
    messages = optimize_multimodal(retrieved, question)     # frames reordered
    client.chat.completions.create(model=..., messages=messages)

The Context Index only sees block *keys* (strings); images never enter
the clustering code.  The same :class:`~contextpilot.ContextPilot`
instance can therefore index text documents and image blocks together.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .blocks import ImageBlock
from .prompt import (
    DEFAULT_HINT_TEMPLATE,
    DEFAULT_IN_ORDER_TEMPLATE,
    DEFAULT_INTRO,
    DEFAULT_SYSTEM_INSTRUCTION,
    build_multimodal_messages,
)

_default_pilot = None


def _get_pilot():
    global _default_pilot
    if _default_pilot is None:
        from ..server.live_index import ContextPilot

        _default_pilot = ContextPilot(use_gpu=False)
    return _default_pilot


def reorder_blocks(
    blocks: Sequence[ImageBlock],
    *,
    pilot=None,
    conversation_id: Optional[str] = None,
) -> List[ImageBlock]:
    """Reorder one request's blocks for prefix sharing with the live index.

    Returns the same blocks in ContextPilot's cache-friendly order.
    Duplicate keys within a request are collapsed to their first
    occurrence (the engine cannot benefit from repeating an image).
    """
    pilot = pilot or _get_pilot()
    by_key: Dict[str, ImageBlock] = {}
    keys: List[str] = []
    for b in blocks:
        if b.key not in by_key:
            by_key[b.key] = b
            keys.append(b.key)
    if len(keys) < 2:
        return [by_key[k] for k in keys]
    reordered, _ = pilot.reorder([keys], conversation_id=conversation_id)
    return [by_key[k] for k in reordered[0]]


def reorder_blocks_batch(
    all_blocks: Sequence[Sequence[ImageBlock]],
    *,
    pilot=None,
) -> Tuple[List[List[ImageBlock]], List[int]]:
    """Globally reorder a batch of requests and schedule their execution.

    Returns ``(reordered_batch, original_indices)`` where
    ``reordered_batch[i]`` is the block list for request
    ``all_blocks[original_indices[i]]``.
    """
    pilot = pilot or _get_pilot()
    lookup: Dict[str, ImageBlock] = {}
    key_lists: List[List[str]] = []
    for blocks in all_blocks:
        keys: List[str] = []
        for b in blocks:
            lookup.setdefault(b.key, b)
            if b.key not in keys:
                keys.append(b.key)
        key_lists.append(keys)
    reordered, order = pilot.reorder(key_lists)
    return [[lookup[k] for k in ctx] for ctx in reordered], list(order)


def optimize_multimodal(
    blocks: Sequence[ImageBlock],
    query: str,
    *,
    pilot=None,
    conversation_id: Optional[str] = None,
    reorder: bool = True,
    order_hint: str = "sentence",
    system_instruction: Optional[str] = DEFAULT_SYSTEM_INSTRUCTION,
    intro: Optional[str] = DEFAULT_INTRO,
    hint_template: str = DEFAULT_HINT_TEMPLATE,
    in_order_template: Optional[str] = DEFAULT_IN_ORDER_TEMPLATE,
    extra_suffix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Reorder image blocks and return OpenAI chat messages.

    Args:
        blocks: Retrieved image blocks (any order, e.g. by retrieval score).
        query: User question.
        pilot: A :class:`~contextpilot.ContextPilot` instance; defaults to
            a module-level singleton (shared with :func:`reorder_blocks`).
        conversation_id: Key for multi-turn tracking in the live index.
        reorder: ``False`` sends blocks in *chronological* order (baseline).
        order_hint: How the true order is communicated
            (``"sentence" | "labels" | "both" | "none"``).
        Remaining arguments are forwarded to
        :func:`~contextpilot.multimodal.prompt.build_multimodal_messages`.
    """
    from .prompt import true_order

    if reorder:
        displayed = reorder_blocks(blocks, pilot=pilot, conversation_id=conversation_id)
    else:
        displayed = true_order(blocks)
    return build_multimodal_messages(
        displayed,
        query,
        system_instruction=system_instruction,
        intro=intro,
        order_hint=order_hint,
        hint_template=hint_template,
        in_order_template=in_order_template,
        extra_suffix=extra_suffix,
    )


def optimize_multimodal_batch(
    all_blocks: Sequence[Sequence[ImageBlock]],
    all_queries: Sequence[str],
    *,
    pilot=None,
    order_hint: str = "sentence",
    system_instruction: Optional[str] = DEFAULT_SYSTEM_INSTRUCTION,
    intro: Optional[str] = DEFAULT_INTRO,
    hint_template: str = DEFAULT_HINT_TEMPLATE,
    in_order_template: Optional[str] = DEFAULT_IN_ORDER_TEMPLATE,
    extra_suffix: Optional[str] = None,
) -> Tuple[List[List[Dict[str, Any]]], List[int]]:
    """Batch variant: global reorder + cache-aware execution order.

    Returns ``(messages_batch, original_indices)`` — send
    ``messages_batch[i]`` for query ``all_queries[original_indices[i]]``,
    in that order, to maximise prefix hits.
    """
    if len(all_blocks) != len(all_queries):
        raise ValueError(
            f"all_blocks ({len(all_blocks)}) and all_queries "
            f"({len(all_queries)}) must have the same length."
        )
    reordered_batch, order = reorder_blocks_batch(all_blocks, pilot=pilot)
    messages_batch = []
    for displayed, orig_idx in zip(reordered_batch, order):
        messages_batch.append(
            build_multimodal_messages(
                displayed,
                all_queries[orig_idx],
                system_instruction=system_instruction,
                intro=intro,
                order_hint=order_hint,
                hint_template=hint_template,
                in_order_template=in_order_template,
                extra_suffix=extra_suffix,
            )
        )
    return messages_batch, order
