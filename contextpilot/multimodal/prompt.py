"""
Prompt assembly for reordered multimodal (image / video-frame) contexts.

ContextPilot reorders context blocks so that blocks shared across requests
form a common prefix.  For video frames that destroys the chronological
order the model would normally rely on.  This module rebuilds an OpenAI
chat request whose *frame prefix* is in the cache-friendly order and whose
*suffix* tells the model the true order:

    [system]
    [user]  intro (constant)
            label_1 + image_1     ┐ shared, reorderable prefix
            label_2 + image_2     │ (labels are position-independent)
            ...                   ┘
            order hint             ← varies per request, placed AFTER frames
            question

Order-hint modes (``order_hint``):

* ``"sentence"`` – one sentence listing the true chronological order of
  the images above (what the user asked for).
* ``"labels"``   – rely on per-frame labels (timestamps) only; no sentence.
* ``"both"``     – labels + sentence.
* ``"none"``     – neither (ablation: shuffled frames, no order information).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .blocks import ImageBlock

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that answers questions about a video "
    "using the sampled frames provided."
)

DEFAULT_INTRO = "The following images are frames sampled from a video."

# Placeholders: {order} = comma-separated names in true order.
DEFAULT_HINT_TEMPLATE = (
    "Note: the frames above are NOT shown in chronological order. "
    "Their true chronological order, from earliest to latest, is: {order}."
)
DEFAULT_IN_ORDER_TEMPLATE = (
    "Note: the frames above are shown in chronological order, from earliest to latest."
)

_VALID_HINT_MODES = ("sentence", "labels", "both", "none")


def _display_name(block: ImageBlock, display_pos: int) -> str:
    """Name used to refer to a block inside the order-hint sentence."""
    if block.label:
        return block.label
    return f"image {display_pos}"


def true_order(blocks: Sequence[ImageBlock]) -> List[ImageBlock]:
    """Blocks sorted by their true sequential position (``order``).

    Blocks without an ``order`` keep their given relative order and are
    placed after ordered blocks.
    """
    ordered = [b for b in blocks if b.order is not None]
    unordered = [b for b in blocks if b.order is None]
    ordered.sort(key=lambda b: b.order)  # stable
    return ordered + unordered


def is_chronological(blocks: Sequence[ImageBlock]) -> bool:
    orders = [b.order for b in blocks if b.order is not None]
    return all(a <= b for a, b in zip(orders, orders[1:]))


def build_order_hint(
    displayed: Sequence[ImageBlock],
    *,
    hint_template: str = DEFAULT_HINT_TEMPLATE,
    in_order_template: Optional[str] = DEFAULT_IN_ORDER_TEMPLATE,
) -> str:
    """Build the one-sentence order hint for blocks as *displayed*.

    Returns an empty string if no block carries an ``order``.
    """
    if not any(b.order is not None for b in displayed):
        return ""
    if is_chronological(displayed):
        return in_order_template or ""
    pos = {b.key: i + 1 for i, b in enumerate(displayed)}
    names = [_display_name(b, pos[b.key]) for b in true_order(displayed)]
    return hint_template.format(order=", ".join(names))


def build_multimodal_messages(
    displayed_blocks: Sequence[ImageBlock],
    query: str,
    *,
    system_instruction: Optional[str] = DEFAULT_SYSTEM_INSTRUCTION,
    intro: Optional[str] = DEFAULT_INTRO,
    order_hint: str = "sentence",
    hint_template: str = DEFAULT_HINT_TEMPLATE,
    in_order_template: Optional[str] = DEFAULT_IN_ORDER_TEMPLATE,
    extra_suffix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Assemble OpenAI chat messages from blocks in *display* order.

    Args:
        displayed_blocks: Blocks in the order they should appear in the
            prompt (normally the ContextPilot-reordered order).
        query: The user question (placed last).
        system_instruction: System message text; ``None`` to omit.
        intro: Constant text placed before the first image; ``None`` to omit.
            Must not depend on the request (it is part of the shared prefix).
        order_hint: ``"sentence" | "labels" | "both" | "none"`` (see module doc).
        hint_template: Template for the sentence; ``{order}`` placeholder.
        in_order_template: Sentence used when the displayed order already
            is chronological.  ``None``/``""`` to emit nothing in that case.
        extra_suffix: Optional text appended after the hint, before the query
            (e.g. answer-format instructions).
    """
    if order_hint not in _VALID_HINT_MODES:
        raise ValueError(f"order_hint must be one of {_VALID_HINT_MODES}, got {order_hint!r}")

    include_labels = order_hint in ("labels", "both")
    include_sentence = order_hint in ("sentence", "both")

    content: List[Dict[str, Any]] = []
    if intro:
        content.append({"type": "text", "text": intro})
    for b in displayed_blocks:
        content.extend(b.to_content_parts(include_label=include_labels))

    tail: List[str] = []
    if include_sentence:
        hint = build_order_hint(
            displayed_blocks,
            hint_template=hint_template,
            in_order_template=in_order_template,
        )
        if hint:
            tail.append(hint)
    if extra_suffix:
        tail.append(extra_suffix)
    tail.append(query)
    content.append({"type": "text", "text": "\n\n".join(tail)})

    messages: List[Dict[str, Any]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": content})
    return messages
