"""
Intercept-proxy support for multimodal OpenAI chat requests.

When a ``/v1/chat/completions`` body carries a user message whose content
is a list with several ``image_url`` parts (video frames), the proxy can
treat each image as a context block:

    [text: intro]  [text: label] [image] [text: label] [image] ... [text: question]
     └─ prefix ─┘   └──────────── blocks ────────────────────┘    └─ suffix ─┘

Text parts *between* images are attached to the image that follows them
(they are labels); text before the first image is the prefix, text after
the last image is the suffix.  ``extract_image_blocks`` turns this into
:class:`ImageBlock` objects with content-derived keys, and
``reconstruct_image_blocks`` writes back a reordered block sequence plus an
order-hint sentence at the start of the suffix.

The true sequential position of each block (``order``) is inferred from
its label when possible (``t=12.5s`` or ``Frame 3``); otherwise the
display position in the incoming request is used (i.e. the client is
assumed to send frames chronologically).
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .blocks import ImageBlock
from .prompt import (
    DEFAULT_HINT_TEMPLATE,
    DEFAULT_IN_ORDER_TEMPLATE,
    build_order_hint,
)

_TS_RE = re.compile(r"\bt\s*=\s*(\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE)
_FRAME_RE = re.compile(r"\bframe\s*#?\s*(\d+)\b", re.IGNORECASE)

# Content hash of the image_url value; data: URIs make this content-derived.
def image_key(url: str) -> str:
    return "img:" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def order_from_label(label: str) -> Optional[float]:
    """Parse a timestamp (``t=12.5s``) or frame number (``Frame 3``) from a label."""
    if not label:
        return None
    m = _TS_RE.search(label)
    if m:
        return float(m.group(1))
    m = _FRAME_RE.search(label)
    if m:
        return float(m.group(1))
    return None


@dataclass
class ImageExtraction:
    """Result of :func:`extract_image_blocks`."""

    msg_index: int
    blocks: List[ImageBlock]
    prefix_parts: List[Dict[str, Any]] = field(default_factory=list)
    suffix_parts: List[Dict[str, Any]] = field(default_factory=list)
    # Whether ``order`` came from labels (True) or display position (False)
    order_from_labels: bool = False

    @property
    def keys(self) -> List[str]:
        return [b.key for b in self.blocks]


def _is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") in ("image_url", "input_image")


def _image_url_of(part: Dict[str, Any]) -> Optional[str]:
    if part.get("type") == "image_url":
        v = part.get("image_url")
        if isinstance(v, dict):
            return v.get("url")
        if isinstance(v, str):
            return v
    if part.get("type") == "input_image":
        return part.get("image_url")
    return None


def extract_image_blocks(
    body: Dict[str, Any], *, min_images: int = 2
) -> Optional[ImageExtraction]:
    """Find the last user message with >= ``min_images`` images and parse it.

    Returns ``None`` when no such message exists.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None

    for mi in range(len(messages) - 1, -1, -1):
        msg = messages[mi]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        n_img = sum(1 for p in content if _is_image_part(p))
        if n_img < min_images:
            continue

        img_idx = [i for i, p in enumerate(content) if _is_image_part(p)]
        first_img, last_img = img_idx[0], img_idx[-1]

        # Labels: if every image after the first is immediately preceded by
        # a text part, the text part right before the first image is its
        # label too (not part of the intro).
        def _is_text(p):
            return isinstance(p, dict) and p.get("type") in ("text", "input_text")

        labelled = all(_is_text(content[i - 1]) for i in img_idx[1:])
        if labelled and first_img > 0 and _is_text(content[first_img - 1]):
            first_img -= 1

        prefix = [copy.deepcopy(p) for p in content[:first_img]]
        suffix = [copy.deepcopy(p) for p in content[last_img + 1 :]]

        blocks: List[ImageBlock] = []
        pending_label: List[str] = []
        pending_extra: List[Dict[str, Any]] = []
        for p in content[first_img : last_img + 1]:
            if _is_image_part(p):
                url = _image_url_of(p)
                if url is None:
                    continue
                label = "\n".join(pending_label).strip()
                img = p.get("image_url") if isinstance(p.get("image_url"), dict) else {}
                blocks.append(
                    ImageBlock(
                        key=image_key(url),
                        image_url=url,
                        label=label,
                        detail=img.get("detail") if isinstance(img, dict) else None,
                        meta={"part": copy.deepcopy(p), "label_parts": pending_extra},
                    )
                )
                pending_label, pending_extra = [], []
            elif isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                pending_label.append(str(p.get("text", "")))
                pending_extra.append(copy.deepcopy(p))
            else:  # unknown part type between images: keep it attached to next image
                pending_extra.append(copy.deepcopy(p))

        if len(blocks) < min_images:
            continue

        parsed = [order_from_label(b.label) for b in blocks]
        from_labels = all(o is not None for o in parsed) and len(set(parsed)) == len(parsed)
        for i, b in enumerate(blocks):
            b.order = parsed[i] if from_labels else float(i)

        return ImageExtraction(
            msg_index=mi,
            blocks=blocks,
            prefix_parts=prefix,
            suffix_parts=suffix,
            order_from_labels=from_labels,
        )
    return None


def _block_parts(block: ImageBlock) -> List[Dict[str, Any]]:
    """Original parts for a block (label parts + the image part itself)."""
    label_parts = block.meta.get("label_parts") or []
    part = block.meta.get("part")
    if part is None:  # block not produced by extract_image_blocks
        return block.to_content_parts(include_label=True)
    return [copy.deepcopy(p) for p in label_parts] + [copy.deepcopy(part)]


def reconstruct_image_blocks(
    body: Dict[str, Any],
    extraction: ImageExtraction,
    displayed: Sequence[ImageBlock],
    *,
    order_hint: str = "sentence",
    hint_template: str = DEFAULT_HINT_TEMPLATE,
    in_order_template: Optional[str] = DEFAULT_IN_ORDER_TEMPLATE,
) -> Dict[str, Any]:
    """Return a deep copy of ``body`` with blocks in ``displayed`` order.

    ``order_hint``: ``"sentence"`` inserts the order hint at the start of
    the suffix (before the question); ``"none"`` inserts nothing.  Labels
    are always preserved as they were sent.
    """
    if order_hint not in ("sentence", "none", "labels", "both"):
        raise ValueError(f"unsupported order_hint {order_hint!r}")

    new_body = copy.deepcopy(body)
    content: List[Dict[str, Any]] = list(extraction.prefix_parts)
    for b in displayed:
        content.extend(_block_parts(b))

    hint = ""
    if order_hint in ("sentence", "both"):
        hint = build_order_hint(
            displayed, hint_template=hint_template, in_order_template=in_order_template
        )

    suffix = [copy.deepcopy(p) for p in extraction.suffix_parts]
    if hint:
        if suffix and suffix[0].get("type") in ("text", "input_text"):
            suffix[0]["text"] = hint + "\n\n" + str(suffix[0].get("text", ""))
        else:
            suffix.insert(0, {"type": "text", "text": hint})
    content.extend(suffix)

    new_body["messages"][extraction.msg_index]["content"] = content
    return new_body
