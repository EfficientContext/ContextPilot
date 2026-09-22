"""
Multimodal (image / video-frame) context support for ContextPilot.

Public API:

* :class:`ImageBlock` — an image treated as a reorderable context block
  with a stable key and a true sequential position (``order``).
* :func:`video_frames_to_blocks` — helper to turn frame files into blocks.
* :func:`optimize_multimodal` / :func:`optimize_multimodal_batch` —
  reorder blocks for KV-cache prefix sharing and return OpenAI chat
  messages whose suffix tells the model the true chronological order.
* :func:`build_multimodal_messages` / :func:`build_order_hint` — prompt
  assembly primitives.
* :func:`extract_image_blocks` / :func:`reconstruct_image_blocks` —
  intercept-proxy helpers for OpenAI chat bodies containing images.
"""

from .blocks import ImageBlock, frame_label, video_frames_to_blocks
from .prompt import (
    DEFAULT_HINT_TEMPLATE,
    DEFAULT_IN_ORDER_TEMPLATE,
    DEFAULT_INTRO,
    DEFAULT_SYSTEM_INSTRUCTION,
    build_multimodal_messages,
    build_order_hint,
    is_chronological,
    true_order,
)
from .api import (
    optimize_multimodal,
    optimize_multimodal_batch,
    reorder_blocks,
    reorder_blocks_batch,
)
from .intercept import (
    ImageExtraction,
    extract_image_blocks,
    reconstruct_image_blocks,
)

__all__ = [
    "ImageBlock",
    "frame_label",
    "video_frames_to_blocks",
    "build_multimodal_messages",
    "build_order_hint",
    "is_chronological",
    "true_order",
    "optimize_multimodal",
    "optimize_multimodal_batch",
    "reorder_blocks",
    "reorder_blocks_batch",
    "ImageExtraction",
    "extract_image_blocks",
    "reconstruct_image_blocks",
    "DEFAULT_HINT_TEMPLATE",
    "DEFAULT_IN_ORDER_TEMPLATE",
    "DEFAULT_INTRO",
    "DEFAULT_SYSTEM_INSTRUCTION",
]
