"""
Multimodal context blocks for ContextPilot.

A :class:`ImageBlock` is the multimodal analogue of a retrieved text
document: an image (typically a video frame) plus the metadata needed to
(1) give it a *stable identity* so the Context Index can recognise the
same frame across requests, and (2) restore its *true position* in the
source sequence after ContextPilot reorders blocks for prefix-cache
sharing.

Identity (``key``) is content-derived by default (SHA-256 of the image
bytes / URL), so two requests that retrieve the same frame — even from
different retrieval ranks — map to the same key and can share KV cache.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union


def _sha256(data: Union[bytes, str]) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass
class ImageBlock:
    """One image (e.g. a video frame) treated as a reorderable context block.

    Attributes:
        key: Stable identity used by the Context Index.  Requests that
            contain the same ``key`` are recognised as sharing content.
            Defaults to a hash of the image payload.
        image_url: Value for the OpenAI ``image_url.url`` field — either a
            ``data:`` URI or an http(s) URL.
        order: True position of this block in its source sequence
            (e.g. a timestamp in seconds, or a frame index).  Used to
            generate the *order hint* after reordering.  ``None`` means
            the block has no meaningful sequential position.
        label: Optional short text rendered immediately before the image
            (e.g. ``"Frame 12 (t=34.5s)"``).  Labels are part of the
            shared prefix, so they must be a function of the block itself
            — never of its position in the request.
        detail: Optional OpenAI ``image_url.detail`` (``"low"``/``"high"``/``"auto"``).
        meta: Free-form metadata (video id, frame path, retrieval score…).
    """

    key: str
    image_url: str
    order: Optional[float] = None
    label: str = ""
    detail: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # ── constructors ─────────────────────────────────────────────────
    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        mime: str = "image/jpeg",
        order: Optional[float] = None,
        label: str = "",
        key: Optional[str] = None,
        detail: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "ImageBlock":
        b64 = base64.b64encode(data).decode("ascii")
        return cls(
            key=key or _sha256(data),
            image_url=f"data:{mime};base64,{b64}",
            order=order,
            label=label,
            detail=detail,
            meta=dict(meta or {}),
        )

    @classmethod
    def from_path(
        cls,
        path: Union[str, os.PathLike],
        *,
        order: Optional[float] = None,
        label: str = "",
        key: Optional[str] = None,
        detail: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "ImageBlock":
        path = os.fspath(path)
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f:
            data = f.read()
        m = {"path": path}
        m.update(meta or {})
        return cls.from_bytes(
            data, mime=mime, order=order, label=label, key=key, detail=detail, meta=m
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        order: Optional[float] = None,
        label: str = "",
        key: Optional[str] = None,
        detail: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "ImageBlock":
        """Wrap an existing ``image_url`` value (http(s) or ``data:`` URI).

        The default key is a hash of the URL string: for ``data:`` URIs
        this is content-derived; for http(s) URLs it assumes the URL is
        a stable identifier for the image.
        """
        return cls(
            key=key or _sha256(url),
            image_url=url,
            order=order,
            label=label,
            detail=detail,
            meta=dict(meta or {}),
        )

    # ── rendering ────────────────────────────────────────────────────
    def to_content_parts(self, *, include_label: bool = True) -> List[Dict[str, Any]]:
        """Render as OpenAI chat ``content`` parts (label text + image)."""
        parts: List[Dict[str, Any]] = []
        if include_label and self.label:
            parts.append({"type": "text", "text": self.label})
        img: Dict[str, Any] = {"url": self.image_url}
        if self.detail:
            img["detail"] = self.detail
        parts.append({"type": "image_url", "image_url": img})
        return parts

    def __hash__(self) -> int:  # allow use in sets / dict keys
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ImageBlock) and other.key == self.key


def frame_label(index: int, timestamp: Optional[float] = None, *, one_based: bool = True) -> str:
    """Default label for a video frame: ``"[Frame 3 | t=12.5s]"``.

    Labels are position-independent (they describe the frame, not its
    slot in the prompt) so that reordering does not change the tokens
    of the shared prefix.
    """
    n = index + 1 if one_based else index
    if timestamp is None:
        return f"[Frame {n}]"
    return f"[Frame {n} | t={timestamp:.1f}s]"


def video_frames_to_blocks(
    frame_paths: Sequence[Union[str, os.PathLike]],
    timestamps: Optional[Sequence[float]] = None,
    *,
    video_id: Optional[str] = None,
    label_fn=frame_label,
    detail: Optional[str] = None,
) -> List[ImageBlock]:
    """Turn a list of frame image paths into ImageBlocks in chronological order.

    ``order`` is the timestamp when given, else the frame index.  If
    ``video_id`` is given, the key is ``f"{video_id}#{index}"`` (cheap,
    stable across processes, and independent of JPEG re-encoding);
    otherwise the key is the content hash.
    """
    blocks: List[ImageBlock] = []
    for i, p in enumerate(frame_paths):
        ts = None if timestamps is None else float(timestamps[i])
        key = f"{video_id}#{i}" if video_id is not None else None
        blocks.append(
            ImageBlock.from_path(
                p,
                order=ts if ts is not None else float(i),
                label=label_fn(i, ts),
                key=key,
                detail=detail,
                meta={"video_id": video_id, "frame_index": i, "timestamp": ts},
            )
        )
    return blocks
