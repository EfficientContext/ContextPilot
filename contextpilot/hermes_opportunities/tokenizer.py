"""Optional, opt-in exact tokenizer helper for offline simulation only.

Mirrors the philosophy of the actual-token prompt shadow (#53): exact token
counts are surfaced ONLY when an explicitly configured tokenizer backend is
available. By default this module resolves to ``None`` (status ``unavailable``),
and callers must never fabricate token figures in that case.

This helper runs in-memory over block text purely to produce integer counts; it
never emits or persists any text. It is used by the prompt-dedup A/B *simulation*
harness, which measures candidate savings offline and never mutates runtime
payloads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class TokenizerBackend:
    """A resolved exact tokenizer.

    ``name`` is a low-cardinality backend identifier safe to emit (e.g.
    ``"tiktoken:cl100k_base"``); ``count`` maps a string to its exact token
    count. Counting happens in-memory only -- the text is never emitted.
    """

    name: str
    count: Callable[[str], int]


def resolve_tokenizer(spec: object | None) -> TokenizerBackend | None:
    """Resolve an explicitly-configured exact tokenizer backend, or ``None``.

    Off by default: ``spec=None`` (the default everywhere) returns ``None`` so
    the A/B harness reports ``tokenizer_status=unavailable`` and emits no actual
    token fields. ``spec`` may be:

    * ``None`` -> not configured; returns ``None``.
    * a :class:`TokenizerBackend` -> used directly (test/dependency injection).
    * a string ``"tiktoken:<encoding>"`` -> best-effort load of a tiktoken
      encoding. If tiktoken (or the encoding) is unavailable, returns ``None``
      rather than guessing; the caller then reports ``unavailable``.

    Any backend that cannot be resolved exactly yields ``None`` -- we never
    substitute a chars/4 estimate behind an "actual tokens" label.
    """
    if spec is None:
        return None
    if isinstance(spec, TokenizerBackend):
        return spec
    if isinstance(spec, str):
        return _resolve_named(spec)
    return None


def _resolve_named(spec: str) -> TokenizerBackend | None:
    spec = spec.strip()
    if not spec:
        return None
    if spec.startswith("tiktoken:"):
        encoding = spec.split(":", 1)[1].strip() or "cl100k_base"
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding(encoding)
        except Exception:  # noqa: BLE001 - missing dep/encoding -> unavailable, never fake
            return None
        return TokenizerBackend(
            name=f"tiktoken:{encoding}",
            count=lambda text: len(enc.encode(text)),
        )
    # Unknown backend spec: stay unavailable rather than fabricate counts.
    return None
