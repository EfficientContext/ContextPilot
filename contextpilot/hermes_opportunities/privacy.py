"""Privacy primitives: salted hashing and the forbidden-output guard.

These functions are the privacy backbone of the analyzer. Message/tool/system
text may be read into memory for hashing, but it must never reach an output
file; ``_assert_no_forbidden_keys`` is the defensive backstop that enforces it.
"""
from __future__ import annotations

import hashlib

# Columns we are explicitly forbidden from EMITTING in any report. We may read
# message content in-memory for hashing, but it must never reach an output file.
FORBIDDEN_OUTPUT_KEYS = {
    "content",
    "system_prompt",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "tool_calls",
    "codex_reasoning_items",
    "codex_message_items",
}


def _salted_hash(text: str, salt: str, *, length: int = 16) -> str:
    return hashlib.sha256(f"{salt}:{text}".encode("utf-8", "replace")).hexdigest()[:length]


def _salt_fingerprint(salt: str) -> str:
    # Confirms a salt was applied without revealing it.
    return hashlib.sha256(f"fingerprint:{salt}".encode()).hexdigest()[:12]


def _assert_no_forbidden_keys(data: dict) -> None:
    """Defensive guard: ensure no forbidden raw-content key reached the output."""

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in FORBIDDEN_OUTPUT_KEYS:
                    raise RuntimeError(f"refusing to emit forbidden key: {k}")
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
