"""
Convenience functions for common ContextPilot workflows.

These functions provide a minimal-code interface so users can add
ContextPilot to an existing program with just two lines:

    import contextpilot as cp          # line 1
    messages = cp.optimize(docs, query) # line 2
"""

from typing import List, Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_pilot = None


def _get_pilot():
    """Return (or lazily create) the module-level ContextPilot instance."""
    global _default_pilot
    if _default_pilot is None:
        from .server.live_index import ContextPilot
        _default_pilot = ContextPilot(use_gpu=False)
    return _default_pilot


# ---------------------------------------------------------------------------
# Prompt helpers (internal)
# ---------------------------------------------------------------------------

def _build_system_prompt(
    reordered_docs: List[str],
    original_docs: List[str],
    system_instruction: Optional[str] = None,
) -> str:
    """Build a system prompt with reordered documents and importance ranking."""
    docs_section = "\n".join(
        f"[{i + 1}] {doc}" for i, doc in enumerate(reordered_docs)
    )

    # Importance ranking: maps each doc back to its position in the
    # reordered list, then lists those positions in the *original* order
    # so the model still prioritises by relevance.
    pos = {doc: i + 1 for i, doc in enumerate(reordered_docs)}
    importance_ranking = " > ".join(
        str(pos[doc]) for doc in original_docs if doc in pos
    )

    parts = []
    if system_instruction:
        parts.append(system_instruction)
    parts.append(
        f"Answer the question based on the provided documents.\n\n"
        f"<documents>\n{docs_section}\n</documents>\n\n"
        f"Read the documents in this importance ranking: {importance_ranking}\n"
        f"Prioritize information from higher-ranked documents."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize(
    docs: List[str],
    query: str,
    *,
    conversation_id: Optional[str] = None,
    system_instruction: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Optimize context ordering and return ready-to-use OpenAI messages.

    Takes your retrieved documents and query, reorders documents for
    maximum KV-cache prefix sharing, and returns a ``messages`` list
    that can be passed directly to ``client.chat.completions.create()``.

    Example::

        import contextpilot as cp
        messages = cp.optimize(docs, query)
        response = client.chat.completions.create(
            model="Qwen/Qwen3-4B", messages=messages
        )

    Args:
        docs: List of document strings (from RAG retrieval, Mem0, etc.).
        query: The user question.
        conversation_id: Optional key for multi-turn deduplication.
            Pass a unique ID (e.g. ``user_id`` or ``session_id``) to
            enable cross-turn deduplication.
        system_instruction: Optional extra instruction prepended to the
            system message (e.g. ``"Answer in Chinese."``).

    Returns:
        A list of message dicts (``role`` / ``content``) ready for the
        OpenAI chat completions API.
    """
    pilot = _get_pilot()
    reordered, _indices = pilot.reorder(docs, conversation_id=conversation_id)
    reordered_docs = reordered[0]  # single context → first element

    system_content = _build_system_prompt(
        reordered_docs, docs, system_instruction
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]


def optimize_batch(
    all_docs: List[List[str]],
    all_queries: List[str],
    *,
    system_instruction: Optional[str] = None,
) -> Tuple[List[List[Dict[str, str]]], List[int]]:
    """Batch-optimize contexts and return messages in scheduled order.

    Globally reorders all contexts and determines the optimal execution
    order for maximum prefix sharing across the batch.

    Example::

        import contextpilot as cp
        messages_batch, order = cp.optimize_batch(all_docs, all_queries)
        for messages, orig_idx in zip(messages_batch, order):
            resp = client.chat.completions.create(
                model="Qwen/Qwen3-4B", messages=messages
            )

    Args:
        all_docs: ``List[List[str]]`` — documents for each query.
        all_queries: ``List[str]`` — one query per entry in *all_docs*.
        system_instruction: Optional extra instruction prepended to every
            system message.

    Returns:
        ``(messages_batch, original_indices)`` where *messages_batch[i]*
        corresponds to the original query at ``all_queries[original_indices[i]]``.
    """
    if len(all_docs) != len(all_queries):
        raise ValueError(
            f"all_docs ({len(all_docs)}) and all_queries "
            f"({len(all_queries)}) must have the same length."
        )

    # Use a fresh instance for batch mode to avoid cross-contamination
    # with the singleton used by optimize() (which tracks conversation state).
    from .server.live_index import ContextPilot
    pilot = ContextPilot(use_gpu=False)
    reordered_contexts, order = pilot.reorder(all_docs)

    messages_batch = []
    for ctx, orig_idx in zip(reordered_contexts, order):
        system_content = _build_system_prompt(
            ctx, all_docs[orig_idx], system_instruction
        )
        messages_batch.append([
            {"role": "system", "content": system_content},
            {"role": "user", "content": all_queries[orig_idx]},
        ])

    return messages_batch, order
