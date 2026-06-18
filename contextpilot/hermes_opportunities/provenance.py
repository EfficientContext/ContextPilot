"""Privacy-safe provenance profiling -- the standalone token-monitor view.

``build_provenance_profile`` rolls a list of :class:`HeavySession` rows up by
their provenance ``source`` into numeric token/usage aggregates. It is the
lightweight "where are my tokens going?" report: no SciPy/RAG machinery, no
content, prompts, reasoning, or raw session ids/hashes -- only low-cardinality
source labels and integer counters.
"""
from __future__ import annotations

from collections.abc import Iterable

from .models import (
    UNKNOWN_SOURCE,
    HeavySession,
    ProvenanceProfile,
    ProvenanceSourceStat,
)


def build_provenance_profile(
    heavy_sessions: Iterable[HeavySession],
) -> ProvenanceProfile:
    """Aggregate heavy sessions into a per-source token-usage profile.

    Sessions with no recorded ``source`` are folded into the ``"unknown"``
    bucket so the output stays a low-cardinality enum view. ``by_source`` rows
    are returned sorted by descending total tokens (then source name) for stable,
    human-meaningful ordering.
    """
    buckets: dict[str, ProvenanceSourceStat] = {}
    for session in heavy_sessions:
        source = session.source or UNKNOWN_SOURCE
        stat = buckets.get(source)
        if stat is None:
            stat = ProvenanceSourceStat(
                source=source,
                session_count=0,
                input_tokens=0,
                output_tokens=0,
                message_count=0,
                tool_call_count=0,
                api_call_count=0,
                total_tokens=0,
            )
            buckets[source] = stat
        stat.session_count += 1
        stat.input_tokens += session.input_tokens
        stat.output_tokens += session.output_tokens
        stat.message_count += session.message_count
        stat.tool_call_count += session.tool_call_count
        stat.api_call_count += session.api_call_count
        stat.total_tokens += session.input_tokens + session.output_tokens

    by_source = sorted(
        buckets.values(), key=lambda s: (-s.total_tokens, s.source)
    )
    return ProvenanceProfile(
        source_count=len(by_source),
        session_count=sum(s.session_count for s in by_source),
        input_tokens=sum(s.input_tokens for s in by_source),
        output_tokens=sum(s.output_tokens for s in by_source),
        total_tokens=sum(s.total_tokens for s in by_source),
        by_source=by_source,
    )
