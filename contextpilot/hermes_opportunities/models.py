"""Privacy-safe data structures and shared tunables for the analyzer.

Every dataclass here is privacy-safe by construction: it carries only salted
hashes, numeric counters, and low-cardinality enums -- never raw message/tool/
system/skill text. The in-memory carriers ``_ToolMessage`` and ``_LLMContent``
DO hold content, but exclusively for in-process hashing; their ``content`` must
never be emitted to a report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Tunables (overridable via CLI).
DEFAULT_MIN_BLOCK_CHARS = 40       # ignore trivial lines when fingerprinting
DEFAULT_MIN_BLOCK_REPEAT = 3       # a block must recur this often to be a "repeat"
DEFAULT_LARGE_OUTPUT_CHARS = 8000  # tool outputs at/above this are "large"
DEFAULT_TOP_N = 20
EST_CHARS_PER_TOKEN = 4


def _est_tokens(chars: int) -> int:
    return chars // EST_CHARS_PER_TOKEN


# Recognized LLM-bound block types. These are low-cardinality enums, safe to
# emit verbatim (they describe the *origin* of a block, never its text).
BLOCK_TYPES = (
    "system_prompt",
    "skill_prompt",
    "user_prompt",
    "assistant_context",
    "tool_result",
    "unknown",
)


# ---------------------------------------------------------------------------
# Tool-output redundancy structures (all privacy-safe: hashes + counters only)
# ---------------------------------------------------------------------------


@dataclass
class DuplicateToolOutput:
    content_hash: str
    tool_name: str | None
    occurrences: int
    char_length: int
    est_tokens: int
    est_wasted_tokens: int  # tokens spent re-sending identical output: (n-1) * est_tokens


@dataclass
class RepeatedBlock:
    block_hash: str
    occurrences: int
    char_length: int
    est_tokens: int
    est_wasted_tokens: int  # (n-1) * est_tokens


@dataclass
class TypeCount:
    block_type: str
    count: int


@dataclass
class BlockTypeStat:
    """Aggregate redundancy within a single LLM-bound block type."""

    block_type: str
    item_count: int            # source items (prompts/messages) of this type
    block_count: int           # total fingerprintable block instances
    unique_block_count: int    # distinct fingerprints
    repeated_block_count: int  # fingerprints recurring >= min_repeat within type
    est_redundant_tokens: int  # sum over repeats of (occ-1) * est_tokens


@dataclass
class CrossTypeBlockGroup:
    """A single block fingerprint observed in 2+ distinct block types.

    This is the headline signal: the same chunk of text is being shipped to the
    LLM from, e.g., a skill/system prompt *and* a tool result, so it is paying
    for the same tokens twice from different sources.
    """

    block_hash: str
    block_types: list[str]               # sorted distinct types this block spans
    type_occurrences: list[TypeCount]    # per-type occurrence counts
    occurrences: int                     # total occurrences across all types
    char_length: int
    est_tokens: int
    est_wasted_tokens: int               # (occurrences - 1) * est_tokens


@dataclass
class ToolSizeStat:
    tool_name: str
    output_count: int
    total_chars: int
    max_chars: int
    avg_chars: int
    total_est_tokens: int
    large_output_count: int  # outputs >= large_output_chars threshold


@dataclass
class HeavySession:
    session_hash: str
    source: str | None
    input_tokens: int
    output_tokens: int
    message_count: int
    tool_call_count: int
    api_call_count: int


@dataclass
class TelemetryCoverage:
    events: int
    chars_saved: int
    tokens_saved: int                 # legacy derived counter (chars_saved // 4), not tokenizer/API usage
    avg_tokens_saved_per_event: float # derived from the legacy chars/4 counter
    coverage_ratio_pct: float         # derived ratio using the legacy chars/4 counter
    malformed_records_skipped: int


# ---------------------------------------------------------------------------
# Worker Context Routing — SHADOW MODE structures (P0 data collection only)
# ---------------------------------------------------------------------------


@dataclass
class RouterLabelCount:
    """Aggregate over all blocks assigned one router label."""

    route_label: str
    block_count: int            # distinct fingerprints with this label
    occurrence_count: int       # total occurrences across the window
    total_est_tokens: int       # est tokens these blocks occupy (occ * est)
    est_candidate_tokens: int   # ADVISORY routable tokens (0 unless routable)


@dataclass
class RouterReasonCount:
    """Aggregate keyed by (block_type, route_label, reason_code)."""

    block_type: str
    route_label: str
    reason_code: str
    block_count: int
    occurrence_count: int
    total_est_tokens: int
    est_candidate_tokens: int


@dataclass
class RouterCandidateBlock:
    """A single routable-candidate fingerprint (salted hash + counters only)."""

    block_hash: str
    block_type: str
    route_label: str
    reason_code: str
    occurrences: int
    char_length: int
    est_tokens: int
    est_candidate_tokens: int   # ADVISORY upper bound only


@dataclass
class WorkerRoutingShadow:
    """Shadow-mode worker-context routing report (P0: data collection only)."""

    enabled: bool
    item_count: int                 # LLM-bound items classified
    classified_block_count: int     # distinct fingerprints classified
    total_occurrences: int
    must_keep_block_count: int
    must_keep_occurrence_count: int
    est_must_keep_tokens: int
    est_candidate_tokens_total: int          # ADVISORY routable ceiling
    est_drop_candidate_tokens: int           # ADVISORY
    est_summarizable_candidate_tokens: int   # ADVISORY
    label_counts: list[RouterLabelCount]
    reason_counts: list[RouterReasonCount]
    top_candidate_blocks: list[RouterCandidateBlock]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parent Aggregation Artifacts — SHADOW MODE structures (P0 telemetry only)
# ---------------------------------------------------------------------------


@dataclass
class ArtifactSourceCount:
    """Provenance counter: occurrences of one artifact body from one source."""

    source_type: str
    count: int


@dataclass
class ParentAggregationGroup:
    """One EXACT artifact body observed 2+ times across parent/worker contexts.

    Salted hash + counters only -- never the body text.
    """

    content_hash: str
    artifact_kind: str
    canonical_source_type: str           # dominant origin, chosen deterministically
    occurrences: int
    char_length: int
    est_tokens: int
    est_duplicate_tokens: int            # ADVISORY: (occurrences - 1) * est_tokens
    source_type_counts: list[ArtifactSourceCount]  # provenance: tool_result xN, ...


@dataclass
class ArtifactKindStat:
    """Aggregate over all candidate artifact bodies of one kind."""

    artifact_kind: str
    group_count: int             # distinct bodies of this kind
    occurrence_count: int        # total occurrences of those bodies
    duplicate_group_count: int   # bodies seen >= 2 times
    est_tokens: int              # sum of est tokens for distinct bodies
    est_duplicate_tokens: int    # ADVISORY duplicate tokens for this kind


@dataclass
class ParentAggregationArtifacts:
    """Shadow-mode parent-aggregation artifact report (P0: telemetry only)."""

    enabled: bool
    item_count: int                  # candidate artifact items considered
    artifact_body_count: int         # distinct bodies (groups)
    total_occurrences: int
    duplicate_group_count: int
    est_total_tokens: int            # est tokens for distinct bodies
    est_duplicate_tokens: int        # ADVISORY duplicate-artifact tokens
    by_kind: list[ArtifactKindStat]
    source_type_counts: list[ArtifactSourceCount]   # provenance across candidates
    top_duplicate_groups: list[ParentAggregationGroup]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


@dataclass
class OpportunityReport:
    date: str
    since_hours: int
    all_sessions: bool
    salt_fingerprint: str
    tool_message_count: int
    total_tool_output_chars: int
    total_tool_output_est_tokens: int
    exact_duplicate_groups: list[DuplicateToolOutput]
    duplicate_tool_output_groups: int
    duplicate_tool_output_wasted_tokens: int
    repeated_block_count: int
    repeated_block_wasted_tokens: int
    repeated_blocks: list[RepeatedBlock]
    large_tool_outputs_by_tool: list[ToolSizeStat]
    heavy_sessions: list[HeavySession]
    telemetry: TelemetryCoverage
    # LLM-bound block analysis (system/skill prompts, prompts, tool results).
    llm_bound_item_count: int
    llm_block_types: list[BlockTypeStat]
    cross_type_block_groups: list[CrossTypeBlockGroup]
    cross_type_wasted_tokens: int
    # Worker Context Routing shadow mode (P0 data collection; never prunes).
    worker_routing: WorkerRoutingShadow
    # Parent Aggregation Artifacts shadow mode (P0 telemetry; never dedups).
    parent_aggregation: ParentAggregationArtifacts
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# In-memory content carriers (hold raw text for hashing ONLY; never emitted)
# ---------------------------------------------------------------------------


@dataclass
class _ToolMessage:
    tool_name: str | None
    content: str


@dataclass
class _LLMContent:
    """A chunk of content that Hermes would actually send to the LLM.

    Held in-memory only for hashing; ``content`` must never be emitted.
    """

    block_type: str
    content: str
