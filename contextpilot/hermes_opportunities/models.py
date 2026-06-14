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
# Prompt duplicate — SHADOW MODE structures (system/skill prompts only)
# ---------------------------------------------------------------------------

# Block types this advisory section is allowed to scan. Static prompt text only;
# never user/assistant/tool message bodies.
PROMPT_DUPLICATE_BLOCK_TYPES = ("system_prompt", "skill_prompt")


@dataclass
class PromptDuplicateBlock:
    """One exact block fingerprint seen 2+ times in system/skill prompt text.

    Salted hash + counters only -- never the block text. Char figures are ACTUAL
    duplicated characters; the token figure is an ADVISORY chars/4 estimate.
    """

    block_hash: str
    block_types: list[str]                       # which prompt types this block spans
    occurrences: int
    char_length: int
    chars_duplicated: int                        # ACTUAL: (occurrences - 1) * char_length
    advisory_est_duplicate_tokens_chars_div_4: int  # ADVISORY only, NOT actual tokens


@dataclass
class PromptDuplicateTypeCount:
    """Per-prompt-type occurrence rollup for duplicate blocks."""

    block_type: str              # system_prompt | skill_prompt
    duplicate_block_count: int   # distinct duplicate fingerprints touching this type
    occurrence_count: int        # total occurrences within this type
    chars_duplicated: int        # ACTUAL duplicated chars attributable within this type


@dataclass
class PromptDuplicateShadow:
    """Advisory report of exact duplicate blocks in system/skill prompts.

    SHADOW/ADVISORY ONLY: this measures static prompt duplication; it never
    rewrites, dedups, or otherwise mutates prompts, and its char/token figures
    must never be reported as realized savings.
    """

    enabled: bool
    item_count: int                  # system/skill prompt items scanned
    scanned_block_types: list[str]
    duplicate_group_count: int
    total_duplicate_occurrences: int
    total_chars_duplicated: int      # ACTUAL duplicated chars (advisory, not realized)
    advisory_est_duplicate_tokens_chars_div_4: int  # ADVISORY chars/4, NOT actual tokens
    by_block_type: list[PromptDuplicateTypeCount]
    top_duplicate_blocks: list[PromptDuplicateBlock]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt Dedup A/B — OFFLINE SIMULATION structures (system/skill prompts only)
# ---------------------------------------------------------------------------

# The reference placeholder a *simulated* replacement would leave in place of a
# later duplicate occurrence. Used for accounting only -- ContextPilot never
# emits this string into a real payload. ``<type>`` / ``<hash>`` are filled with
# the canonical prompt type and the salted block fingerprint.
PROMPT_DEDUP_AB_REFERENCE_TEMPLATE = (
    "[Prompt duplicate omitted in simulation; canonical=<type>:<hash>]"
)

# Safe candidate classes, simulated separately. The skill-only class is the
# lowest-risk first canary candidate; the others are reported but higher risk.
PROMPT_DEDUP_AB_CLASSES = (
    "same_type_skill_prompt_only",
    "same_type_system_prompt_only",
    "cross_type_system_skill",
)


@dataclass
class PromptDedupABClass:
    """Simulated A/B accounting for one candidate class.

    All figures are OFFLINE SIMULATION over static system/skill prompt text and
    are NOT realized savings -- ContextPilot performs no replacement at runtime.
    ``chars_delta_simulated`` is signed: positive means the simulated reference
    replacement would shrink the payload, negative means it would grow it (a
    short duplicate replaced by a longer placeholder).

    Actual-token fields are populated ONLY when an exact tokenizer backend is
    configured; otherwise they are ``None`` and ``tokenizer_status`` is
    ``"unavailable"`` -- never a fabricated chars/4 figure.
    """

    candidate_class: str
    risk_label: str                       # "low" (canary candidate) | "high"
    candidate_group_count: int            # distinct exact-duplicate block groups
    replacement_occurrence_count: int     # occurrences beyond the first, summed
    chars_before: int                     # chars of all candidate occurrences
    chars_after_simulated: int            # first kept full, later -> reference str
    chars_delta_simulated: int            # chars_before - chars_after_simulated
    tokenizer_status: str                 # "available" | "unavailable"
    actual_tokens_before: int | None      # only when tokenizer available
    actual_tokens_after: int | None       # only when tokenizer available
    actual_tokens_delta: int | None       # only when tokenizer available
    note: str


@dataclass
class PromptDedupABSimulation:
    """Offline A/B simulation harness for prompt dedup (system/skill prompts).

    OFFLINE SIMULATION + MEASUREMENT ONLY. This is the evidence gate to evaluate
    *before* any canary replacement: it scans only ``system_prompt`` /
    ``skill_prompt`` LLM-bound blocks, keeps the first occurrence of every exact
    duplicate and replaces only later occurrences in a *simulated* accounting. It
    never mutates the DB, runtime, or emitted prompts, and its char/token deltas
    are NOT realized savings.
    """

    enabled: bool
    item_count: int                       # system/skill prompt items scanned
    scanned_block_types: list[str]
    tokenizer_status: str                 # "available" | "unavailable"
    tokenizer_backend: str | None         # backend name when available, else None
    reference_string_template: str
    classes: list[PromptDedupABClass]
    notes: list[str] = field(default_factory=list)


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
    # Prompt duplicate shadow (system/skill prompts only; advisory, never realized).
    prompt_duplicates: PromptDuplicateShadow
    # Prompt dedup A/B simulation (system/skill prompts only; offline, never realized).
    prompt_dedup_ab: PromptDedupABSimulation
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
