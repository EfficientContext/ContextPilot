"""Privacy-safe Hermes context opportunity analyzer for ContextPilot.

Unlike ``hermes_contextpilot_monitor.py`` (which never reads message bodies),
this analyzer *does* inspect message content and tool outputs in order to find
concrete token-reduction opportunities: exact duplicate tool outputs, repeated
line/block fingerprints, oversized tool outputs per tool, heavy sessions, and
ContextPilot telemetry coverage.

It reads content only in-memory to compute salted hashes and aggregate
counters. Reports never contain raw message/tool text, system prompts, or raw
session ids -- only salted SHA-256 fingerprints and numeric aggregates. This
makes it safe to run continuously from a cron job and ship the reports.

The package is split into focused modules:

* :mod:`.models`      -- privacy-safe dataclasses, tunables, in-memory carriers
* :mod:`.privacy`     -- salted hashing + the forbidden-output guard
* :mod:`.db`          -- read-only Hermes state-DB loaders
* :mod:`.telemetry`   -- metadata-only ContextPilot telemetry parsing
* :mod:`.detection`   -- content-aware redundancy detection
* :mod:`.dedup_ab`    -- offline prompt-dedup A/B simulation (no mutation)
* :mod:`.prompt_dedup_canary` -- default-off runtime canary, skill-only
* :mod:`.routing`     -- Worker Context Routing shadow mode (P0)
* :mod:`.aggregation` -- Parent Aggregation Artifacts shadow mode (P0)
* :mod:`.report`      -- report assembly + serialization
* :mod:`.cli`         -- command-line entry point

Safety contract: everything here is reporting/measurement only **except**
:mod:`.prompt_dedup_canary`, which is the single default-off runtime mutation
path. That canary is limited to same-type skill-prompt exact duplicates and only
runs when explicitly enabled by environment variable.
"""
from __future__ import annotations

from .aggregation import (
    ARTIFACT_KINDS,
    DEFAULT_MIN_ARTIFACT_CHARS,
    PARENT_AGGREGATION_SOURCE_TYPES,
    analyze_parent_aggregation_artifacts,
    classify_artifact_kind,
)
from .cli import main
from .dedup_ab import simulate_prompt_dedup_ab
from .db import (
    load_heavy_sessions,
    load_llm_bound_content,
    load_tool_messages,
    total_input_tokens,
)
from .detection import (
    analyze_llm_bound_blocks,
    detect_exact_duplicate_tool_outputs,
    detect_prompt_duplicate_blocks,
    detect_repeated_blocks,
    summarize_tool_sizes,
)
from .models import (
    BLOCK_TYPES,
    DEFAULT_LARGE_OUTPUT_CHARS,
    DEFAULT_MIN_BLOCK_CHARS,
    DEFAULT_MIN_BLOCK_REPEAT,
    DEFAULT_TOP_N,
    EST_CHARS_PER_TOKEN,
    PROMPT_DEDUP_AB_CLASSES,
    PROMPT_DEDUP_AB_REFERENCE_TEMPLATE,
    PROMPT_DUPLICATE_BLOCK_TYPES,
    ArtifactKindStat,
    ArtifactSourceCount,
    BlockTypeStat,
    CrossTypeBlockGroup,
    DuplicateToolOutput,
    HeavySession,
    OpportunityReport,
    ParentAggregationArtifacts,
    ParentAggregationGroup,
    PromptDedupABClass,
    PromptDedupABSimulation,
    PromptDuplicateBlock,
    PromptDuplicateShadow,
    PromptDuplicateTypeCount,
    RepeatedBlock,
    RouterCandidateBlock,
    RouterLabelCount,
    RouterReasonCount,
    TelemetryCoverage,
    ToolSizeStat,
    TypeCount,
    WorkerRoutingShadow,
    _est_tokens,
    _LLMContent,
    _ToolMessage,
)
from .privacy import (
    FORBIDDEN_OUTPUT_KEYS,
    _assert_no_forbidden_keys,
    _salt_fingerprint,
    _salted_hash,
)
from .prompt_dedup_canary import (
    CANARY_DEDUP_CLASS,
    DEFAULT_PROMPT_DEDUP_MODE,
    PROMPT_DEDUP_CANARY_REFERENCE_TEMPLATE,
    PROMPT_DEDUP_DISABLE_ENV,
    PROMPT_DEDUP_MODE_ENV,
    PROMPT_DEDUP_MODES,
    SAFETY_DENYLIST,
    PromptDedupCanaryResult,
    apply_prompt_dedup_canary,
    build_canary_telemetry_record,
    resolve_prompt_dedup_mode,
)
from .report import build_report, write_report
from .routing import (
    ROUTER_LABELS,
    _ROUTABLE_LABELS,
    analyze_worker_routing_shadow,
    classify_router_label,
)
from .telemetry import parse_telemetry
from .tokenizer import TokenizerBackend, resolve_tokenizer

__all__ = [
    # tunables / enums
    "DEFAULT_MIN_BLOCK_CHARS",
    "DEFAULT_MIN_BLOCK_REPEAT",
    "DEFAULT_LARGE_OUTPUT_CHARS",
    "DEFAULT_TOP_N",
    "DEFAULT_MIN_ARTIFACT_CHARS",
    "EST_CHARS_PER_TOKEN",
    "BLOCK_TYPES",
    "PROMPT_DUPLICATE_BLOCK_TYPES",
    "ROUTER_LABELS",
    "ARTIFACT_KINDS",
    "PARENT_AGGREGATION_SOURCE_TYPES",
    "FORBIDDEN_OUTPUT_KEYS",
    # prompt-dedup canary (runtime; default off)
    "PROMPT_DEDUP_MODE_ENV",
    "PROMPT_DEDUP_DISABLE_ENV",
    "PROMPT_DEDUP_MODES",
    "DEFAULT_PROMPT_DEDUP_MODE",
    "CANARY_DEDUP_CLASS",
    "PROMPT_DEDUP_CANARY_REFERENCE_TEMPLATE",
    "SAFETY_DENYLIST",
    # dataclasses
    "DuplicateToolOutput",
    "RepeatedBlock",
    "TypeCount",
    "BlockTypeStat",
    "CrossTypeBlockGroup",
    "ToolSizeStat",
    "HeavySession",
    "TelemetryCoverage",
    "PromptDedupABClass",
    "PromptDedupABSimulation",
    "PromptDuplicateBlock",
    "PromptDuplicateTypeCount",
    "PromptDuplicateShadow",
    "RouterLabelCount",
    "RouterReasonCount",
    "RouterCandidateBlock",
    "WorkerRoutingShadow",
    "ArtifactSourceCount",
    "ParentAggregationGroup",
    "ArtifactKindStat",
    "ParentAggregationArtifacts",
    "OpportunityReport",
    # loaders
    "load_tool_messages",
    "load_llm_bound_content",
    "load_heavy_sessions",
    "total_input_tokens",
    # telemetry
    "parse_telemetry",
    # detection
    "detect_exact_duplicate_tool_outputs",
    "detect_repeated_blocks",
    "summarize_tool_sizes",
    "analyze_llm_bound_blocks",
    "detect_prompt_duplicate_blocks",
    "simulate_prompt_dedup_ab",
    "TokenizerBackend",
    "resolve_tokenizer",
    # prompt-dedup canary (runtime; default off)
    "PromptDedupCanaryResult",
    "resolve_prompt_dedup_mode",
    "apply_prompt_dedup_canary",
    "build_canary_telemetry_record",
    # routing (shadow)
    "classify_router_label",
    "analyze_worker_routing_shadow",
    # aggregation (shadow)
    "classify_artifact_kind",
    "analyze_parent_aggregation_artifacts",
    # report
    "build_report",
    "write_report",
    # cli
    "main",
]
