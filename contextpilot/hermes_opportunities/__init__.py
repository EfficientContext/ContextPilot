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
* :mod:`.routing`     -- Worker Context Routing shadow mode (P0)
* :mod:`.aggregation` -- Parent Aggregation Artifacts shadow mode (P0)
* :mod:`.report`      -- report assembly + serialization
* :mod:`.cli`         -- command-line entry point

Everything here is reporting/measurement only: no module ever replaces,
summarizes, routes, or otherwise mutates context at runtime.
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
from .db import (
    load_heavy_sessions,
    load_llm_bound_content,
    load_tool_messages,
    total_input_tokens,
)
from .detection import (
    analyze_llm_bound_blocks,
    detect_exact_duplicate_tool_outputs,
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
    ArtifactKindStat,
    ArtifactSourceCount,
    BlockTypeStat,
    CrossTypeBlockGroup,
    DuplicateToolOutput,
    HeavySession,
    OpportunityReport,
    ParentAggregationArtifacts,
    ParentAggregationGroup,
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
from .report import build_report, write_report
from .routing import (
    ROUTER_LABELS,
    _ROUTABLE_LABELS,
    analyze_worker_routing_shadow,
    classify_router_label,
)
from .telemetry import parse_telemetry

__all__ = [
    # tunables / enums
    "DEFAULT_MIN_BLOCK_CHARS",
    "DEFAULT_MIN_BLOCK_REPEAT",
    "DEFAULT_LARGE_OUTPUT_CHARS",
    "DEFAULT_TOP_N",
    "DEFAULT_MIN_ARTIFACT_CHARS",
    "EST_CHARS_PER_TOKEN",
    "BLOCK_TYPES",
    "ROUTER_LABELS",
    "ARTIFACT_KINDS",
    "PARENT_AGGREGATION_SOURCE_TYPES",
    "FORBIDDEN_OUTPUT_KEYS",
    # dataclasses
    "DuplicateToolOutput",
    "RepeatedBlock",
    "TypeCount",
    "BlockTypeStat",
    "CrossTypeBlockGroup",
    "ToolSizeStat",
    "HeavySession",
    "TelemetryCoverage",
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
