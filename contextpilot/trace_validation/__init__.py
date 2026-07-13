"""Trace-derived validation-set framework for ContextPilot.

A fixed, replayable corpus + a gate runner so any future accuracy-affecting or
runtime-payload-changing change can be checked against stable Hermes traces
instead of an ad-hoc "run once and see".

Two pieces, with a deliberate privacy split:

* :mod:`.builder` reads a local Hermes SQLite DB (read-only) and exports a fixed
  JSONL corpus under a local, gitignored directory. Raw content lives ONLY in
  that local artifact; the sidecar manifest is privacy-safe.
* :mod:`.runner` replays the corpus through ContextPilot's optimization in a
  baseline (``off``) mode vs a configured candidate mode and checks
  accuracy-preservation invariants, emitting a privacy-safe pass/fail report.

Both reuse the analyzer's read-only DB loaders, salted hashing, and forbidden-key
guard so the privacy primitives stay defined in one place.
"""
from __future__ import annotations

from .builder import (
    DEFAULT_OUT_DIR,
    DEFAULT_SALT,
    DEFAULT_STATE_DB,
    build_manifest,
    case_to_json,
    load_trace_cases,
    write_validation_set,
)
from .builder import main as build_main
from .models import (
    DEFAULT_CASE_LIMIT,
    DEFAULT_MIN_INPUT_TOKENS,
    MUTABLE_BLOCK_TYPE,
    VALIDATION_SET_SCHEMA_VERSION,
    TraceCase,
    TraceMessage,
    ValidationCaseResult,
    ValidationReport,
)
from .runner import (
    INVARIANT_NAMES,
    assert_report_privacy_safe,
    check_invariants,
    load_cases,
    optimize_case,
    render_markdown,
    report_to_dict,
    run_validation,
)
from .runner import main as run_main

__all__ = [
    # models / constants
    "VALIDATION_SET_SCHEMA_VERSION",
    "DEFAULT_CASE_LIMIT",
    "DEFAULT_MIN_INPUT_TOKENS",
    "MUTABLE_BLOCK_TYPE",
    "INVARIANT_NAMES",
    "TraceMessage",
    "TraceCase",
    "ValidationCaseResult",
    "ValidationReport",
    # builder
    "DEFAULT_STATE_DB",
    "DEFAULT_OUT_DIR",
    "DEFAULT_SALT",
    "load_trace_cases",
    "case_to_json",
    "build_manifest",
    "write_validation_set",
    "build_main",
    # runner
    "load_cases",
    "optimize_case",
    "check_invariants",
    "run_validation",
    "report_to_dict",
    "render_markdown",
    "assert_report_privacy_safe",
    "run_main",
]
