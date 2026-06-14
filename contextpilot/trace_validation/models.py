"""Data structures for the trace-derived validation-set framework.

Two layers, with a deliberate privacy split:

* :class:`TraceCase` / :class:`TraceMessage` are the *local* corpus carriers.
  They DO hold raw LLM-bound text (``content``) because the validation set must
  be able to replay the exact payload ContextPilot would process. They are only
  ever serialized into the gitignored local artifact produced by the builder --
  never into a committed fixture or a runner report.
* :class:`ValidationCaseResult` / :class:`ValidationReport` are the *report*
  carriers. They are privacy-safe by construction: salted case ids, integer
  counters, low-cardinality enums and pass/fail booleans only -- never raw
  prompt/message/tool text.

The runner's report is additionally passed through the analyzer's
``_assert_no_forbidden_keys`` guard and a raw-substring scan before it is
emitted, so a regression that accidentally threads content into a report fails
loudly instead of leaking.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Bumped when the on-disk JSONL case schema changes in a non-additive way.
VALIDATION_SET_SCHEMA_VERSION = 1

# Conservative sampling defaults: the builder is meant to capture a small,
# representative corpus, not exfiltrate a whole history.
DEFAULT_CASE_LIMIT = 25
DEFAULT_SINCE_HOURS = 24
DEFAULT_MIN_INPUT_TOKENS = 0
DEFAULT_MIN_MESSAGES = 1

# The only block type the prompt-dedup canary is ever allowed to mutate. Every
# other block type is "protected" and must survive optimization byte-identical.
MUTABLE_BLOCK_TYPE = "skill_prompt"


@dataclass
class TraceMessage:
    """One ordered LLM-bound message in a replayed case.

    ``content`` is RAW text and is local-only: it appears in the gitignored
    corpus artifact, never in a committed fixture or a runner report.
    """

    role: str | None
    block_type: str
    content: str


@dataclass
class TraceCase:
    """A single replayable case derived from one Hermes session.

    ``case_id`` is a salted hash of the session id (never the raw id). The
    counters are privacy-safe; ``messages`` carries raw content and is local-only.
    """

    case_id: str
    source: str | None
    input_tokens: int
    message_count: int
    messages: list[TraceMessage]


@dataclass
class ValidationCaseResult:
    """Privacy-safe per-case outcome of a validation run.

    Salted id + counters + enums + invariant booleans only. ``chars_saved`` and
    ``blocks_replaced`` are REALIZED processed-payload figures (actual before/
    after character delta), not opportunity counts. Token figures are populated
    only when an exact tokenizer backend was configured.
    """

    case_id: str
    source: str | None
    message_count: int
    skill_item_count: int
    mutated: bool
    blocks_replaced: int
    chars_saved: int                      # REALIZED before-after char delta
    invariants: dict[str, bool]
    passed: bool
    failed_invariants: list[str] = field(default_factory=list)
    actual_tokens_before: int | None = None
    actual_tokens_after: int | None = None
    actual_tokens_saved: int | None = None


@dataclass
class ValidationReport:
    """Privacy-safe summary of a whole validation run (gate + accounting)."""

    schema_version: int
    generated_date: str
    salt_fingerprint: str
    baseline_mode: str
    candidate_mode: str
    case_count: int
    passed: bool                          # overall gate
    passed_cases: int
    failed_cases: int
    total_blocks_replaced: int
    total_chars_saved: int                # REALIZED before-after char delta
    tokenizer_status: str                 # "available" | "unavailable"
    tokenizer_backend: str | None
    total_actual_tokens_saved: int | None
    invariant_names: list[str]
    cases: list[ValidationCaseResult]
    notes: list[str] = field(default_factory=list)
