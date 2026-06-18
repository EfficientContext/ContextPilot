"""Trace validation runner: gate accuracy-preservation against fixed cases.

Loads the local JSONL corpus produced by :mod:`.builder` and, for every case,
runs ContextPilot's optimization in two controlled modes:

* a **baseline** (``off``) pass, which must leave the payload byte-identical, and
* a **candidate** pass, whose mode is taken from the configured environment
  (``CONTEXTPILOT_PROMPT_DEDUP_MODE``) or overridden on the command line.

It then checks accuracy-preservation invariants between the two payloads --
message count, order, and roles preserved; protected (non-skill) content
byte-identical; mutation confined to the allowed scope and never growing the
payload; and realized-savings accounting consistent. The realized ``chars_saved``
is the ACTUAL processed-payload before/after character delta, never an
opportunity count. Exact-token figures appear only when a tokenizer backend is
configured; otherwise the status is ``unavailable`` and no token fields are set.

The emitted report is privacy-safe: salted ids, counters, enums, and pass/fail
only. It is passed through the analyzer's forbidden-key guard and a raw-content
substring scan before it is printed, so a regression cannot leak raw text. The
process exits non-zero on any gate failure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Callable

from contextpilot.hermes_opportunities.models import DEFAULT_MIN_BLOCK_CHARS, _LLMContent
from contextpilot.hermes_opportunities.privacy import (
    _assert_no_forbidden_keys,
    _salt_fingerprint,
)
from contextpilot.hermes_opportunities.prompt_dedup_canary import (
    PROMPT_DEDUP_CANARY_REFERENCE_TEMPLATE,
    PromptDedupCanaryResult,
    apply_prompt_dedup_canary,
    resolve_prompt_dedup_mode,
)
from contextpilot.hermes_opportunities.artifact_dedup_canary import (
    MUTABLE_ARTIFACT_BLOCK_TYPES,
    ArtifactDedupCanaryResult,
    ArtifactSpanLink,
    _parse_artifact_reference,
    _segment_fenced_blocks,
    apply_artifact_dedup_canary,
    dangling_artifact_references,
    resolve_artifact_dedup_mode,
)
from contextpilot.hermes_opportunities.tokenizer import resolve_tokenizer

from .builder import DEFAULT_SALT
from .models import (
    MUTABLE_BLOCK_TYPE,
    VALIDATION_SET_SCHEMA_VERSION,
    ValidationCaseResult,
    ValidationReport,
)

# Stable invariant identifiers, also used as the report's gate vocabulary.
INVARIANT_NAMES = [
    "message_count_preserved",
    "order_and_roles_preserved",
    "protected_content_preserved",
    "mutation_scope_allowed",
    "savings_accounting_consistent",
]

# The fixed prefix/needle of the canary's reference string. A mutated skill line
# must equal a string of this shape -- anything else means the optimizer emitted
# unexpected (possibly raw) text, which is a gate failure.
_REF_PREFIX = PROMPT_DEDUP_CANARY_REFERENCE_TEMPLATE.split("<type>", 1)[0]  # "[...ref="
_REF_NEEDLE = f"ref={MUTABLE_BLOCK_TYPE}:"


def load_cases(corpus_path: Path) -> list[dict]:
    """Load the JSONL corpus into a list of case dicts (raw content in-memory).

    Tolerates blank lines; raises on malformed JSON so a corrupt corpus fails
    loudly rather than silently validating a partial set.
    """
    cases: list[dict] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def _messages(case: dict) -> list[dict]:
    return [
        {
            "role": m.get("role"),
            "block_type": m.get("block_type", "unknown"),
            "content": m.get("content", ""),
        }
        for m in case.get("messages", [])
    ]


def optimize_case(
    messages: list[dict], *, mode: str, salt: str, min_block_chars: int
) -> tuple[list[dict], PromptDedupCanaryResult]:
    """Run the prompt-dedup canary over a case's messages in the given mode.

    Returns ``(out_messages, result)``. The canary mutates only ``skill_prompt``
    content in place; ``out_messages`` mirrors the input role/block_type/order
    with the (possibly) rewritten content so the caller can diff payloads.
    """
    contents = [_LLMContent(m["block_type"], m["content"]) for m in messages]
    result = apply_prompt_dedup_canary(
        contents, salt=salt, min_block_chars=min_block_chars, mode=mode
    )
    out = [
        {"role": m["role"], "block_type": m["block_type"], "content": c.content}
        for m, c in zip(messages, contents)
    ]
    return out, result


def _is_reference_line(line: str) -> bool:
    """True if a line is a canary reference placeholder (no raw content)."""
    return line.startswith(_REF_PREFIX) and _REF_NEEDLE in line and line.endswith("]")


def _mutation_scope_ok(base: dict, cand: dict) -> bool:
    """A single message changed only within the allowed (skill-only) scope."""
    if base["content"] == cand["content"]:
        return True
    # Only skill_prompt content may ever change.
    if base["block_type"] != MUTABLE_BLOCK_TYPE:
        return False
    # Never grow the payload.
    if len(cand["content"]) > len(base["content"]):
        return False
    base_lines = base["content"].split("\n")
    cand_lines = cand["content"].split("\n")
    # The canary replaces lines 1:1; a differing line count is out of scope.
    if len(base_lines) != len(cand_lines):
        return False
    for b, c in zip(base_lines, cand_lines):
        if b == c:
            continue
        # A changed line must be a reference placeholder strictly shorter than
        # what it replaced -- never new free text and never a growth.
        if not (_is_reference_line(c) and len(c) < len(b)):
            return False
    return True


def check_invariants(
    baseline: list[dict], candidate: list[dict], result: PromptDedupCanaryResult
) -> tuple[dict[str, bool], int]:
    """Check accuracy-preservation invariants between two payloads.

    Returns ``(invariant -> passed, realized_chars_saved)`` where
    ``realized_chars_saved`` is the ACTUAL summed before/after character delta of
    the processed payload (not an opportunity count).
    """
    inv: dict[str, bool] = {}

    inv["message_count_preserved"] = len(baseline) == len(candidate)

    if inv["message_count_preserved"]:
        inv["order_and_roles_preserved"] = all(
            b["role"] == c["role"] and b["block_type"] == c["block_type"]
            for b, c in zip(baseline, candidate)
        )
        inv["protected_content_preserved"] = all(
            b["content"] == c["content"]
            for b, c in zip(baseline, candidate)
            if b["block_type"] != MUTABLE_BLOCK_TYPE
        )
        inv["mutation_scope_allowed"] = all(
            _mutation_scope_ok(b, c) for b, c in zip(baseline, candidate)
        )
        realized = sum(
            len(b["content"]) - len(c["content"])
            for b, c in zip(baseline, candidate)
        )
    else:
        # Count mismatch makes positional comparison meaningless; fail the rest.
        inv["order_and_roles_preserved"] = False
        inv["protected_content_preserved"] = False
        inv["mutation_scope_allowed"] = False
        realized = 0

    # Realized savings must equal the optimizer's own realized figure, must be
    # non-negative, and a non-zero saving must coincide with a reported mutation.
    inv["savings_accounting_consistent"] = (
        realized >= 0
        and realized == result.chars_saved
        and (realized > 0) == bool(result.mutated)
        and (result.blocks_replaced > 0) == bool(result.mutated)
    )
    return inv, realized


def _raw_content_strings(cases: list[dict], *, min_len: int = 12) -> list[str]:
    """Collect non-trivial raw content lines for the privacy substring scan."""
    out: list[str] = []
    for case in cases:
        for m in case.get("messages", []):
            text = (m.get("content") or "")
            for line in text.split("\n"):
                line = line.strip()
                if len(line) >= min_len:
                    out.append(line)
    return out


def assert_report_privacy_safe(report_dict: dict, raw_texts: list[str]) -> None:
    """Guard the report before emission: no forbidden keys, no raw content."""
    _assert_no_forbidden_keys(report_dict)
    blob = json.dumps(report_dict, ensure_ascii=False)
    for text in raw_texts:
        if text and text in blob:
            raise RuntimeError("refusing to emit report containing raw case content")


def run_validation(
    cases: list[dict],
    *,
    baseline_mode: str = "off",
    candidate_mode: str,
    salt: str,
    min_block_chars: int = DEFAULT_MIN_BLOCK_CHARS,
    date: str,
    tokenizer_spec: object | None = None,
    optimize_fn: Callable[..., tuple[list[dict], PromptDedupCanaryResult]] | None = None,
) -> ValidationReport:
    """Validate every case under baseline vs candidate and build the gate report."""
    # Resolve at call time (not as a default arg) so the module-level
    # ``optimize_case`` stays monkeypatchable from tests and callers.
    optimize_fn = optimize_fn or optimize_case
    tokenizer = resolve_tokenizer(tokenizer_spec)
    tok_status = "available" if tokenizer is not None else "unavailable"

    case_results: list[ValidationCaseResult] = []
    total_blocks = 0
    total_chars = 0
    total_actual_saved = 0 if tokenizer is not None else None

    for case in cases:
        msgs = _messages(case)
        baseline_msgs, _ = optimize_fn(
            list(msgs), mode=baseline_mode, salt=salt, min_block_chars=min_block_chars
        )
        candidate_msgs, result = optimize_fn(
            list(msgs), mode=candidate_mode, salt=salt, min_block_chars=min_block_chars
        )

        inv, realized = check_invariants(baseline_msgs, candidate_msgs, result)
        failed = [name for name, ok in inv.items() if not ok]

        at_before = at_after = at_saved = None
        if tokenizer is not None:
            at_before = sum(tokenizer.count(m["content"]) for m in baseline_msgs)
            at_after = sum(tokenizer.count(m["content"]) for m in candidate_msgs)
            at_saved = at_before - at_after
            total_actual_saved += at_saved

        skill_items = sum(1 for m in msgs if m["block_type"] == MUTABLE_BLOCK_TYPE)
        total_blocks += result.blocks_replaced if result.mutated else 0
        total_chars += realized

        case_results.append(
            ValidationCaseResult(
                case_id=str(case.get("case_id", "")),
                source=case.get("source"),
                message_count=len(msgs),
                skill_item_count=skill_items,
                mutated=bool(result.mutated),
                blocks_replaced=result.blocks_replaced if result.mutated else 0,
                chars_saved=realized,
                invariants=inv,
                passed=not failed,
                failed_invariants=failed,
                actual_tokens_before=at_before,
                actual_tokens_after=at_after,
                actual_tokens_saved=at_saved,
            )
        )

    passed_cases = sum(1 for c in case_results if c.passed)
    failed_cases = len(case_results) - passed_cases
    notes = [
        "baseline runs the optimizer in 'off' mode and must leave the payload "
        "byte-identical; the candidate mode is the change under test",
        "chars_saved is the REALIZED processed-payload before/after char delta, "
        "not an opportunity count",
    ]
    if tokenizer is None:
        notes.append(
            "actual-token savings unavailable (no exact tokenizer backend configured); "
            "no actual-token fields are reported"
        )

    return ValidationReport(
        schema_version=VALIDATION_SET_SCHEMA_VERSION,
        generated_date=date,
        salt_fingerprint=_salt_fingerprint(salt),
        baseline_mode=baseline_mode,
        candidate_mode=candidate_mode,
        case_count=len(case_results),
        passed=failed_cases == 0,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        total_blocks_replaced=total_blocks,
        total_chars_saved=total_chars,
        tokenizer_status=tok_status,
        tokenizer_backend=tokenizer.name if tokenizer is not None else None,
        total_actual_tokens_saved=total_actual_saved,
        invariant_names=list(INVARIANT_NAMES),
        cases=case_results,
        notes=notes,
    )


def report_to_dict(report: ValidationReport) -> dict:
    from dataclasses import asdict

    return asdict(report)


def render_markdown(report: ValidationReport) -> str:
    gate = "PASS ✅" if report.passed else "FAIL ❌"
    lines = [
        f"# ContextPilot trace validation — {report.generated_date}",
        "",
        f"Gate: **{gate}**",
        f"Salt fingerprint: `{report.salt_fingerprint}`",
        f"Baseline mode: `{report.baseline_mode}` | Candidate mode: `{report.candidate_mode}`",
        "",
        "## Summary",
        f"- Cases: {report.case_count} ({report.passed_cases} passed, "
        f"{report.failed_cases} failed)",
        f"- Blocks replaced (realized): {report.total_blocks_replaced}",
        f"- Chars saved (realized before/after delta): {report.total_chars_saved}",
    ]
    if report.tokenizer_status == "available":
        lines.append(
            f"- Actual tokens saved ({report.tokenizer_backend}): "
            f"{report.total_actual_tokens_saved}"
        )
    else:
        lines.append("- Actual tokens saved: unavailable (no tokenizer backend)")
    lines.append("")
    lines.append("## Invariants checked")
    for name in report.invariant_names:
        lines.append(f"- {name}")
    if report.failed_cases:
        lines.append("")
        lines.append("## Failures")
        for c in report.cases:
            if not c.passed:
                lines.append(
                    f"- `{c.case_id}` (source={c.source}): "
                    f"{', '.join(c.failed_invariants)}"
                )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Artifact-dedup canary validation (provenance-aware tool-artifact reuse)
# ---------------------------------------------------------------------------

# Stable invariant identifiers for the artifact-dedup gate. Mirrors the prompt
# gate but swaps in artifact-scope and reference-resolvability checks.
ARTIFACT_INVARIANT_NAMES = [
    "message_count_preserved",
    "order_and_roles_preserved",
    "protected_content_preserved",
    "artifact_mutation_scope_allowed",
    "artifact_reference_resolvable",
    "savings_accounting_consistent",
]


def _span_links(case: dict) -> list[ArtifactSpanLink]:
    links = []
    for raw in case.get("span_links") or []:
        try:
            links.append(
                ArtifactSpanLink(
                    source_index=int(raw["source_index"]),
                    source_start=int(raw["source_start"]),
                    source_end=int(raw["source_end"]),
                    target_index=int(raw["target_index"]),
                    target_start=int(raw["target_start"]),
                    target_end=int(raw["target_end"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return links


def optimize_artifact_case(
    messages: list[dict], *, mode: str, salt: str, min_block_chars: int, span_links: list[ArtifactSpanLink] | None = None
) -> tuple[list[dict], ArtifactDedupCanaryResult]:
    """Run the artifact-dedup canary over a case's messages in the given mode.

    Returns ``(out_messages, result)``. The canary mutates only mutable artifact
    bodies in place; ``out_messages`` mirrors the input role/block_type/order
    with the (possibly) rewritten content so the caller can diff payloads.
    """
    contents = [_LLMContent(m["block_type"], m["content"]) for m in messages]
    result = apply_artifact_dedup_canary(
        contents, salt=salt, min_block_chars=min_block_chars, mode=mode, span_links=span_links
    )
    out = [
        {"role": m["role"], "block_type": m["block_type"], "content": c.content}
        for m, c in zip(messages, contents)
    ]
    return out, result


def _artifact_mutation_scope_ok(base: dict, cand: dict) -> bool:
    """A single message changed only within the allowed (artifact-only) scope."""
    if base["content"] == cand["content"]:
        return True
    # Only mutable artifact bodies may ever change.
    if base["block_type"] not in MUTABLE_ARTIFACT_BLOCK_TYPES:
        return False
    if len(cand["content"]) >= len(base["content"]):
        return False

    # Whole-body replacement remains valid.
    if _parse_artifact_reference(cand["content"]) is not None:
        return True

    # Declared source-span replacement: a byte-identical line-aligned span may be
    # swapped for one standalone strictly shorter reference while surrounding
    # prose remains byte-identical.
    base_text = base["content"]
    cand_text = cand["content"]
    prefix = 0
    while prefix < len(base_text) and prefix < len(cand_text) and base_text[prefix] == cand_text[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < len(base_text) - prefix
        and suffix < len(cand_text) - prefix
        and base_text[len(base_text) - 1 - suffix] == cand_text[len(cand_text) - 1 - suffix]
    ):
        suffix += 1
    base_end = len(base_text) - suffix
    cand_end = len(cand_text) - suffix
    old_mid = base_text[prefix:base_end]
    new_mid = cand_text[prefix:cand_end]
    if (
        old_mid
        and _parse_artifact_reference(new_mid.strip()) is not None
        and len(new_mid.strip()) < len(old_mid)
        and (prefix == 0 or base_text[prefix - 1] == "\n")
        and (base_end == len(base_text) or base_text[base_end] == "\n")
    ):
        return True

    # Fenced sub-artifact replacement: prose must be byte-identical and only a
    # whole fenced segment may be swapped for one strictly shorter reference line.
    pos = 0
    changed = False
    for kind, text in _segment_fenced_blocks(base["content"]):
        if kind != "fence":
            if not cand["content"].startswith(text, pos):
                return False
            pos += len(text)
            continue
        if cand["content"].startswith(text, pos):
            pos += len(text)
            continue
        newline = cand["content"].find("\n", pos)
        end = len(cand["content"]) if newline == -1 else newline
        ref = cand["content"][pos:end]
        if _parse_artifact_reference(ref) is None or len(ref) >= len(text):
            return False
        pos = end
        changed = True
    return changed and pos == len(cand["content"])


def check_artifact_invariants(
    baseline: list[dict],
    candidate: list[dict],
    result: ArtifactDedupCanaryResult,
    *,
    salt: str,
    span_links: list[ArtifactSpanLink] | None = None,
) -> tuple[dict[str, bool], int]:
    """Check accuracy-preservation invariants for an artifact-dedup pass.

    Returns ``(invariant -> passed, realized_chars_saved)`` where
    ``realized_chars_saved`` is the ACTUAL summed before/after character delta of
    the processed payload (not an opportunity count).
    """
    inv: dict[str, bool] = {}

    inv["message_count_preserved"] = len(baseline) == len(candidate)

    if inv["message_count_preserved"]:
        inv["order_and_roles_preserved"] = all(
            b["role"] == c["role"] and b["block_type"] == c["block_type"]
            for b, c in zip(baseline, candidate)
        )
        inv["protected_content_preserved"] = all(
            b["content"] == c["content"]
            for b, c in zip(baseline, candidate)
            if b["block_type"] not in MUTABLE_ARTIFACT_BLOCK_TYPES
        )
        inv["artifact_mutation_scope_allowed"] = all(
            _artifact_mutation_scope_ok(b, c) for b, c in zip(baseline, candidate)
        )
        cand_contents = [
            _LLMContent(c["block_type"], c["content"]) for c in candidate
        ]
        inv["artifact_reference_resolvable"] = (
            dangling_artifact_references(cand_contents, salt=salt, span_links=span_links) == []
        )
        realized = sum(
            len(b["content"]) - len(c["content"])
            for b, c in zip(baseline, candidate)
        )
    else:
        # Count mismatch makes positional comparison meaningless; fail the rest.
        inv["order_and_roles_preserved"] = False
        inv["protected_content_preserved"] = False
        inv["artifact_mutation_scope_allowed"] = False
        inv["artifact_reference_resolvable"] = False
        realized = 0

    inv["savings_accounting_consistent"] = (
        realized >= 0
        and realized == result.chars_saved
        and (realized > 0) == bool(result.mutated)
        and (result.blocks_replaced > 0) == bool(result.mutated)
    )
    return inv, realized


def run_artifact_validation(
    cases: list[dict],
    *,
    baseline_mode: str = "off",
    candidate_mode: str,
    salt: str,
    min_block_chars: int = DEFAULT_MIN_BLOCK_CHARS,
    date: str,
    tokenizer_spec: object | None = None,
    optimize_fn: Callable[..., tuple[list[dict], ArtifactDedupCanaryResult]] | None = None,
) -> ValidationReport:
    """Validate every case under baseline vs candidate for the artifact canary."""
    optimize_fn = optimize_fn or optimize_artifact_case
    tokenizer = resolve_tokenizer(tokenizer_spec)
    tok_status = "available" if tokenizer is not None else "unavailable"

    case_results: list[ValidationCaseResult] = []
    total_blocks = 0
    total_chars = 0
    total_actual_saved = 0 if tokenizer is not None else None

    for case in cases:
        msgs = _messages(case)
        span_links = _span_links(case)
        if span_links:
            baseline_msgs, _ = optimize_fn(
                list(msgs), mode=baseline_mode, salt=salt, min_block_chars=min_block_chars, span_links=span_links
            )
            candidate_msgs, result = optimize_fn(
                list(msgs), mode=candidate_mode, salt=salt, min_block_chars=min_block_chars, span_links=span_links
            )
        else:
            baseline_msgs, _ = optimize_fn(
                list(msgs), mode=baseline_mode, salt=salt, min_block_chars=min_block_chars
            )
            candidate_msgs, result = optimize_fn(
                list(msgs), mode=candidate_mode, salt=salt, min_block_chars=min_block_chars
            )

        inv, realized = check_artifact_invariants(
            baseline_msgs, candidate_msgs, result, salt=salt, span_links=span_links
        )
        failed = [name for name, ok in inv.items() if not ok]

        at_before = at_after = at_saved = None
        if tokenizer is not None:
            at_before = sum(tokenizer.count(m["content"]) for m in baseline_msgs)
            at_after = sum(tokenizer.count(m["content"]) for m in candidate_msgs)
            at_saved = at_before - at_after
            total_actual_saved += at_saved

        artifact_items = sum(
            1 for m in msgs if m["block_type"] in MUTABLE_ARTIFACT_BLOCK_TYPES
        )
        total_blocks += result.blocks_replaced if result.mutated else 0
        total_chars += realized

        case_results.append(
            ValidationCaseResult(
                case_id=str(case.get("case_id", "")),
                source=case.get("source"),
                message_count=len(msgs),
                skill_item_count=artifact_items,
                mutated=bool(result.mutated),
                blocks_replaced=result.blocks_replaced if result.mutated else 0,
                chars_saved=realized,
                invariants=inv,
                passed=not failed,
                failed_invariants=failed,
                actual_tokens_before=at_before,
                actual_tokens_after=at_after,
                actual_tokens_saved=at_saved,
            )
        )

    passed_cases = sum(1 for c in case_results if c.passed)
    failed_cases = len(case_results) - passed_cases
    notes = [
        "baseline runs the artifact canary in 'off' mode and must leave the "
        "payload byte-identical; the candidate mode is the change under test",
        "chars_saved is the REALIZED processed-payload before/after char delta, "
        "not an opportunity count",
    ]
    if tokenizer is None:
        notes.append(
            "actual-token savings unavailable (no exact tokenizer backend configured); "
            "no actual-token fields are reported"
        )

    return ValidationReport(
        schema_version=VALIDATION_SET_SCHEMA_VERSION,
        generated_date=date,
        salt_fingerprint=_salt_fingerprint(salt),
        baseline_mode=baseline_mode,
        candidate_mode=candidate_mode,
        case_count=len(case_results),
        passed=failed_cases == 0,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        total_blocks_replaced=total_blocks,
        total_chars_saved=total_chars,
        tokenizer_status=tok_status,
        tokenizer_backend=tokenizer.name if tokenizer is not None else None,
        total_actual_tokens_saved=total_actual_saved,
        invariant_names=list(ARTIFACT_INVARIANT_NAMES),
        cases=case_results,
        notes=notes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run ContextPilot trace validation: check accuracy-preservation "
            "invariants of a candidate optimization against a fixed local corpus. "
            "Exits non-zero on any gate failure."
        )
    )
    parser.add_argument("corpus", type=Path, help="path to the JSONL validation corpus")
    parser.add_argument(
        "--gate",
        choices=["prompt", "artifact"],
        default="prompt",
        help=(
            "which validation gate to run: 'prompt' for skill-prompt dedup "
            "or 'artifact' for provenance-aware tool/artifact reuse (default: prompt)"
        ),
    )
    parser.add_argument(
        "--candidate-mode",
        default=None,
        help=(
            "dedup mode to validate (off|shadow|canary). Defaults to the "
            "resolved CONTEXTPILOT_*_DEDUP_MODE env for the selected gate."
        ),
    )
    parser.add_argument(
        "--baseline-mode",
        default="off",
        help="reference mode the candidate is compared against (default: off)",
    )
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument(
        "--min-block-chars", type=int, default=DEFAULT_MIN_BLOCK_CHARS
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "opt-in exact tokenizer backend for actual-token accounting, e.g. "
            "'tiktoken:cl100k_base' (off by default -> tokens reported unavailable)"
        ),
    )
    parser.add_argument(
        "--format", choices=["json", "markdown"], default="json"
    )
    parser.add_argument("--date", default=dt.date.today().isoformat())
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        raise SystemExit(f"validation corpus not found: {args.corpus}")

    if args.candidate_mode is not None:
        candidate_mode = args.candidate_mode
    elif args.gate == "artifact":
        candidate_mode = resolve_artifact_dedup_mode()
    else:
        candidate_mode = resolve_prompt_dedup_mode()

    cases = load_cases(args.corpus)
    run_fn = run_artifact_validation if args.gate == "artifact" else run_validation
    report = run_fn(
        cases,
        baseline_mode=args.baseline_mode,
        candidate_mode=candidate_mode,
        salt=args.salt,
        min_block_chars=args.min_block_chars,
        date=args.date,
        tokenizer_spec=args.tokenizer,
    )

    report_dict = report_to_dict(report)
    # Hard privacy gate: never emit a report carrying forbidden keys or raw text.
    assert_report_privacy_safe(report_dict, _raw_content_strings(cases))

    if args.format == "markdown":
        print(render_markdown(report))
    else:
        print(json.dumps(report_dict, ensure_ascii=False, indent=2))

    return 0 if report.passed else 1
