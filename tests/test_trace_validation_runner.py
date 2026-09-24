"""Tests for the trace-validation runner gate.

The committed fixture is fully synthetic. The runner may read raw case content
from the local JSONL corpus, but its emitted report must remain privacy-safe and
must fail when mutations touch protected content or savings accounting lies.
"""

import json
from pathlib import Path

import pytest

from contextpilot.hermes_opportunities.prompt_dedup_canary import PromptDedupCanaryResult
from contextpilot.trace_validation.runner import (
    assert_report_privacy_safe,
    load_cases,
    render_markdown,
    report_to_dict,
    run_validation,
)

FIXTURE = Path("tests/fixtures/trace_validation/synthetic_cases.jsonl")
SALT = "test-trace-salt"


def test_canary_validation_passes_on_synthetic_fixture():
    cases = load_cases(FIXTURE)
    report = run_validation(
        cases,
        baseline_mode="off",
        candidate_mode="canary",
        salt=SALT,
        min_block_chars=40,
        date="2026-06-14",
    )
    assert report.passed is True
    assert report.failed_cases == 0
    # The first synthetic skill case has safe duplicate lines; the third is
    # denylisted by "must" and should remain unchanged.
    assert report.total_blocks_replaced == 2
    assert report.total_chars_saved > 0
    assert any(c.mutated for c in report.cases)
    assert report.tokenizer_status == "unavailable"
    assert report.total_actual_tokens_saved is None


def test_shadow_validation_passes_without_realized_savings():
    cases = load_cases(FIXTURE)
    report = run_validation(
        cases,
        baseline_mode="off",
        candidate_mode="shadow",
        salt=SALT,
        min_block_chars=40,
        date="2026-06-14",
    )
    assert report.passed is True
    assert report.total_blocks_replaced == 0
    assert report.total_chars_saved == 0
    assert all(not c.mutated for c in report.cases)


def test_report_is_privacy_safe_and_markdown_contains_no_raw_fixture_text():
    cases = load_cases(FIXTURE)
    report = run_validation(
        cases,
        baseline_mode="off",
        candidate_mode="canary",
        salt=SALT,
        min_block_chars=40,
        date="2026-06-14",
    )
    report_dict = report_to_dict(report)
    raw_needles = [
        "Please summarize the synthetic quarterly figures attached above for the demo.",
        "Synthetic reusable skill paragraph that explains how the demo helper",
        "synthetic_weather=clear",
    ]
    assert_report_privacy_safe(report_dict, raw_needles)
    blob = json.dumps(report_dict, ensure_ascii=False)
    md = render_markdown(report)
    for needle in raw_needles:
        assert needle not in blob
        assert needle not in md


def test_runner_fails_when_candidate_mutates_protected_user_content():
    cases = load_cases(FIXTURE)

    def bad_optimizer(messages, *, mode, salt, min_block_chars):
        out = [dict(m) for m in messages]
        if mode == "bad" and out:
            for m in out:
                if m["block_type"] == "user_prompt":
                    m["content"] = "[dropped]"
                    break
            result = PromptDedupCanaryResult(
                mode="canary",
                prompt_dedup_class="same_type_skill_prompt_only",
                mutated=True,
                item_count=0,
                skill_item_count=0,
                candidate_block_count=0,
                candidate_chars=0,
                blocks_replaced=1,
                chars_saved=1,
                denylisted_block_count=0,
            )
            return out, result
        result = PromptDedupCanaryResult(
            mode="off",
            prompt_dedup_class="same_type_skill_prompt_only",
            mutated=False,
            item_count=0,
            skill_item_count=0,
            candidate_block_count=0,
            candidate_chars=0,
            blocks_replaced=0,
            chars_saved=0,
            denylisted_block_count=0,
        )
        return out, result

    report = run_validation(
        cases[:1],
        baseline_mode="off",
        candidate_mode="bad",
        salt=SALT,
        min_block_chars=40,
        date="2026-06-14",
        optimize_fn=bad_optimizer,
    )
    assert report.passed is False
    assert report.failed_cases == 1
    failed = report.cases[0].failed_invariants
    assert "protected_content_preserved" in failed
    assert "mutation_scope_allowed" in failed
    assert "savings_accounting_consistent" in failed


def test_privacy_guard_rejects_raw_content_in_report():
    with pytest.raises(RuntimeError):
        assert_report_privacy_safe({"ok": True, "note": "raw secret line"}, ["raw secret line"])
