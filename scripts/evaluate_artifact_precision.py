#!/usr/bin/env python3
"""Run a reproducible synthetic precision/recall check for artifact canary rewrites.

This is intentionally *not* a product/model precision benchmark. It is a small,
hand-labeled synthetic self-consistency suite that checks the current exact
whole-body, fenced-block, and declared source-span rewrite gates against planted
positive/negative cases. Top-level metric names are prefixed with ``synthetic``
so they are not confused with field precision on real traces.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextpilot.hermes_opportunities.artifact_dedup_canary import (
    ARTIFACT_DEDUP_DISABLE_ENV,
    ARTIFACT_DEDUP_MODE_ENV,
    ARTIFACT_DEDUP_CANARY_REFERENCE_TEMPLATE,
    ArtifactSpanLink,
    apply_artifact_dedup_canary,
    dangling_artifact_references,
)
from contextpilot.hermes_opportunities.models import _LLMContent

SALT = "precision-eval-salt"
MIN = 40
ART = ("Synthetic artifact payload alpha bravo charlie delta echo foxtrot. " * 6).strip()
SHORT = "Short duplicate body just above forty chars."
FENCE = (
    "```log\n"
    + ("synthetic repeated fenced worker output alpha bravo charlie\n" * 8).rstrip("\n")
    + "\n```"
)
SPAN = ("declared source span line alpha bravo charlie\n" * 8).rstrip("\n")


@dataclass(frozen=True)
class LabeledCase:
    name: str
    expected_replacements: int
    items: list[_LLMContent]
    span_links: list[ArtifactSpanLink]


def _content(block_type: str, text: str) -> _LLMContent:
    return _LLMContent(block_type, text)


def _clone(items: Iterable[_LLMContent]) -> list[_LLMContent]:
    return [_LLMContent(item.block_type, item.content) for item in items]


def _span_case(*, exact: bool = True, forward: bool = False, oob: bool = False) -> tuple[list[_LLMContent], list[ArtifactSpanLink]]:
    src = "tool pre\n" + SPAN + "\ntool post\n"
    target_span = SPAN if exact else SPAN.replace("charlie", "changed", 1)
    tgt = "parent pre\n" + target_span + "\nparent post\n"
    source_start = src.index(SPAN)
    source_end = source_start + len(SPAN)
    target_start = tgt.index(target_span)
    target_end = target_start + len(target_span)
    items = [_content("tool_result", src), _content("assistant_context", tgt)]
    if forward:
        link = ArtifactSpanLink(1, target_start, target_end, 0, source_start, source_end)
    elif oob:
        link = ArtifactSpanLink(0, source_start, len(src) + 100, 1, target_start, target_end)
    else:
        link = ArtifactSpanLink(0, source_start, source_end, 1, target_start, target_end)
    return items, [link]


def build_cases() -> list[LabeledCase]:
    cases: list[LabeledCase] = [
        LabeledCase("whole_tool_exact_duplicate", 1, [_content("tool_result", ART), _content("tool_result", ART)], []),
        LabeledCase("whole_cross_type_exact_duplicate", 1, [_content("tool_result", ART), _content("assistant_context", ART)], []),
        LabeledCase("whole_near_duplicate_not_mutated", 0, [_content("tool_result", ART), _content("tool_result", ART + " changed")], []),
        LabeledCase("fenced_internal_duplicate", 1, [_content("assistant_context", "before\n" + FENCE + "\nmiddle\n" + FENCE + "\nafter")], []),
        LabeledCase("two_fenced_duplicates", 2, [_content("assistant_context", "a\n" + FENCE + "\nb\n" + FENCE + "\nc\n" + FENCE + "\nd")], []),
        LabeledCase("protected_user_system_duplicates", 0, [_content("user_ctx", ART), _content("system_ctx", ART)], []),
        LabeledCase("short_duplicate_never_grow", 0, [_content("tool_result", SHORT), _content("tool_result", SHORT)], []),
        LabeledCase("unterminated_fence_not_mutated", 0, [_content("assistant_context", ("prefix\n```log\n" + ("unterminated line alpha bravo charlie\n" * 8)) * 2)], []),
        LabeledCase("copied_plain_span_without_declared_link", 0, [_content("tool_result", "tool\n" + SPAN + "\nend"), _content("assistant_context", "parent\n" + SPAN + "\nend")], []),
        LabeledCase("protected_duplicate_tool_vs_user", 0, [_content("tool_result", ART), _content("user_ctx", ART)], []),
        LabeledCase("protected_duplicate_tool_vs_system", 0, [_content("tool_result", ART), _content("system_ctx", ART)], []),
    ]
    items, links = _span_case(exact=True)
    cases.append(LabeledCase("declared_source_span_exact", 1, items, links))
    items, links = _span_case(exact=False)
    cases.append(LabeledCase("declared_span_content_differs", 0, items, links))
    items, links = _span_case(exact=True, forward=True)
    cases.append(LabeledCase("forward_span_link_rejected", 0, items, links))
    items, links = _span_case(exact=True, oob=True)
    cases.append(LabeledCase("oob_span_link_rejected", 0, items, links))
    items, links = _span_case(exact=True)
    # Duplicate declaration for the same target is deduplicated to one event.
    cases.append(LabeledCase("duplicate_span_declaration_counts_once", 1, items, links + links))
    return cases


def _mode_gate_checks() -> dict[str, bool]:
    base = [_content("tool_result", ART), _content("tool_result", ART)]
    off_items = _clone(base)
    off = apply_artifact_dedup_canary(off_items, salt=SALT, min_block_chars=MIN, mode="off")
    shadow_items = _clone(base)
    shadow = apply_artifact_dedup_canary(shadow_items, salt=SALT, min_block_chars=MIN, mode="shadow")
    disable_items = _clone(base)
    old_mode = os.environ.get(ARTIFACT_DEDUP_MODE_ENV)
    old_disable = os.environ.get(ARTIFACT_DEDUP_DISABLE_ENV)
    try:
        os.environ[ARTIFACT_DEDUP_MODE_ENV] = "canary"
        os.environ[ARTIFACT_DEDUP_DISABLE_ENV] = "1"
        disabled = apply_artifact_dedup_canary(disable_items, salt=SALT, min_block_chars=MIN)
    finally:
        if old_mode is None:
            os.environ.pop(ARTIFACT_DEDUP_MODE_ENV, None)
        else:
            os.environ[ARTIFACT_DEDUP_MODE_ENV] = old_mode
        if old_disable is None:
            os.environ.pop(ARTIFACT_DEDUP_DISABLE_ENV, None)
        else:
            os.environ[ARTIFACT_DEDUP_DISABLE_ENV] = old_disable
    return {
        "off_no_mutation": not off.mutated and [i.content for i in off_items] == [i.content for i in base],
        "shadow_no_mutation": not shadow.mutated and [i.content for i in shadow_items] == [i.content for i in base],
        "disable_env_no_mutation": not disabled.mutated and [i.content for i in disable_items] == [i.content for i in base],
    }


def _validation_gate_checks() -> dict[str, bool]:
    forged_ref = ARTIFACT_DEDUP_CANARY_REFERENCE_TEMPLATE.replace("<type>", "tool_result").replace("<hash>", "deadbeef")
    forged = [_content("assistant_context", forged_ref)]
    return {
        "forged_reference_detected": dangling_artifact_references(forged, salt=SALT) == [0],
    }


def build_report() -> dict:
    rows = []
    tp = fp = fn = tn = 0
    expected_total = predicted_total = chars_saved = 0
    for case in build_cases():
        items = _clone(case.items)
        result = apply_artifact_dedup_canary(
            items,
            salt=SALT,
            min_block_chars=MIN,
            mode="canary",
            span_links=case.span_links,
        )
        actual = result.blocks_replaced
        expected = case.expected_replacements
        dangling = dangling_artifact_references(items, salt=SALT, span_links=case.span_links)
        case_pass = actual == expected and not dangling
        this_tp = min(actual, expected)
        this_fp = max(0, actual - expected)
        this_fn = max(0, expected - actual)
        this_tn = 1 if actual == 0 and expected == 0 else 0
        tp += this_tp
        fp += this_fp
        fn += this_fn
        tn += this_tn
        expected_total += expected
        predicted_total += actual
        chars_saved += result.chars_saved
        rows.append(
            {
                "name": case.name,
                "expected_replacements": expected,
                "actual_replacements": actual,
                "pass": case_pass,
                "chars_saved": result.chars_saved,
                "span_replacements": result.span_blocks_replaced,
                "dangling": dangling,
            }
        )

    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "corpus": "synthetic_labeled_artifact_precision_v1",
        "claim_scope": "synthetic exact/provenance gate self-consistency; not field/model/product precision",
        "case_count": len(rows),
        "synthetic_event_tp": tp,
        "synthetic_event_fp": fp,
        "synthetic_event_fn": fn,
        "synthetic_negative_case_tn": tn,
        "synthetic_negative_case_fpr": fp / (fp + tn) if fp + tn else 0.0,
        "synthetic_event_precision": tp / (tp + fp) if tp + fp else 1.0,
        "synthetic_event_recall": tp / (tp + fn) if tp + fn else 1.0,
        "synthetic_case_accuracy": sum(1 for row in rows if row["pass"]) / len(rows),
        "predicted_replacements": predicted_total,
        "expected_replacements": expected_total,
        "synthetic_realized_chars_saved": chars_saved,
        "mode_gate_checks": _mode_gate_checks(),
        "validation_gate_checks": _validation_gate_checks(),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, help="Optional JSON output path")
    args = parser.parse_args()
    report = build_report()
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
