#!/usr/bin/env python3
"""Build an independent provenance-linker dataset.

Modes:
* ``synthetic``: committed-safe toy examples with gold claim→evidence links.
* ``trace``: local raw export from an existing trace-validation JSONL corpus.

The output schema is independent from artifact-dedup validation: it is designed
for training/shadowing a general provenance linker that maps assistant claims to
supporting evidence blocks.  It does not mutate online payloads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextpilot.provenance_linking import (  # noqa: E402
    ProvenanceBlock,
    ProvenanceClaim,
    ProvenanceExample,
    ProvenanceLink,
    example_from_trace_case,
    shadow_link_claims,
    to_jsonable,
    write_jsonl,
)


def _claim(block: ProvenanceBlock, text: str, cid: str) -> ProvenanceClaim:
    start = block.text.index(text)
    return ProvenanceClaim(cid, block.block_id, text, start, start + len(text))


def _span(block: ProvenanceBlock, text: str) -> tuple[int, int]:
    start = block.text.index(text)
    return start, start + len(text)


def synthetic_examples() -> list[ProvenanceExample]:
    examples: list[ProvenanceExample] = []

    # Coding is one example domain, not the design itself.
    e1_tool = ProvenanceBlock(
        "coding_tool_pytest",
        "tool_result",
        "pytest output\n46 passed, 2 warnings in 0.33s\nfull suite: 592 passed, 37 skipped in 19.02s\n",
        {"domain": "coding", "tool": "pytest"},
    )
    e1_review = ProvenanceBlock(
        "coding_worker_review",
        "worker_output",
        "Read-only review: PASS. No correctness blockers. Minor naming nit only.\n",
        {"domain": "coding", "worker": "reviewer"},
    )
    e1_assistant = ProvenanceBlock(
        "coding_parent_summary",
        "assistant_context",
        "Full suite passed: 592 passed, 37 skipped. Read-only review PASS, so PR can continue.",
        {"domain": "coding"},
    )
    c1 = _claim(e1_assistant, "Full suite passed: 592 passed, 37 skipped.", "claim_coding_tests_passed")
    c2 = _claim(e1_assistant, "Read-only review PASS, so PR can continue.", "claim_coding_pr_continue")
    s1, t1 = _span(e1_tool, "592 passed, 37 skipped")
    s2, t2 = _span(e1_review, "PASS. No correctness blockers")
    examples.append(
        ProvenanceExample(
            "synthetic_coding_review",
            "coding",
            [e1_tool, e1_review, e1_assistant],
            [c1, c2],
            [
                ProvenanceLink(c1.claim_id, e1_tool.block_id, s1, t1, "supports", 1.0, "gold"),
                ProvenanceLink(c2.claim_id, e1_review.block_id, s2, t2, "supports", 1.0, "gold"),
            ],
        )
    )

    e2_web = ProvenanceBlock(
        "research_web_source",
        "web_result",
        "Paper abstract: the system uses retrieval plus citation verification to reduce unsupported claims.",
        {"domain": "research"},
    )
    e2_worker = ProvenanceBlock(
        "research_worker_summary",
        "worker_output",
        "Worker finding: citation verification catches unsupported claims before final answer.",
        {"domain": "research"},
    )
    e2_assistant = ProvenanceBlock(
        "research_parent_summary",
        "assistant_context",
        "The method reduces unsupported claims by using retrieval plus citation verification.",
        {"domain": "research"},
    )
    c3 = _claim(e2_assistant, e2_assistant.text, "claim_research_attribution")
    s3, t3 = _span(e2_web, "retrieval plus citation verification")
    examples.append(
        ProvenanceExample(
            "synthetic_research_claim",
            "research",
            [e2_web, e2_worker, e2_assistant],
            [c3],
            [ProvenanceLink(c3.claim_id, e2_web.block_id, s3, t3, "summarized_support", 1.0, "gold")],
        )
    )

    e3_sensor = ProvenanceBlock(
        "ops_metric_source",
        "tool_result",
        "monitor: p95 latency = 183ms, error_rate = 0.02%, queue_depth = 4",
        {"domain": "ops"},
    )
    e3_assistant = ProvenanceBlock(
        "ops_summary",
        "assistant_context",
        "Latency is healthy: p95 latency = 183ms and error_rate = 0.02%.",
        {"domain": "ops"},
    )
    c4 = _claim(e3_assistant, e3_assistant.text, "claim_ops_latency_healthy")
    s4, t4 = _span(e3_sensor, "p95 latency = 183ms")
    examples.append(
        ProvenanceExample(
            "synthetic_ops_metrics",
            "ops",
            [e3_sensor, e3_assistant],
            [c4],
            [ProvenanceLink(c4.claim_id, e3_sensor.block_id, s4, t4, "extracted_support", 1.0, "gold")],
        )
    )
    return examples


def build_from_trace_jsonl(path: Path, *, limit: int | None) -> list[ProvenanceExample]:
    examples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        examples.append(example_from_trace_case(json.loads(line)))
        if limit is not None and len(examples) >= limit:
            break
    return examples


def manifest_for(examples: list[ProvenanceExample], *, mode: str, output: Path) -> dict:
    shadow_counts = [len(shadow_link_claims(ex)) for ex in examples]
    return {
        "schema_version": 1,
        "dataset_kind": "provenance_linking",
        "mode": mode,
        "output": str(output),
        "example_count": len(examples),
        "block_count": sum(len(ex.blocks) for ex in examples),
        "claim_count": sum(len(ex.claims) for ex in examples),
        "gold_link_count": sum(len(ex.gold_links) for ex in examples),
        "shadow_link_count": sum(shadow_counts),
        "privacy_note": "synthetic is commit-safe; trace mode may contain raw local content and must stay local/gitignored",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "trace"], default="synthetic")
    parser.add_argument("--trace-jsonl", type=Path, help="trace-validation JSONL for trace mode")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-shadow", action="store_true", help="include shadow_links alongside gold/raw examples")
    parser.add_argument(
        "--allow-unsafe-trace-output",
        action="store_true",
        help="allow trace-mode raw output outside ~/contextpilot/provenance_datasets (not recommended)",
    )
    args = parser.parse_args()

    if args.mode == "synthetic":
        examples = synthetic_examples()
    else:
        safe_root = (Path.home() / "contextpilot" / "provenance_datasets").resolve()
        out_resolved = args.output.resolve()
        if not args.allow_unsafe_trace_output and safe_root not in [out_resolved, *out_resolved.parents]:
            raise SystemExit(
                "trace mode writes raw content; use an output under "
                f"{safe_root} or pass --allow-unsafe-trace-output explicitly"
            )
        if args.trace_jsonl is None:
            raise SystemExit("--trace-jsonl is required for --mode trace")
        examples = build_from_trace_jsonl(args.trace_jsonl, limit=args.limit)

    rows = [to_jsonable(ex, shadow_links=shadow_link_claims(ex) if args.with_shadow else None) for ex in examples]
    write_jsonl(args.output, rows)
    manifest = manifest_for(examples, mode=args.mode, output=args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"ok": True, **manifest, "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
