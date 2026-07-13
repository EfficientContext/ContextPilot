"""Command-line entry point for the Hermes context opportunity analyzer.

Thin orchestration layer: parse args, run the read-only loaders, build the
privacy-safe report, and write it. Hardened for unattended cron use -- on any
failure it emits only the exception class name (never a traceback that could
echo the DB path or SQL) and a non-zero exit code.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .aggregation import DEFAULT_MIN_ARTIFACT_CHARS
from .db import (
    load_heavy_sessions,
    load_llm_bound_content,
    load_tool_messages,
    total_input_tokens,
)
from .models import (
    DEFAULT_LARGE_OUTPUT_CHARS,
    DEFAULT_MIN_BLOCK_CHARS,
    DEFAULT_MIN_BLOCK_REPEAT,
    DEFAULT_TOP_N,
)
from .report import build_report, write_report
from .telemetry import parse_telemetry
from .tokenizer import resolve_tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Privacy-safe Hermes context opportunity analyzer for ContextPilot."
    )
    parser.add_argument("--state-db", type=Path, default=Path("/root/.hermes/state.db"))
    parser.add_argument(
        "--telemetry-file",
        type=Path,
        default=Path.home() / ".hermes" / "contextpilot" / "telemetry.jsonl",
        help="metadata-only ContextPilot telemetry file",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path.home() / "contextpilot" / "opportunities"
    )
    parser.add_argument("--since-hours", type=int, default=24)
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="ignore --since-hours; scan all non-archived sessions and active messages",
    )
    parser.add_argument(
        "--salt",
        default="contextpilot-hermes-opportunity-v1",
        help="salt for stable per-install content/session fingerprints",
    )
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--min-block-chars", type=int, default=DEFAULT_MIN_BLOCK_CHARS)
    parser.add_argument("--min-block-repeat", type=int, default=DEFAULT_MIN_BLOCK_REPEAT)
    parser.add_argument(
        "--large-output-chars", type=int, default=DEFAULT_LARGE_OUTPUT_CHARS
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--disable-worker-routing-shadow",
        action="store_true",
        help=(
            "skip the shadow-mode Worker Context Routing classification "
            "(P0 data collection; enabled by default, never prunes context)"
        ),
    )
    parser.add_argument(
        "--disable-parent-aggregation",
        action="store_true",
        help=(
            "skip the shadow-mode Parent Aggregation Artifact telemetry "
            "(P0 telemetry only; enabled by default, never dedups/replaces context)"
        ),
    )
    parser.add_argument(
        "--min-artifact-chars", type=int, default=DEFAULT_MIN_ARTIFACT_CHARS
    )
    parser.add_argument(
        "--disable-prompt-duplicate-shadow",
        action="store_true",
        help=(
            "skip the advisory system/skill prompt duplicate-block scan "
            "(enabled by default; advisory only, never rewrites/dedups prompts)"
        ),
    )
    parser.add_argument(
        "--disable-prompt-dedup-ab",
        action="store_true",
        help=(
            "skip the offline prompt-dedup A/B simulation section "
            "(enabled by default; offline simulation only, never mutates prompts; "
            "this is the evidence gate before any canary replace)"
        ),
    )
    parser.add_argument(
        "--prompt-dedup-tokenizer",
        default=None,
        help=(
            "opt-in exact tokenizer backend for the prompt-dedup A/B simulation, "
            "e.g. 'tiktoken:cl100k_base' (off by default; without it the A/B "
            "section reports tokenizer_status=unavailable and no actual-token fields)"
        ),
    )
    args = parser.parse_args(argv)

    if not args.state_db.exists():
        raise SystemExit(f"Hermes state DB not found: {args.state_db}")

    # Harden for unattended cron use: never dump a traceback (which would echo
    # the DB path / SQL); emit only the exception class name and a non-zero code.
    try:
        # Opt-in tokenizer; off by default -> A/B simulation reports actual tokens
        # as unavailable rather than fabricating chars/4 figures.
        dedup_ab_tokenizer = resolve_tokenizer(args.prompt_dedup_tokenizer)
        tool_messages = load_tool_messages(
            args.state_db, since_hours=args.since_hours, all_sessions=args.all_sessions
        )
        llm_contents = load_llm_bound_content(
            args.state_db, since_hours=args.since_hours, all_sessions=args.all_sessions
        )
        heavy_sessions = load_heavy_sessions(
            args.state_db,
            since_hours=args.since_hours,
            salt=args.salt,
            top_n=args.top_n,
            all_sessions=args.all_sessions,
        )
        total_input = total_input_tokens(
            args.state_db, since_hours=args.since_hours, all_sessions=args.all_sessions
        )
        telemetry = parse_telemetry(
            args.telemetry_file,
            since_hours=args.since_hours,
            total_input_tokens=total_input,
            all_sessions=args.all_sessions,
        )
        report = build_report(
            date=args.date,
            since_hours=args.since_hours,
            salt=args.salt,
            tool_messages=tool_messages,
            heavy_sessions=heavy_sessions,
            telemetry=telemetry,
            llm_contents=llm_contents,
            all_sessions=args.all_sessions,
            min_block_chars=args.min_block_chars,
            min_block_repeat=args.min_block_repeat,
            large_output_chars=args.large_output_chars,
            top_n=args.top_n,
            worker_routing_shadow=not args.disable_worker_routing_shadow,
            parent_aggregation_shadow=not args.disable_parent_aggregation,
            prompt_duplicate_shadow=not args.disable_prompt_duplicate_shadow,
            prompt_dedup_ab=not args.disable_prompt_dedup_ab,
            prompt_dedup_ab_tokenizer=dedup_ab_tokenizer,
            min_artifact_chars=args.min_artifact_chars,
        )
        json_path, md_path = write_report(report, args.out_dir)
    except Exception as exc:  # noqa: BLE001 - cron-safe: report class only, no payload
        print(json.dumps({"ok": False, "error": type(exc).__name__}))
        return 1

    print(
        json.dumps(
            {"ok": True, "json": str(json_path), "markdown": str(md_path)},
            ensure_ascii=False,
        )
    )
    return 0
