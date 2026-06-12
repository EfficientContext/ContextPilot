#!/usr/bin/env python3
"""Show how many tokens ContextPilot saved — a simple, user-facing summary.

Reads only the metadata-only telemetry file ContextPilot writes
(``~/.hermes/contextpilot/telemetry.jsonl`` by default) and prints a concise
savings summary. It never reads conversation messages, system prompts,
reasoning, or tool payloads — the telemetry file only contains numeric counters.

This is the lightweight companion to the heavier
``analyze_hermes_context_opportunities.py`` analyzer: no Hermes DB access, no
content scanning, no Hermes internals imported. Just "how much did I save?".

Examples::

    python scripts/contextpilot_savings.py                 # last 24h
    python scripts/contextpilot_savings.py --all-time       # everything
    python scripts/contextpilot_savings.py --since-hours 1  # last hour
    python scripts/contextpilot_savings.py --format json    # machine readable
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_TELEMETRY_FILE = Path.home() / ".hermes" / "contextpilot" / "telemetry.jsonl"
DEFAULT_SINCE_HOURS = 24


def summarize_telemetry(
    telemetry_path: Path,
    *,
    since_hours: float | None,
) -> Dict[str, Any]:
    """Aggregate the metadata-only telemetry file into a savings summary.

    ``since_hours`` of ``None`` means all-time (no time filtering). Only numeric
    counters are read; no conversation/tool/system content can be present in the
    file, and this function never emits raw text regardless.

    Malformed JSONL lines (and non-dict / non-numeric records) are skipped and
    counted under ``skipped_lines``.
    """
    result: Dict[str, Any] = {
        "telemetry_file": str(telemetry_path),
        "file_exists": telemetry_path.exists(),
        "all_time": since_hours is None,
        "since_hours": since_hours,
        "window_start_iso": None,
        "events": 0,
        "chars_saved": 0,
        "tokens_saved": 0,
        "avg_tokens_per_event": None,
        "skipped_lines": 0,
    }

    if not telemetry_path.exists():
        return result

    if since_hours is None:
        cutoff = None
    else:
        cutoff = dt.datetime.now(dt.timezone.utc).timestamp() - since_hours * 3600
        result["window_start_iso"] = (
            dt.datetime.fromtimestamp(cutoff, dt.timezone.utc).isoformat()
        )

    events = 0
    chars = 0
    tokens = 0
    skipped = 0
    with telemetry_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                skipped += 1
                continue
            if not isinstance(record, dict):
                skipped += 1
                continue
            ts = record.get("ts")
            if cutoff is not None:
                if not isinstance(ts, (int, float)):
                    skipped += 1
                    continue
                if ts < cutoff:
                    continue
            cs = record.get("chars_saved")
            if not isinstance(cs, (int, float)) or cs < 0:
                skipped += 1
                continue
            saved_tokens = record.get("tokens_saved")
            if isinstance(saved_tokens, (int, float)) and saved_tokens < 0:
                skipped += 1
                continue
            events += 1
            chars += int(cs)
            tokens += (
                int(saved_tokens)
                if isinstance(saved_tokens, (int, float))
                else int(cs) // 4
            )

    result["events"] = events
    result["chars_saved"] = chars
    result["tokens_saved"] = tokens
    result["skipped_lines"] = skipped
    if events > 0:
        result["avg_tokens_per_event"] = round(tokens / events, 1)
    return result


def _format_window(summary: Dict[str, Any]) -> str:
    if summary["all_time"]:
        return "all time"
    hours = summary["since_hours"]
    if isinstance(hours, float) and hours.is_integer():
        hours = int(hours)
    return f"last {hours}h"


def render_text(summary: Dict[str, Any]) -> str:
    window = _format_window(summary)
    path = summary["telemetry_file"]

    if not summary["file_exists"]:
        return (
            "No ContextPilot telemetry found.\n"
            f"  Looked for: {path}\n\n"
            "To start seeing token savings:\n"
            "  1. Enable ContextPilot in Hermes "
            "(hermes plugins → toggle contextpilot).\n"
            "  2. Restart Hermes.\n"
            "  3. Run a workload that reads the same content more than once "
            "(e.g. read a file, then read it again).\n"
            "ContextPilot only saves tokens when content repeats across turns."
        )

    if summary["events"] == 0:
        return (
            f"No ContextPilot savings recorded in the {window} window.\n"
            f"  Telemetry file: {path}\n\n"
            "If you expected savings:\n"
            "  - Make sure ContextPilot is enabled and Hermes was restarted.\n"
            "  - Try --all-time to widen the window.\n"
            "  - Run a workload with repeated content (read the same file "
            "twice); savings only fire when content repeats across turns."
        )

    lines = [
        f"ContextPilot token savings ({window})",
        f"  Events:                {summary['events']}",
        f"  Chars saved:           {summary['chars_saved']:,}",
        f"  Estimated tokens saved: ~{summary['tokens_saved']:,}",
    ]
    if summary["avg_tokens_per_event"] is not None:
        lines.append(
            f"  Avg tokens/event:      ~{summary['avg_tokens_per_event']:,}"
        )
    lines.append(f"  Telemetry file:        {path}")
    if summary["skipped_lines"]:
        lines.append(
            f"  (skipped {summary['skipped_lines']} malformed telemetry line(s))"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show how many tokens ContextPilot saved (metadata-only).",
    )
    parser.add_argument(
        "--telemetry-file",
        type=Path,
        default=DEFAULT_TELEMETRY_FILE,
        help="metadata-only ContextPilot telemetry file "
        "(default: ~/.hermes/contextpilot/telemetry.jsonl)",
    )
    parser.add_argument(
        "--since-hours",
        type=float,
        default=DEFAULT_SINCE_HOURS,
        help="only count savings in the last N hours (default: 24)",
    )
    parser.add_argument(
        "--all-time",
        action="store_true",
        help="count all savings, ignoring --since-hours",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="output format (default: text)",
    )
    args = parser.parse_args(argv)

    since_hours = None if args.all_time else args.since_hours
    summary = summarize_telemetry(args.telemetry_file, since_hours=since_hours)

    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
