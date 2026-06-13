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
        # ``tokens_saved`` is a LEGACY DERIVED estimate (chars/4), NOT a real
        # tokenizer/API count. ``tokens_saved_method`` makes that explicit so it
        # is never mistaken for actual tokens.
        "tokens_saved": 0,
        "tokens_saved_method": "estimated_chars_div_4",
        "avg_tokens_per_event": None,
        # EXACT tokenizer measurements, surfaced separately and only populated
        # from records that carry ``actual_token_status == "available"``. No
        # fake/derived numbers are ever written into these fields.
        "actual_token_status": "unavailable",
        "actual_token_events": 0,
        "actual_tokens_before": 0,
        "actual_tokens_after": 0,
        "actual_tokens_saved": 0,
        "actual_tokenizer_backends": [],
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
    actual_events = 0
    actual_before = 0
    actual_after = 0
    actual_saved = 0
    actual_backends: set[str] = set()
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

            # EXACT tokenizer measurement, only when the writer marked it as
            # available. Anything else (missing/unavailable) is left out -- we
            # never substitute the chars/4 estimate into the actual-token totals.
            if record.get("actual_token_status") == "available":
                ats = record.get("actual_tokens_saved")
                if isinstance(ats, (int, float)):
                    actual_events += 1
                    actual_saved += int(ats)
                    atb = record.get("actual_tokens_before")
                    if isinstance(atb, (int, float)):
                        actual_before += int(atb)
                    ata = record.get("actual_tokens_after")
                    if isinstance(ata, (int, float)):
                        actual_after += int(ata)
                    backend = record.get("actual_tokenizer_backend")
                    if isinstance(backend, str) and backend:
                        actual_backends.add(backend)

    result["events"] = events
    result["chars_saved"] = chars
    result["tokens_saved"] = tokens
    result["skipped_lines"] = skipped
    if events > 0:
        result["avg_tokens_per_event"] = round(tokens / events, 1)
    if actual_events > 0:
        result["actual_token_status"] = "available"
        result["actual_token_events"] = actual_events
        result["actual_tokens_before"] = actual_before
        result["actual_tokens_after"] = actual_after
        result["actual_tokens_saved"] = actual_saved
        result["actual_tokenizer_backends"] = sorted(actual_backends)
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
        f"ContextPilot savings ({window})",
        f"  Events:                  {summary['events']}",
        f"  Chars saved:             {summary['chars_saved']:,}",
        # Make provenance unmistakable: this is a chars/4 estimate, not real tokens.
        f"  Est. tokens saved (chars/4, derived): {summary['tokens_saved']:,}",
    ]
    if summary["avg_tokens_per_event"] is not None:
        lines.append(
            f"  Avg est. tokens/event:   {summary['avg_tokens_per_event']:,}"
        )
    # Actual tokenizer tokens are shown ONLY when the telemetry recorded them
    # from an exact tokenizer backend; otherwise we say so rather than fake it.
    if summary["actual_token_status"] == "available":
        backends = ", ".join(summary["actual_tokenizer_backends"]) or "unknown"
        lines.append(
            f"  Actual tokens saved (tokenizer): {summary['actual_tokens_saved']:,}"
        )
        lines.append(
            f"    backend: {backends} | status: available | "
            f"events: {summary['actual_token_events']}"
        )
    else:
        lines.append(
            "  Actual tokens saved (tokenizer): unavailable "
            "(no exact tokenizer backend recorded)"
        )
    lines.append(f"  Telemetry file:          {path}")
    if summary["skipped_lines"]:
        lines.append(
            f"  (skipped {summary['skipped_lines']} malformed telemetry line(s))"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Show ContextPilot processed-payload savings (metadata-only); "
            "exact tokenizer tokens are shown only when telemetry recorded them."
        ),
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
