"""ContextPilot telemetry parsing (metadata-only).

Aggregates the metadata-only ContextPilot telemetry file into a privacy-safe
``TelemetryCoverage``. Never reads message content; tolerates malformed lines
by skipping and counting them.
"""
from __future__ import annotations

import json
from pathlib import Path

from .db import _window_cutoff
from .models import EST_CHARS_PER_TOKEN, TelemetryCoverage


def parse_telemetry(
    telemetry_path: Path,
    *,
    since_hours: int,
    total_input_tokens: int,
    all_sessions: bool = False,
) -> TelemetryCoverage:
    """Aggregate the metadata-only ContextPilot telemetry file.

    Tolerates malformed lines (non-JSON, non-dict, missing counters) by
    skipping and counting them. Never reads message content. With
    ``all_sessions=True`` the time window is ignored.
    """
    events = 0
    chars = 0
    tokens = 0
    malformed = 0
    if telemetry_path and telemetry_path.exists():
        cutoff = _window_cutoff(since_hours, all_sessions)
        with telemetry_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    malformed += 1
                    continue
                if not isinstance(record, dict):
                    malformed += 1
                    continue
                ts = record.get("ts")
                if cutoff is not None and isinstance(ts, (int, float)) and ts < cutoff:
                    continue
                cs = record.get("chars_saved")
                if not isinstance(cs, (int, float)):
                    malformed += 1
                    continue
                saved_tokens = record.get("tokens_saved")
                events += 1
                chars += int(cs)
                tokens += (
                    int(saved_tokens)
                    if isinstance(saved_tokens, (int, float))
                    else int(cs) // EST_CHARS_PER_TOKEN
                )

    denom = tokens + total_input_tokens
    coverage = (tokens / denom * 100.0) if denom else 0.0
    avg = (tokens / events) if events else 0.0
    return TelemetryCoverage(
        events=events,
        chars_saved=chars,
        tokens_saved=tokens,
        avg_tokens_saved_per_event=round(avg, 2),
        coverage_ratio_pct=round(coverage, 2),
        malformed_records_skipped=malformed,
    )
