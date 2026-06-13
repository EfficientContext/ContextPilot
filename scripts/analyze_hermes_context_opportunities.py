#!/usr/bin/env python3
"""Privacy-safe Hermes context opportunity analyzer for ContextPilot.

This is a thin wrapper around the :mod:`contextpilot.hermes_opportunities`
package, kept at its historical script path so existing cron jobs and callers
keep working. All logic lives in the package; see its modules for details.

Unlike ``hermes_contextpilot_monitor.py`` (which never reads message bodies),
this analyzer *does* inspect message content and tool outputs in order to find
concrete token-reduction opportunities, but it reads content only in-memory to
compute salted hashes and aggregate counters. Reports never contain raw
message/tool text, system prompts, or raw session ids -- only salted SHA-256
fingerprints and numeric aggregates. This makes it safe to run continuously
from a cron job and ship the reports.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a standalone script (``python scripts/analyze_...py``) by
# making the repo root importable, so ``contextpilot`` resolves without an
# editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-export the full public API at the historical module path. Tests and other
# callers that load this script by file path keep accessing every name (incl.
# the in-memory carriers and shadow-mode enums) via this module.
from contextpilot.hermes_opportunities import *  # noqa: F401,F403,E402
from contextpilot.hermes_opportunities import (  # noqa: F401,E402
    EST_CHARS_PER_TOKEN,
    FORBIDDEN_OUTPUT_KEYS,
    _assert_no_forbidden_keys,
    _est_tokens,
    _LLMContent,
    _ROUTABLE_LABELS,
    _salt_fingerprint,
    _salted_hash,
    _ToolMessage,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
