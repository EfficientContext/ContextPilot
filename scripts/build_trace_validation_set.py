#!/usr/bin/env python3
"""Build a trace-derived ContextPilot validation set from a local Hermes DB.

Thin wrapper around :mod:`contextpilot.trace_validation.builder`. Reads the
Hermes state DB read-only and writes a FIXED JSONL corpus (raw content,
local-only / gitignored) plus a privacy-safe manifest sidecar.

Examples::

    # last 24h, conservative sampling, default local output dir
    python scripts/build_trace_validation_set.py

    # heavier sessions only, all history, to a chosen private dir
    python scripts/build_trace_validation_set.py --all-sessions \\
        --min-input-tokens 20000 --limit 50 --out .contextpilot_validation

    # exclude system/skill prompts from the corpus
    python scripts/build_trace_validation_set.py --no-system-prompt
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contextpilot.trace_validation.builder import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
