#!/usr/bin/env python3
"""Run ContextPilot trace validation against a fixed local corpus.

Thin wrapper around :mod:`contextpilot.trace_validation.runner`. Replays the
JSONL corpus through ContextPilot's optimization in a baseline (``off``) mode vs
a configured candidate mode, checks accuracy-preservation invariants, prints a
privacy-safe JSON/Markdown summary, and exits non-zero on any gate failure.

Examples::

    # validate the canary candidate against a corpus, JSON gate report
    python scripts/run_trace_validation.py \\
        ~/contextpilot/validation_sets/validation_set_2026-06-14.jsonl \\
        --candidate-mode canary

    # use the resolved CONTEXTPILOT_PROMPT_DEDUP_MODE env, Markdown output
    CONTEXTPILOT_PROMPT_DEDUP_MODE=canary \\
    python scripts/run_trace_validation.py <corpus.jsonl> --format markdown

    # with exact-token accounting
    python scripts/run_trace_validation.py <corpus.jsonl> \\
        --candidate-mode canary --tokenizer tiktoken:cl100k_base
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contextpilot.trace_validation.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
