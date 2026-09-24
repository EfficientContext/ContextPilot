"""RED-phase tests for the lightweight provenance profiler / token monitor.

These tests pin two requirements for using ContextPilot purely as a standalone
token monitor / profiler against a Hermes state DB:

1. ``scripts/analyze_hermes_context_opportunities.py`` must import and expose its
   public API even when SciPy is not installed. SciPy is a heavy, optional
   ContextPilot dependency that ``contextpilot/__init__`` pulls in transitively
   (``contextpilot.pipeline`` -> ``contextpilot.context_index`` ->
   ``scipy.cluster.hierarchy``). The analyzer only reads Hermes' SQLite state
   DB and never needs the RAG pipeline, so a missing SciPy must not break it.
   This reproduces the real ``ModuleNotFoundError: No module named 'scipy'``
   observed when running the analyzer inside the Hermes venv.

2. The analyzer must offer a privacy-safe provenance profile that aggregates
   token usage per source (the token-monitor view) using numeric aggregates and
   low-cardinality source enums only -- never raw content, prompts, reasoning,
   or raw session ids/hashes.

Both tests fail today (RED): importing the script triggers
``contextpilot/__init__``, which eagerly imports ``contextpilot.pipeline`` and
hence SciPy, so the script cannot even load when SciPy is absent.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_hermes_context_opportunities.py"
)


class _BlockScipyFinder:
    """Meta-path finder that makes ``scipy`` look uninstalled.

    Raising ``ModuleNotFoundError`` from ``find_spec`` reproduces exactly what
    the Hermes venv does at import time, regardless of whether SciPy happens to
    be installed in the test environment.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy" or fullname.startswith("scipy."):
            raise ModuleNotFoundError(
                f"No module named 'scipy' (blocked for test: {fullname})"
            )
        return None


def _purge(prefixes):
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            del sys.modules[name]


_PURGE_PREFIXES = ["scipy", "contextpilot", "analyze_hermes_context_opportunities"]


def _load_analyzer_without_scipy():
    """Load the analyzer script by file path with ``scipy`` forced absent."""
    finder = _BlockScipyFinder()
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == p or name.startswith(p + ".") for p in _PURGE_PREFIXES)
    }
    sys.meta_path.insert(0, finder)
    _purge(_PURGE_PREFIXES)
    try:
        spec = importlib.util.spec_from_file_location(
            "analyze_hermes_context_opportunities", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        # Restore the import cache exactly as it was so this import-isolation
        # test cannot perturb later multiprocessing/pickling tests that depend
        # on module object identity.
        _purge(_PURGE_PREFIXES)
        sys.modules.update(saved_modules)


def test_analyzer_imports_without_scipy():
    module = _load_analyzer_without_scipy()
    # The token-monitor entry points must all be reachable without SciPy.
    assert callable(module.main)
    assert callable(module.load_tool_messages)
    assert callable(module.load_heavy_sessions)
    assert callable(module.build_report)


def test_provenance_profile_is_privacy_safe():
    module = _load_analyzer_without_scipy()
    build_provenance_profile = getattr(module, "build_provenance_profile", None)
    assert callable(build_provenance_profile), (
        "analyzer must expose build_provenance_profile() for the token-monitor view"
    )
    HeavySession = module.HeavySession
    sessions = [
        HeavySession("hash-a", "discord", 1000, 200, 6, 4, 3),
        HeavySession("hash-b", "discord", 500, 100, 4, 2, 2),
        HeavySession("hash-c", "slack", 300, 50, 3, 1, 1),
    ]
    profile = build_provenance_profile(sessions)

    by_source = {e.source: e for e in profile.by_source}
    assert set(by_source) == {"discord", "slack"}
    assert by_source["discord"].input_tokens == 1500
    assert by_source["discord"].output_tokens == 300
    assert by_source["discord"].session_count == 2
    assert by_source["slack"].input_tokens == 300
    assert by_source["slack"].session_count == 1

    # Provenance output is numeric aggregates + low-cardinality source enums only;
    # no raw content/prompts/reasoning and no raw session ids/hashes may leak.
    data = dataclasses.asdict(profile)
    module._assert_no_forbidden_keys(data)
    blob = repr(data)
    for raw_hash in ("hash-a", "hash-b", "hash-c"):
        assert raw_hash not in blob
