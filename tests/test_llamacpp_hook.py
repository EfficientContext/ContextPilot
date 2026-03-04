"""
Tests for the native llama.cpp hook (contextpilot/_llamacpp_hook.py).

All tests are offline — no real llama-server required.
The C++ compilation tests require clang++ (macOS) or g++ (Linux).
"""

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from contextpilot._llamacpp_hook import (
    build,
    preload_env,
    launch,
    _IS_MAC,
    _PRELOAD_VAR,
    _LIB_SUFFIX,
    _COMPILER,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def compiled_lib() -> Path:
    """Compile the hook once for the whole test module."""
    return build(force=True)


# ── Build tests ───────────────────────────────────────────────────────────────

class TestBuild:
    def test_returns_path(self, compiled_lib):
        assert isinstance(compiled_lib, Path)

    def test_file_exists(self, compiled_lib):
        assert compiled_lib.exists()

    def test_correct_suffix(self, compiled_lib):
        assert compiled_lib.suffix == _LIB_SUFFIX

    def test_is_in_tmp(self, compiled_lib):
        assert str(compiled_lib).startswith(tempfile.gettempdir())

    def test_cached_on_second_call(self, compiled_lib):
        lib2 = build()
        assert lib2 == compiled_lib

    def test_force_recompiles(self, compiled_lib):
        lib2 = build(force=True)
        assert lib2.exists()
        # Same path, but file is re-created
        assert lib2 == compiled_lib

    def test_bad_source_raises_runtime_error(self):
        import contextpilot._llamacpp_hook as mod
        original = mod._HOOK_SRC
        mod._HOOK_SRC = "this is not valid C++ @#$%"
        mod._cached_lib = None  # force recompile
        try:
            with pytest.raises(RuntimeError, match="Failed to compile"):
                build(force=True)
        finally:
            mod._HOOK_SRC = original
            mod._cached_lib = None
            build(force=True)  # restore valid cached lib


# ── Symbol export tests ───────────────────────────────────────────────────────

class TestSymbolExport:
    def test_exports_c_linkage_symbol(self, compiled_lib):
        """The interposed symbol must have C linkage (no C++ name-mangling)."""
        result = subprocess.run(
            ["nm", str(compiled_lib)],
            capture_output=True, text=True,
        )
        # C linkage on macOS: _llama_memory_seq_rm (prefixed with _)
        # C linkage on Linux: llama_memory_seq_rm
        assert "llama_memory_seq_rm" in result.stdout

    def test_symbol_is_exported_text(self, compiled_lib):
        """Symbol must be in the text (code) section — T in nm output."""
        result = subprocess.run(
            ["nm", str(compiled_lib)],
            capture_output=True, text=True,
        )
        lines = result.stdout.splitlines()
        matching = [l for l in lines if "llama_memory_seq_rm" in l]
        assert any(" T " in l for l in matching), \
            f"Expected 'T' (text) symbol, got: {matching}"

    def test_no_mangled_cxx_symbol(self, compiled_lib):
        """Must NOT export the C++ mangled version of the function."""
        result = subprocess.run(
            ["nm", str(compiled_lib)],
            capture_output=True, text=True,
        )
        # C++ mangled name would look like _Z19llama_memory_seq_rm...
        assert "_Z19llama_memory_seq_rm" not in result.stdout


# ── preload_env tests ─────────────────────────────────────────────────────────

class TestPreloadEnv:
    def test_contains_preload_var(self, compiled_lib):
        env = preload_env(hook_lib=compiled_lib, index_url="http://localhost:8765")
        assert _PRELOAD_VAR in env

    def test_preload_var_points_to_lib(self, compiled_lib):
        env = preload_env(hook_lib=compiled_lib, index_url="http://localhost:8765")
        assert env[_PRELOAD_VAR] == str(compiled_lib)

    def test_index_url_passed_explicitly(self, compiled_lib):
        env = preload_env(hook_lib=compiled_lib, index_url="http://myserver:9999")
        assert env.get("CONTEXTPILOT_INDEX_URL") == "http://myserver:9999"

    def test_index_url_from_env_var(self, compiled_lib):
        with patch.dict(os.environ, {"CONTEXTPILOT_INDEX_URL": "http://envserver:8765"}):
            env = preload_env(hook_lib=compiled_lib)
        assert env.get("CONTEXTPILOT_INDEX_URL") == "http://envserver:8765"

    def test_no_index_url_if_not_set(self, compiled_lib):
        env_without = {k: v for k, v in os.environ.items()
                       if k != "CONTEXTPILOT_INDEX_URL"}
        with patch.dict(os.environ, env_without, clear=True):
            env = preload_env(hook_lib=compiled_lib)
        assert "CONTEXTPILOT_INDEX_URL" not in env

    def test_builds_lib_if_not_provided(self):
        env = preload_env(index_url="http://localhost:8765")
        assert _PRELOAD_VAR in env
        assert Path(env[_PRELOAD_VAR]).exists()

    def test_explicit_url_overrides_env(self, compiled_lib):
        with patch.dict(os.environ, {"CONTEXTPILOT_INDEX_URL": "http://fromenv:1"}):
            env = preload_env(compiled_lib, index_url="http://explicit:2")
        assert env["CONTEXTPILOT_INDEX_URL"] == "http://explicit:2"


# ── launch tests ──────────────────────────────────────────────────────────────

class TestLaunch:
    def test_returns_popen(self, compiled_lib):
        proc = launch(
            [sys.executable, "-c", "import sys; sys.exit(0)"],
            index_url="http://localhost:8765",
        )
        rc = proc.wait()
        assert rc == 0

    def test_preload_var_set_in_child(self, compiled_lib):
        """Child process should see DYLD_INSERT_LIBRARIES / LD_PRELOAD."""
        proc = launch(
            [sys.executable, "-c",
             f"import os, sys; sys.exit(0 if '{_PRELOAD_VAR}' in os.environ else 1)"],
            index_url="http://localhost:8765",
        )
        assert proc.wait() == 0

    def test_index_url_set_in_child(self, compiled_lib):
        proc = launch(
            [sys.executable, "-c",
             "import os, sys; "
             "sys.exit(0 if os.environ.get('CONTEXTPILOT_INDEX_URL') == "
             "'http://localhost:8765' else 1)"],
            index_url="http://localhost:8765",
        )
        assert proc.wait() == 0

    def test_extra_env_forwarded(self, compiled_lib):
        proc = launch(
            [sys.executable, "-c",
             "import os, sys; sys.exit(0 if os.environ.get('MY_VAR') == 'hello' else 1)"],
            index_url="http://localhost:8765",
            extra_env={"MY_VAR": "hello"},
        )
        assert proc.wait() == 0

    def test_popen_kwargs_forwarded(self, compiled_lib):
        proc = launch(
            [sys.executable, "-c", "print('hi')"],
            index_url="http://localhost:8765",
            stdout=subprocess.PIPE,
        )
        out, _ = proc.communicate()
        assert b"hi" in out


# ── CLI entry-point tests ─────────────────────────────────────────────────────

class TestCLI:
    def test_build_only_prints_path(self):
        result = subprocess.run(
            [sys.executable, "-m", "contextpilot._llamacpp_hook"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "contextpilot_llama_hook" in result.stdout

    def test_launcher_mode_exits_with_child_code(self):
        """When given a command, CLI should exec it and relay its exit code."""
        result = subprocess.run(
            [sys.executable, "-m", "contextpilot._llamacpp_hook",
             sys.executable, "-c", "import sys; sys.exit(42)"],
            capture_output=True,
            env={**os.environ, "CONTEXTPILOT_INDEX_URL": "http://localhost:8765"},
        )
        assert result.returncode == 42

    def test_launcher_mode_sets_preload(self):
        """Child launched via CLI must see the preload env var."""
        result = subprocess.run(
            [sys.executable, "-m", "contextpilot._llamacpp_hook",
             sys.executable, "-c",
             f"import os, sys; sys.exit(0 if '{_PRELOAD_VAR}' in os.environ else 1)"],
            capture_output=True,
            env={**os.environ, "CONTEXTPILOT_INDEX_URL": "http://localhost:8765"},
        )
        assert result.returncode == 0

    def test_force_flag_accepted(self):
        result = subprocess.run(
            [sys.executable, "-m", "contextpilot._llamacpp_hook", "--force"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
