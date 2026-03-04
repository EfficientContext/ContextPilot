"""
Native C++ hook for llama.cpp KV-cache eviction tracking.

Compiles a small shared library at runtime that intercepts llama_memory_seq_rm()
via DYLD_INSERT_LIBRARIES (macOS) or LD_PRELOAD (Linux). When a slot's KV cache
is fully cleared, the hook immediately posts POST /evict_slot to the ContextPilot
server — exact, zero-latency eviction signals with no polling overhead.

How it works
------------
llama-server stores each concurrent request in a *slot* (one per ``--parallel N``
slot).  The slot ID doubles as the KV-cache sequence ID (``llama_seq_id``).
When a slot's cache is discarded before a new request begins,
``llama_memory_seq_rm(mem, seq_id, -1, -1)`` is called with p0=-1, p1=-1 (full
clear).  The hook detects this and posts::

    POST $CONTEXTPILOT_INDEX_URL/evict_slot
    {"slot_id": <seq_id>}

The ContextPilot HTTP server maps ``slot_id → request_id`` (via prior
``POST /register_slot`` calls) and evicts the stale entry from the live index.

Activation (identical pattern to SGLang / vLLM hooks)
------------------------------------------------------
After ``pip install contextpilot``, use the ``contextpilot-llama-server``
console script — one env var, same pattern as SGLang / vLLM::

    CONTEXTPILOT_INDEX_URL=http://localhost:8765 contextpilot-llama-server \\
        -m models/Qwen3-8B-Q4_K_M.gguf \\
        --host 0.0.0.0 --port 8889 \\
        -ngl 99 --cache-reuse 256 --parallel 4 -c 32768

The ``llama-server`` binary is resolved from PATH.  Override with
``LLAMA_SERVER_BIN=/custom/path/llama-server`` for non-standard installs.

Or use the transparent launcher directly (useful when the binary is not in PATH)::

    CONTEXTPILOT_INDEX_URL=http://localhost:8765 \\
    python -m contextpilot._llamacpp_hook \\
        /path/to/llama-server -m model.gguf --port 8889 ...

Or from Python::

    from contextpilot._llamacpp_hook import launch

    proc = launch(
        ["/path/to/llama-server", "-m", "model.gguf", ...],
        index_url="http://localhost:8765",
    )

Requirements
------------
- macOS: ``clang++`` (Xcode Command Line Tools: ``xcode-select --install``)
- Linux: ``g++``
- No other dependencies — the hook uses only POSIX sockets.
"""

import os
import platform
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# ── Embedded C++ hook source ──────────────────────────────────────────────────
#
# Interposes llama_memory_seq_rm() exported by libllama.dylib / libllama.so.
# RTLD_NEXT resolves the real symbol in libllama so we call through.
# The HTTP POST uses a raw POSIX socket (no libcurl needed).
#
_HOOK_SRC = textwrap.dedent(r"""
    #include <dlfcn.h>
    #include <netdb.h>
    #include <netinet/in.h>
    #include <stdint.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <sys/socket.h>
    #include <sys/time.h>
    #include <unistd.h>

    /* Minimal llama.cpp public API types (from include/llama.h).
       We only need the opaque pointer and the two integer typedefs. */
    typedef int32_t llama_pos;
    typedef int32_t llama_seq_id;
    struct llama_memory_i;
    typedef struct llama_memory_i * llama_memory_t;

    typedef bool (*_cp_seq_rm_fn_t)(
            llama_memory_t, llama_seq_id, llama_pos, llama_pos);

    /* ── Fire-and-forget HTTP POST via raw POSIX socket ─────────────── */
    static void _cp_post_evict_slot(int slot_id) {
        const char *env = getenv("CONTEXTPILOT_INDEX_URL");
        if (!env || !*env) return;

        /* Parse  http://host:port  (no path, no HTTPS). */
        const char *h = env;
        if (__builtin_expect(strncmp(h, "http://", 7) == 0, 1)) h += 7;

        char host[256] = {0};
        int  port = 8765;
        const char *col = strrchr(h, ':');
        if (col && col > h) {
            size_t hlen = (size_t)(col - h);
            if (hlen >= sizeof(host)) hlen = sizeof(host) - 1;
            memcpy(host, h, hlen);
            port = atoi(col + 1);
        } else {
            strncpy(host, h, sizeof(host) - 1);
        }

        /* Build HTTP/1.0 request (no keep-alive — simplest possible).
           Use the existing /evict endpoint with a slot_{N} request_id
           so no extra registry endpoints are needed on the server. */
        char body[128];
        snprintf(body, sizeof(body), "{\"request_ids\":[\"slot_%d\"]}", slot_id);
        size_t body_len = strlen(body);

        char req[512];
        int req_len = snprintf(req, sizeof(req),
            "POST /evict HTTP/1.0\r\n"
            "Host: %s:%d\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: %zu\r\n"
            "\r\n%s",
            host, port, body_len, body);
        if (req_len <= 0 || (size_t)req_len >= sizeof(req)) return;

        /* Resolve hostname (cached by libc after first call for "localhost"). */
        struct hostent *he = gethostbyname(host);
        if (!he) return;

        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) return;

        /* Short timeout: never block llama-server's inference loop. */
        struct timeval tv = {0, 200000}; /* 200 ms */
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port   = htons((uint16_t)port);
        memcpy(&addr.sin_addr, he->h_addr_list[0], (size_t)he->h_length);

        if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0)
            (void)send(sock, req, (size_t)req_len, 0);
        /* We don't read the response — fire and forget. */
        close(sock);
    }

    /* ── Interposed function ─────────────────────────────────────────── */
    /*                                                                     */
    /*  llama_memory_seq_rm is exported by libllama.dylib / libllama.so   */
    /*  with C linkage (_llama_memory_seq_rm on macOS).                   */
    /*  extern "C" prevents C++ name-mangling so the dynamic linker sees   */
    /*  the same symbol name and routes calls through our version first.   */
    /*  RTLD_NEXT then finds the real symbol in libllama.                  */
    /*                                                                     */
    /*  p0 < 0 && p1 < 0  ==>  full-sequence clear  ==>  KV cache gone.  */
    /*                                                                     */
    extern "C" bool llama_memory_seq_rm(
            llama_memory_t   mem,
            llama_seq_id     seq_id,
            llama_pos        p0,
            llama_pos        p1)
    {
        static _cp_seq_rm_fn_t _real = NULL;
        if (__builtin_expect(!_real, 0))
            _real = (_cp_seq_rm_fn_t)dlsym(RTLD_NEXT, "llama_memory_seq_rm");

        bool result = _real(mem, seq_id, p0, p1);

        if (p0 < 0 && p1 < 0 && result)
            _cp_post_evict_slot((int)seq_id);

        return result;
    }
""")

# ── Platform constants ────────────────────────────────────────────────────────
_IS_MAC      = platform.system() == "Darwin"
_PRELOAD_VAR = "DYLD_INSERT_LIBRARIES" if _IS_MAC else "LD_PRELOAD"
_LIB_SUFFIX  = ".dylib"               if _IS_MAC else ".so"
_COMPILER    = "clang++"              if _IS_MAC else "g++"

_cached_lib: "Path | None" = None


# ── Public API ────────────────────────────────────────────────────────────────

def build(force: bool = False) -> Path:
    """
    Compile the C++ hook into a shared library and return its path.

    The result is written to ``/tmp/contextpilot_llama_hook.dylib`` (or ``.so``
    on Linux) and cached in memory.  Subsequent calls return the cached path
    unless *force=True*.

    Raises ``RuntimeError`` if the compiler is not found or compilation fails.
    """
    global _cached_lib
    if _cached_lib and _cached_lib.exists() and not force:
        return _cached_lib

    out = Path(tempfile.gettempdir()) / f"contextpilot_llama_hook{_LIB_SUFFIX}"
    src = out.with_suffix(".cpp")
    src.write_text(_HOOK_SRC)

    flags = ["-shared", "-fPIC", "-O2"]
    if _IS_MAC:
        # -dynamiclib produces a proper .dylib; -undefined dynamic_lookup
        # lets RTLD_NEXT work without linking against libllama directly.
        flags += ["-dynamiclib", "-undefined", "dynamic_lookup"]

    result = subprocess.run(
        [_COMPILER, *flags, "-o", str(out), str(src)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"[ContextPilot] Failed to compile llama hook.\n"
            f"Compiler: {_COMPILER}\n"
            f"stderr:\n{result.stderr}"
        )

    _cached_lib = out
    return out


def preload_env(
    hook_lib: "Path | None" = None,
    index_url: "str | None" = None,
) -> "dict[str, str]":
    """
    Return the extra environment variables needed to inject the hook when
    launching llama-server.

    Parameters
    ----------
    hook_lib:
        Path returned by :func:`build`.  Calls :func:`build` automatically
        if ``None``.
    index_url:
        Value for ``CONTEXTPILOT_INDEX_URL``.  Falls back to the current
        process environment if not specified.

    Returns
    -------
    dict
        Merge this into ``os.environ`` (or ``subprocess.Popen(env=...)``)
        before starting llama-server.
    """
    lib = hook_lib or build()
    env: "dict[str, str]" = {_PRELOAD_VAR: str(lib)}
    url = index_url or os.environ.get("CONTEXTPILOT_INDEX_URL")
    if url:
        env["CONTEXTPILOT_INDEX_URL"] = url
    return env


def launch(
    cmd: "list[str]",
    index_url: "str | None" = None,
    extra_env: "dict[str, str] | None" = None,
    **popen_kwargs,
) -> "subprocess.Popen":
    """
    Launch *cmd* with the native hook injected via ``DYLD_INSERT_LIBRARIES``
    (macOS) or ``LD_PRELOAD`` (Linux).

    Parameters
    ----------
    cmd:
        Command to run, e.g. ``["/path/to/llama-server", "-m", "model.gguf", ...]``.
    index_url:
        ``CONTEXTPILOT_INDEX_URL`` to pass to the hook.  If ``None``, the
        current environment value is used.
    extra_env:
        Additional environment variables to set (merged after hook env).
    **popen_kwargs:
        Forwarded to ``subprocess.Popen``.

    Returns
    -------
    subprocess.Popen
        The launched process.

    Example
    -------
    ::

        proc = launch(
            [
                "/Users/me/llama.cpp/build/bin/llama-server",
                "-m", "models/Qwen3-8B-Q4_K_M.gguf",
                "--host", "0.0.0.0", "--port", "8889",
                "-ngl", "99", "--cache-reuse", "256",
                "--parallel", "4", "-c", "32768",
            ],
            index_url="http://localhost:8765",
        )
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    """
    lib = build()
    env = {**os.environ, **preload_env(lib, index_url), **(extra_env or {})}
    return subprocess.Popen(cmd, env=env, **popen_kwargs)


# ── Console-script entry point ────────────────────────────────────────────────


def main() -> None:
    """
    ``contextpilot-llama-server`` console script.

    Finds ``llama-server`` in PATH (or ``$LLAMA_SERVER_BIN``), compiles and
    injects the native hook when ``CONTEXTPILOT_INDEX_URL`` is set, then
    replaces the current process via ``os.execvpe`` — identical activation
    pattern to SGLang / vLLM.
    """
    import shutil

    llama_bin = os.environ.get("LLAMA_SERVER_BIN") or shutil.which("llama-server")
    if llama_bin is None:
        print(
            "contextpilot-llama-server: llama-server not found in PATH.\n"
            "Install with:  brew install llama.cpp\n"
            "Or build from source and add to PATH, or set LLAMA_SERVER_BIN.",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [llama_bin] + sys.argv[1:]
    index_url = os.environ.get("CONTEXTPILOT_INDEX_URL")

    if not index_url:
        # No ContextPilot integration — exec directly, zero overhead.
        os.execvp(llama_bin, cmd)
        return  # unreachable

    try:
        hook_lib = build()
    except RuntimeError as e:
        print(f"[ContextPilot] Hook compilation failed: {e}", file=sys.stderr)
        print(
            "[ContextPilot] Falling back to plain llama-server (no eviction tracking).",
            file=sys.stderr,
        )
        os.execvp(llama_bin, cmd)
        return  # unreachable

    extra = preload_env(hook_lib=hook_lib, index_url=index_url)
    env = {**os.environ, **extra}

    print(f"[ContextPilot] Hook compiled: {hook_lib}", file=sys.stderr)
    print(
        f"[ContextPilot] Launching with {_PRELOAD_VAR} injected: {llama_bin} ...",
        file=sys.stderr,
    )
    os.execvpe(llama_bin, cmd, env)


# ── CLI entry-point ───────────────────────────────────────────────────────────
#
# Two modes:
#
#   Build-only (print path + instructions):
#       python -m contextpilot._llamacpp_hook [--force]
#
#   Transparent launcher (useful for non-PATH binaries):
#       CONTEXTPILOT_INDEX_URL=http://localhost:8765 \
#       python -m contextpilot._llamacpp_hook \
#           /path/to/llama-server -m model.gguf --port 8889 ...
#
if __name__ == "__main__":
    import argparse

    # Separate our own flags from the passthrough command.
    # Everything after the first non-flag arg (or after --) is the command.
    our_args = []
    cmd_args: "list[str]" = []
    passthrough = False
    for arg in sys.argv[1:]:
        if passthrough:
            cmd_args.append(arg)
        elif arg == "--":
            passthrough = True
        elif arg.startswith("-") and not cmd_args:
            our_args.append(arg)
        else:
            # First positional arg starts the passthrough command
            cmd_args.append(arg)
            passthrough = True

    parser = argparse.ArgumentParser(
        description=(
            "Build the ContextPilot native llama.cpp hook and optionally launch "
            "llama-server with it injected.\n\n"
            "Usage as transparent launcher (mirrors SGLang / vLLM pattern):\n"
            "  CONTEXTPILOT_INDEX_URL=http://localhost:8765 \\\n"
            "  python -m contextpilot._llamacpp_hook \\\n"
            "      /path/to/llama-server -m model.gguf --port 8889 ..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=not cmd_args,  # suppress help when acting as launcher
    )
    parser.add_argument("--force", action="store_true", help="Force recompile")
    parser.add_argument(
        "--index-url",
        default=os.environ.get("CONTEXTPILOT_INDEX_URL", "http://localhost:8765"),
        help="ContextPilot server URL (default: $CONTEXTPILOT_INDEX_URL or http://localhost:8765)",
    )
    args, _ = parser.parse_known_args(our_args)

    try:
        lib = build(force=args.force)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if cmd_args:
        # ── Transparent launcher mode ─────────────────────────────────────────
        print(
            f"[ContextPilot] Hook compiled: {lib}",
            file=sys.stderr,
        )
        print(
            f"[ContextPilot] Launching with {_PRELOAD_VAR} injected: {' '.join(cmd_args)}",
            file=sys.stderr,
        )
        proc = launch(cmd_args, index_url=args.index_url)
        try:
            sys.exit(proc.wait())
        except KeyboardInterrupt:
            proc.terminate()
            sys.exit(proc.wait())
    else:
        # ── Build-only mode ───────────────────────────────────────────────────
        print(f"[ContextPilot] Hook compiled: {lib}")
        print()
        print("Option A — transparent launcher (mirrors SGLang / vLLM pattern):")
        print(f"  CONTEXTPILOT_INDEX_URL={args.index_url} \\")
        print(f"  python -m contextpilot._llamacpp_hook \\")
        print(f"      /path/to/llama-server -m model.gguf --port 8889 ...")
        print()
        print("Option B — manual env injection:")
        print(f"  {_PRELOAD_VAR}={lib} \\")
        print(f"  CONTEXTPILOT_INDEX_URL={args.index_url} \\")
        print(f"  /path/to/llama-server -m model.gguf ...")
        print()
        print("Option C — from Python:")
        print("  from contextpilot._llamacpp_hook import launch")
        print("  proc = launch(['/path/to/llama-server', ...], index_url='...')")
