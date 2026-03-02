#!/usr/bin/env python3
"""
Install ContextPilot engine hooks into the current Python environment.

This is a STANDALONE script — it does NOT require the contextpilot package
to be installed. Use it on inference servers where SGLang/vLLM runs in a
separate environment from ContextPilot.

Usage:
    # On the SGLang/vLLM machine (no contextpilot install needed):
    python install_engine_hook.py          # install
    python install_engine_hook.py --remove # uninstall

What it does:
    1. Copies _sglang_hook.py (and _vllm_hook.py if present) into site-packages
       as standalone modules (contextpilot_sglang_hook.py / contextpilot_vllm_hook.py)
    2. Creates a .pth file that auto-imports them on Python startup
    3. When CONTEXTPILOT_INDEX_URL is set, the hooks monkey-patch SGLang/vLLM
       at import time — zero engine source code changes

Requirements: only `requests` (pip install requests)
"""

import os
import shutil
import site
import sys
from pathlib import Path

PTH_NAME = "contextpilot_hook.pth"

# Hooks to install: (source_filename, installed_module_name)
HOOKS = [
    ("_sglang_hook.py", "contextpilot_sglang_hook"),
    ("_vllm_hook.py", "contextpilot_vllm_hook"),
]


def get_site_packages() -> Path:
    paths = site.getsitepackages()
    return Path(paths[0])


def find_hook_source(filename: str) -> Path | None:
    """Find the hook source file relative to this script or in common locations."""
    candidates = [
        Path(__file__).parent / filename,
        Path(__file__).parent / "contextpilot" / filename,
        Path(__file__).parent.parent / "contextpilot" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def install():
    site_pkg = get_site_packages()
    pth_lines = []
    installed = []

    for src_name, mod_name in HOOKS:
        src = find_hook_source(src_name)
        if src is None:
            print(f"  [skip] {src_name} not found")
            continue

        dest = site_pkg / f"{mod_name}.py"

        # Read source and patch the module to be standalone
        # Replace "contextpilot._sglang_hook" references with the standalone name
        content = src.read_text()
        # The hook file references itself in logger name — update it
        content = content.replace(
            f'"contextpilot.{src_name.replace(".py", "").lstrip("_")}"',
            f'"{mod_name}"',
        )

        dest.write_text(content)
        pth_lines.append(f"import {mod_name}")
        installed.append(dest)
        print(f"  [ok] {dest}")

    if not pth_lines:
        print("No hooks found to install.")
        return

    pth_dest = site_pkg / PTH_NAME
    pth_dest.write_text("\n".join(pth_lines) + "\n")
    installed.append(pth_dest)
    print(f"  [ok] {pth_dest}")

    print(f"\nInstalled {len(installed)} files. To activate:")
    print(f"  CONTEXTPILOT_INDEX_URL=http://<contextpilot-host>:8765 sglang serve ...")


def remove():
    site_pkg = get_site_packages()
    removed = []

    for _, mod_name in HOOKS:
        dest = site_pkg / f"{mod_name}.py"
        if dest.exists():
            dest.unlink()
            removed.append(dest)
            print(f"  [removed] {dest}")

    pth_dest = site_pkg / PTH_NAME
    if pth_dest.exists():
        pth_dest.unlink()
        removed.append(pth_dest)
        print(f"  [removed] {pth_dest}")

    if not removed:
        print("Nothing to remove.")
    else:
        print(f"\nRemoved {len(removed)} files.")


if __name__ == "__main__":
    if "--remove" in sys.argv:
        remove()
    else:
        install()
