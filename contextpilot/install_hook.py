"""
Install the ContextPilot .pth hook into the current Python environment's site-packages.

Usage:
    python -m contextpilot.install_hook          # install
    python -m contextpilot.install_hook --remove  # uninstall

This is only needed for editable installs (pip install -e .).
Regular pip install automatically installs the .pth file.
"""

import site
import sys
from pathlib import Path

PTH_NAME = "contextpilot_hook.pth"
PTH_CONTENT = "import contextpilot._sglang_hook\nimport contextpilot._vllm_hook\n"


def get_site_packages() -> Path:
    paths = site.getsitepackages()
    return Path(paths[0])


def install():
    dest = get_site_packages() / PTH_NAME
    dest.write_text(PTH_CONTENT)
    print(f"Installed {dest}")


def remove():
    dest = get_site_packages() / PTH_NAME
    if dest.exists():
        dest.unlink()
        print(f"Removed {dest}")
    else:
        print(f"Not found: {dest}")


if __name__ == "__main__":
    if "--remove" in sys.argv:
        remove()
    else:
        install()
