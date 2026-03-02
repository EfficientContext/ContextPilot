"""
One-command installer for the standalone ContextPilot hook.

Downloads standalone_hook.py from GitHub, installs it into site-packages,
and creates a .pth file so the hook auto-activates on Python startup.

Usage (no clone required):
    curl -sL https://raw.githubusercontent.com/EfficientContext/ContextPilot/main/contextpilot/install_standalone.py | python -
    curl -sL https://raw.githubusercontent.com/EfficientContext/ContextPilot/main/contextpilot/install_standalone.py | python - --remove

Or if contextpilot is installed locally:
    python -m contextpilot.install_standalone
    python -m contextpilot.install_standalone --remove
"""

import site
import sys
import urllib.request
from pathlib import Path

HOOK_URL = (
    "https://raw.githubusercontent.com/EfficientContext/ContextPilot"
    "/main/contextpilot/standalone_hook.py"
)
DEST_FILENAME = "contextpilot_hook.py"
PTH_FILENAME = "contextpilot_hook.pth"
PTH_CONTENT = "import contextpilot_hook\n"


def get_site_packages() -> Path:
    paths = site.getsitepackages()
    return Path(paths[0])


def install():
    site_dir = get_site_packages()
    dest_hook = site_dir / DEST_FILENAME
    dest_pth = site_dir / PTH_FILENAME

    # Try local copy first (when run inside the repo / installed package)
    local = Path(__file__).resolve().parent / "standalone_hook.py"
    if local.exists():
        dest_hook.write_text(local.read_text())
        print(f"Installed {dest_hook}  (from local)")
    else:
        print(f"Downloading {HOOK_URL} …")
        try:
            resp = urllib.request.urlopen(HOOK_URL)
            dest_hook.write_bytes(resp.read())
            print(f"Installed {dest_hook}  (from GitHub)")
        except Exception as e:
            print(f"Error: failed to download hook: {e}", file=sys.stderr)
            sys.exit(1)

    dest_pth.write_text(PTH_CONTENT)
    print(f"Installed {dest_pth}")
    print()
    print("Done! Set CONTEXTPILOT_INDEX_URL when launching your engine:")
    print("  CONTEXTPILOT_INDEX_URL=http://host:8765 python -m sglang.launch_server ...")


def remove():
    site_dir = get_site_packages()
    for name in (DEST_FILENAME, PTH_FILENAME):
        p = site_dir / name
        if p.exists():
            p.unlink()
            print(f"Removed {p}")
        else:
            print(f"Not found: {p}")


if __name__ == "__main__":
    if "--remove" in sys.argv:
        remove()
    else:
        install()
