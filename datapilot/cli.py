from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    # In editable installs, this resolves inside the repo.
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """
    Starts the Streamlit app.

    Usage:
      datapilot
      python -m datapilot
    """
    root = _repo_root()
    app_py = root / "app.py"
    if not app_py.exists():
        raise SystemExit(f"Could not find app.py at: {app_py}")

    # Use module invocation to avoid PATH issues on Windows.
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_py)]
    # Pass through any extra args, e.g. `datapilot --server.port 8502`
    cmd.extend(sys.argv[1:])

    # Ensure the working directory is the repo root so relative paths behave.
    raise SystemExit(subprocess.call(cmd, cwd=str(root)))

