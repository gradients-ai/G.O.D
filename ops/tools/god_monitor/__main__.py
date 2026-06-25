#!/usr/bin/env python3
"""Launcher for the G.O.D Tournament Monitor.

Run from the G.O.D repo root:
    python -m ops.tools.god_monitor            # interactive menu
    python -m ops.tools.god_monitor summary --all

Adds the repo root so `validator.*`, `core.*`, and `ops.*` imports resolve
regardless of how this is invoked.
"""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    from ops.tools.god_monitor.cli import main

    main()
