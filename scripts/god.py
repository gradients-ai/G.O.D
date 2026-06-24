#!/usr/bin/env python3
"""Launcher for the G.O.D Tournament Monitor.

Run from the G.O.D repo root:
    python scripts/god.py            # interactive menu
    ./scripts/god.py summary --all   # direct command

Adds the repo root to sys.path so `god_monitor`, `validator.*` and `core.*`
imports resolve regardless of how this is invoked.
"""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    from god_monitor.cli import main

    main()
