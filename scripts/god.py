#!/usr/bin/env python3
"""Launcher for the G.O.D Tournament Monitor.

Run from the G.O.D repo root:
    python scripts/god.py            # interactive menu
    ./scripts/god.py summary --all   # direct command

Adds the repo root (for `validator.*` / `core.*`) and `utils/` (where the
`god_monitor` package lives) to sys.path so imports resolve regardless of how
this is invoked.
"""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (REPO_ROOT, REPO_ROOT / "utils"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


if __name__ == "__main__":
    from god_monitor.cli import main

    main()
