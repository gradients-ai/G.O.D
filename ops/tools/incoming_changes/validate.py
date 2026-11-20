"""Validate docs/incoming_changes.yaml.

Usage: python -m ops.tools.incoming_changes.validate [path]
"""

import argparse
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from ops.tools.incoming_changes.models import INCOMING_CHANGES_PATH
from ops.tools.incoming_changes.models import REPO_ROOT
from ops.tools.incoming_changes.models import IncomingChanges
from ops.tools.incoming_changes.models import parse_incoming_changes


def missing_docs(changes: IncomingChanges, repo_root: Path = REPO_ROOT) -> list[str]:
    return [f"{change.id}: {path}" for change in changes.changes for path in change.docs if not (repo_root / path).exists()]


def validate(path: Path, repo_root: Path = REPO_ROOT) -> list[str]:
    """Return a list of problems; empty means the file is valid."""
    try:
        changes = parse_incoming_changes(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot read {path}: {exc}"]
    except ValidationError as exc:
        return [f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()]
    return [f"doc path not found: {problem}" for problem in missing_docs(changes, repo_root)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", type=Path, default=INCOMING_CHANGES_PATH)
    args = parser.parse_args()

    problems = validate(args.path)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"{args.path}: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
