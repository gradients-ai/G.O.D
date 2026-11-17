"""Mark docs/incoming_changes.yaml entries shipped for a merged PR.

Usage:
    python -m ops.tools.incoming_changes.ship --pr 1432 --merged-at 2026-12-07T10:15:00Z \
        --label incoming:threshold-recalibration --label documentation

Labels without the `incoming:` prefix are ignored. For each `incoming:<id>` label the entry's
`status` becomes `shipped`, `shipped_on` the merge date (UTC) and the PR number is appended to `prs`.

The file is edited line by line rather than re-dumped through a YAML library, so comments, folded
summaries, key order and spacing stay exactly as written; only the `status`, `shipped_on` and `prs`
lines of the touched entries change. The result is re-parsed and validated before it is written.
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

import yaml

from ops.tools.incoming_changes.models import INCOMING_CHANGES_PATH
from ops.tools.incoming_changes.models import LABEL_PREFIX
from ops.tools.incoming_changes.models import ChangeStatus
from ops.tools.incoming_changes.models import parse_incoming_changes


class ShipError(Exception):
    pass


def change_ids_from_labels(labels: list[str]) -> list[str]:
    ids = [label[len(LABEL_PREFIX) :].strip() for label in labels if label.startswith(LABEL_PREFIX)]
    return list(dict.fromkeys(change_id for change_id in ids if change_id))


def merge_date_utc(merged_at: str) -> datetime.date:
    parsed = datetime.datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc).date()


def _entry_bounds(lines: list[str], change_id: str) -> tuple[int, int, str]:
    """Return (start, end, field_indent) of the list item whose id is change_id."""
    start_pattern = re.compile(rf"^(\s*)- id:\s*['\"]?{re.escape(change_id)}['\"]?\s*(#.*)?$")
    for start, line in enumerate(lines):
        match = start_pattern.match(line.rstrip("\n"))
        if not match:
            continue
        item_indent = len(match.group(1))
        end = start + 1
        while end < len(lines):
            stripped = lines[end].strip()
            indent = len(lines[end]) - len(lines[end].lstrip(" "))
            if stripped and not stripped.startswith("#") and indent <= item_indent:
                break
            end += 1
        # Leave trailing blank/comment lines outside the entry.
        while end > start + 1 and (not lines[end - 1].strip() or lines[end - 1].strip().startswith("#")):
            end -= 1
        return start, end, " " * (item_indent + 2)
    raise ShipError(f"no entry with id {change_id!r}")


def _find_field(lines: list[str], start: int, end: int, indent: str, key: str) -> int | None:
    pattern = re.compile(rf"^{indent}{key}:(\s|$)")
    return next((index for index in range(start, end) if pattern.match(lines[index])), None)


def _set_scalar(lines: list[str], start: int, end: int, indent: str, key: str, value: str) -> int:
    """Set `key: value` in the entry, inserting it at the end if missing. Returns how many lines were inserted."""
    index = _find_field(lines, start, end, indent, key)
    if index is None:
        lines.insert(end, f"{indent}{key}: {value}\n")
        return 1
    lines[index] = f"{indent}{key}: {value}\n"
    return 0


def _add_pr(lines: list[str], start: int, end: int, indent: str, pr: int) -> int:
    """Append pr to the entry's prs list (flow or block style). Returns how many lines were inserted."""
    index = _find_field(lines, start, end, indent, "prs")
    if index is None:
        shipped_on = _find_field(lines, start, end, indent, "shipped_on")
        lines.insert(shipped_on if shipped_on is not None else end, f"{indent}prs: [{pr}]\n")
        return 1

    value = lines[index].split(":", 1)[1].split("#", 1)[0].strip()
    if value:
        current = yaml.safe_load(value) or []
        if not isinstance(current, list):
            raise ShipError(f"prs is not a list: {value!r}")
        if pr not in current:
            current.append(pr)
        lines[index] = f"{indent}prs: [{', '.join(str(item) for item in current)}]\n"
        return 0

    # Block style: "prs:" followed by "- N" items.
    item_pattern = re.compile(rf"^({indent}\s*)-\s*(\d+)\s*(#.*)?$")
    last_item, item_prefix, existing = index, f"{indent}  ", []
    for cursor in range(index + 1, end):
        match = item_pattern.match(lines[cursor].rstrip("\n"))
        if not match:
            break
        last_item, item_prefix = cursor, match.group(1)
        existing.append(int(match.group(2)))
    if pr in existing:
        return 0
    lines.insert(last_item + 1, f"{item_prefix}- {pr}\n")
    return 1


def ship_entry(text: str, change_id: str, pr: int, shipped_on: datetime.date) -> str:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    start, end, indent = _entry_bounds(lines, change_id)
    end += _set_scalar(lines, start, end, indent, "status", ChangeStatus.SHIPPED.value)
    end += _add_pr(lines, start, end, indent, pr)
    _set_scalar(lines, start, end, indent, "shipped_on", shipped_on.isoformat())
    return "".join(lines)


def ship(text: str, change_ids: list[str], pr: int, shipped_on: datetime.date) -> str:
    known = parse_incoming_changes(text)
    unknown = [change_id for change_id in change_ids if known.get(change_id) is None]
    if unknown:
        raise ShipError(f"unknown incoming change ids: {', '.join(unknown)}")

    for change_id in change_ids:
        text = ship_entry(text, change_id, pr, shipped_on)

    updated = parse_incoming_changes(text)
    for change_id in change_ids:
        change = updated.get(change_id)
        if change is None or change.status != ChangeStatus.SHIPPED or change.shipped_on != shipped_on or pr not in change.prs:
            raise ShipError(f"failed to mark {change_id!r} shipped")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pr", type=int, required=True, help="merged PR number")
    parser.add_argument("--merged-at", required=True, help="PR merged_at timestamp (ISO 8601)")
    parser.add_argument("--label", action="append", default=[], help="PR label; repeat for each label")
    parser.add_argument("--path", type=Path, default=INCOMING_CHANGES_PATH)
    args = parser.parse_args()

    change_ids = change_ids_from_labels(args.label)
    if not change_ids:
        print("no incoming:<id> labels; nothing to do")
        return 0

    original = args.path.read_text()
    try:
        updated = ship(original, change_ids, args.pr, merge_date_utc(args.merged_at))
    except ShipError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if updated == original:
        print("already up to date")
        return 0
    args.path.write_text(updated)
    print(f"marked shipped by #{args.pr}: {', '.join(change_ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
