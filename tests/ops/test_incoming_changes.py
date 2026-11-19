import datetime
import difflib
import textwrap

import pytest
from pydantic import ValidationError

from ops.tools.incoming_changes.models import INCOMING_CHANGES_PATH
from ops.tools.incoming_changes.models import ChangeStatus
from ops.tools.incoming_changes.models import IncomingChanges
from ops.tools.incoming_changes.models import parse_incoming_changes
from ops.tools.incoming_changes.ship import ShipError
from ops.tools.incoming_changes.ship import change_ids_from_labels
from ops.tools.incoming_changes.ship import merge_date_utc
from ops.tools.incoming_changes.ship import ship
from ops.tools.incoming_changes.validate import validate


SAMPLE = textwrap.dedent(
    """\
    # header comment
    changes:
      - id: first-change
        title: First
        why: Because.
        summary: >-
          A folded summary
          over two lines.
        areas: [text]
        status: planned
        date: 2026-12-07
        miner_action: Nothing.
        prs: [10]
        docs: []
        shipped_on: null

      # comment between entries
      - id: second-change
        title: Second
        summary: Second summary.
        why: Because.
        areas: [image, compression]
        status: testing
        date: 2026-12-14
        miner_action: Nothing.
        prs:
          - 11
        shipped_on: null
    """
)


def _entry(**overrides) -> dict:
    entry = {
        "id": "some-change",
        "title": "Title",
        "summary": "Summary.",
        "why": "Because.",
        "areas": ["text"],
        "status": "planned",
        "date": "2026-12-07",
        "miner_action": "Nothing.",
        "shipped_on": None,
    }
    entry.update(overrides)
    return entry


def _validate_entries(*entries: dict):
    return IncomingChanges.model_validate({"changes": list(entries)})


def test_repo_file_is_valid():
    assert validate(INCOMING_CHANGES_PATH) == []


def test_sample_parses():
    changes = parse_incoming_changes(SAMPLE)
    assert [change.id for change in changes.changes] == ["first-change", "second-change"]
    assert changes.get("second-change").prs == [11]


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "Not_Kebab"},
        {"areas": ["audio"]},
        {"areas": []},
        {"areas": ["text", "text"]},
        {"status": "done"},
        {"date": "2026-12-08"},  # Tuesday
        {"date": "07/12/2026"},
        {"status": "shipped"},  # missing shipped_on
        {"shipped_on": "2026-12-07"},  # shipped_on without shipped status
        {"prs": [0]},
        {"docs": ["/abs/path.md"]},
        {"docs": ["../outside.md"]},
        {"unexpected": "field"},
        {"summary": "  "},
        {"why": "  "},
    ],
)
def test_invalid_entries_rejected(overrides):
    with pytest.raises(ValidationError):
        _validate_entries(_entry(**overrides))


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate ids: some-change"):
        _validate_entries(_entry(), _entry(title="Other"))


def test_shipped_entry_valid():
    changes = _validate_entries(_entry(status="shipped", shipped_on="2026-12-09", prs=[1]))
    assert changes.changes[0].shipped_on == datetime.date(2026, 12, 9)


def test_missing_doc_reported(tmp_path):
    path = tmp_path / "incoming.yaml"
    path.write_text(SAMPLE.replace("docs: []", "docs: [docs/does-not-exist.md]"))
    assert validate(path, repo_root=tmp_path) == ["doc path not found: first-change: docs/does-not-exist.md"]


def test_labels_and_merge_date():
    assert change_ids_from_labels(["bug", "incoming:a-b", "incoming:a-b", "incoming:c"]) == ["a-b", "c"]
    assert merge_date_utc("2026-12-07T23:30:00Z") == datetime.date(2026, 12, 7)
    assert merge_date_utc("2026-12-08T01:30:00+03:00") == datetime.date(2026, 12, 7)


def test_ship_only_touches_target_lines():
    shipped_on = datetime.date(2026, 12, 8)
    updated = ship(SAMPLE, ["first-change", "second-change"], 42, shipped_on)

    changes = parse_incoming_changes(updated)
    for change in changes.changes:
        assert change.status == ChangeStatus.SHIPPED
        assert change.shipped_on == shipped_on
    assert changes.get("first-change").prs == [10, 42]
    assert changes.get("second-change").prs == [11, 42]

    diff = list(difflib.unified_diff(SAMPLE.splitlines(), updated.splitlines(), n=0, lineterm=""))
    removed = [line[1:].strip() for line in diff if line.startswith("-") and not line.startswith("---")]
    added = [line[1:].strip() for line in diff if line.startswith("+") and not line.startswith("+++")]
    assert sorted(removed) == sorted(["status: planned", "prs: [10]", "shipped_on: null", "status: testing", "shipped_on: null"])
    assert sorted(added) == sorted(
        ["status: shipped", "prs: [10, 42]", "shipped_on: 2026-12-08", "status: shipped", "- 42", "shipped_on: 2026-12-08"]
    )


def test_ship_is_idempotent():
    once = ship(SAMPLE, ["first-change"], 42, datetime.date(2026, 12, 8))
    assert ship(once, ["first-change"], 42, datetime.date(2026, 12, 8)) == once


def test_ship_adds_missing_prs_field():
    text = SAMPLE.replace("    prs: [10]\n", "")
    updated = ship(text, ["first-change"], 7, datetime.date(2026, 12, 8))
    assert parse_incoming_changes(updated).get("first-change").prs == [7]


def test_ship_unknown_id_fails():
    with pytest.raises(ShipError, match="unknown incoming change ids: nope"):
        ship(SAMPLE, ["nope"], 1, datetime.date(2026, 12, 8))
