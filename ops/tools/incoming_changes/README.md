# Incoming Changes Tools

Tools for `docs/incoming_changes.yaml`, the miner-facing list of upcoming changes.

## Contents

- `models.py`: Pydantic schema for the file (ids, areas, statuses, dates, shipped rules).
- `validate.py`: `python -m ops.tools.incoming_changes.validate [path]`; exits non-zero on schema errors, duplicate ids or missing doc paths.
- `ship.py`: `python -m ops.tools.incoming_changes.ship --pr N --merged-at TS --label incoming:<id>`; marks entries shipped. Used by `.github/workflows/incoming-changes-ship.yml`. Edits only the `status`, `shipped_on` and `prs` lines so comments and formatting are kept.
- `__init__.py`: package marker.

See "Incoming Changes For Miners" in `docs/developer.md` for the workflow.
