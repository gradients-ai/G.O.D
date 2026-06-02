# Tests

Automated and end-to-end tests grouped by runtime/domain.

## Contents

- `e2e/`: shell and Python entrypoints for remote/container PvP evaluation flows.
- `trainer/`: trainer service, model prep, cache, and anonymizer tests.
- `validator/`: validator evaluation, scoring, and tournament tests.
- `__init__.py`: package marker.

Most tests can be run with `uv run --extra dev pytest -q`. Some evaluation tests need `--extra gpu` or optional image/model dependencies.
