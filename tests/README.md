# Tests

Automated and end-to-end tests grouped by runtime/domain.

## Contents

- `e2e/`: shell and Python entrypoints for remote/container PvP evaluation flows.
- `trainer/`: trainer service, model prep, cache, and anonymizer tests.
- `validator/`: validator evaluation, scoring, and tournament tests.
- `__init__.py`: package marker.

Most tests can be run locally with:

```bash
uv run --extra dev pytest -q
```

GPU/model-download tests are marked `gpu` and skipped by default so local CPU-only development does not need Torch, Transformers, CUDA, or model downloads. Run them explicitly on a suitable machine with:

```bash
uv run --extra dev --extra gpu pytest -q --run-gpu
```
