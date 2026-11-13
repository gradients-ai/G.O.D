# Tests

Automated tests, grouped by the runtime or package they cover.

## Contents

| Path | Purpose |
| --- | --- |
| `core/` | Shared models and helpers: compression models, tokenizer utilities, Git URLs, URL redaction. |
| `e2e/` | End-to-end scripts and the freeform exposure test. |
| `ops/` | Ops tools and the `.dockerignore` secret exclusions. |
| `subnet_api/` | Subnet API application. |
| `trainer/` | Trainer service, model prep, cache and decoder sandbox. |
| `validator/` | Validator evaluation, scoring, lifecycle, tasks and tournaments. |
| `conftest.py` | The `--run-gpu` option and the `gpu` marker. |
| `test_gpu_pair_selection.py` | GPU grouping and trainer selection from reported topology. |
| `test_instruct_kl_wiring.py` | Wiring of optional KL-regularised instruct training. |
| `test_node_install.py` | Node install paths stay aligned with the optional extras. |
| `test_organic_package.py` | The organic package imports and runs without chain packages. |

## GPU tests

Tests marked `gpu` need a GPU and model downloads. They are skipped unless `--run-gpu` is passed.

## See also

- [Testing and linting](../docs/developer.md#testing-and-linting): the commands that run and lint the tests.
