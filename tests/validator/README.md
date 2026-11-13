# Validator tests

Tests for the validator package, one directory per tested validator subpackage.

## Contents

| Path | Purpose |
| --- | --- |
| `db/` | SQL helpers: continuous task state, placements, compression tasks, the per-SKU reference cache, calibration rows. |
| `evaluation/` | Evaluators, the compression scorer, remote backends and deployment reconciliation. |
| `infrastructure/` | The S3-compatible storage client. |
| `lifecycle/` | Evaluation error handling, compression reference checks at finalisation and deployment cleanup. |
| `nodes/` | Node refresh. |
| `scoring/` | Task and tournament scoring. |
| `tasks/` | Task preparation: dataset splits, the calibration carve, compression eval sets and training hours. |
| `tournament/` | Tournament rounds, boss rounds, the crown and weights. |

## See also

- [Testing and linting](../../docs/developer.md#testing-and-linting): test and lint commands.
