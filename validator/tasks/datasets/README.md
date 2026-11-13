# Task datasets

Dataset validation, column mapping, splitting and upload for validator tasks.

## Contents

| Path | Purpose |
| --- | --- |
| `calibration.py` | The calibration carve: pure functions over row hashes and character counts, shared by the split code and the tests. |
| `columns.py` | Column inspection of a dataset per task type. |
| `constants.py` | Dataset preparation and training-hours constants. |
| `hub_safety.py` | Admits only Hugging Face datasets without loading scripts, pinned to the checked commit. |
| `mapping.py` | Maps source columns to the standard task columns and merges datasets. |
| `models.py` | Dataset payload models. |
| `preparation.py` | Download, train/test split, duplicate and near-duplicate checks, and upload. |

## Calibration invariants

| Invariant | Where |
| --- | --- |
| The test split is drawn first and does not depend on the carve. | `preparation.py` |
| Calibration rows come only from the train side and never equal a test row. | `calibration.py` |
| For a task that splits its dataset, carved row hashes are recorded per dataset, and later splits keep those rows out of the test pool. | `validator/db/sql/calibration.py` |
| A task with `calibration_data` NULL publishes no calibration set. | task row |

## See also

- [The calibration set](../../../docs/miner_compression.md#the-calibration-set): the miner-facing contract.
