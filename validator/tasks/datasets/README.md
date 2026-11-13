# Task Datasets

Dataset validation, mapping, and preparation for validator tasks.

## Contents

- `calibration.py`: the calibration carve (pure functions over raw-row hashes and character counts, shared by the split code and the tests). The miner-facing contract is "The calibration set" in `docs/miner_quant.md`. Validator-side invariants: the test split is drawn exactly as without a carve (`train_test_split`, seed 42, `min(10 %, 1,000)` rows), so scored rows, `reference_loss` and rankings never depend on it; calibration rows come only from the train side after the split, skip any row whose raw hash equals a test row's, and never reach the input task's trainer; no evaluator request ever names `calibration_data`. Carved row hashes are recorded per dataset (`dataset_calibration_rows`, `validator/db/sql/calibration.py`), and every later raw split of that dataset moves recorded rows out of the test pool into train, so a calibration row is never tested later. A task with `calibration_data` NULL has no carve.
- `columns.py`: dataset column inspection and readability checks.
- `constants.py`: dataset preparation constants.
- `hub_safety.py`: rejects Hugging Face datasets with Python loading scripts and pins the checked commit; every Hub load goes through it with `trust_remote_code=False`.
- `mapping.py`: maps source dataset columns into standard task columns.
- `models.py`: dataset payload models.
- `preparation.py`: dataset download, splitting, standardization, and upload.
- `__init__.py`: package marker.
