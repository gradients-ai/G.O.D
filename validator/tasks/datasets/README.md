# Task Datasets

Dataset validation, mapping, and preparation for validator tasks.

## Contents

- `columns.py`: dataset column inspection and readability checks.
- `constants.py`: dataset preparation constants.
- `hub_safety.py`: rejects Hugging Face datasets with Python loading scripts and pins the checked commit; every Hub load goes through it with `trust_remote_code=False`.
- `mapping.py`: maps source dataset columns into standard task columns.
- `models.py`: dataset payload models.
- `preparation.py`: dataset download, splitting, standardization, and upload.
- `__init__.py`: package marker.
