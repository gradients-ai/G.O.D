# Trainer Model Prep

Model preparation and baseline-stat routines used before selected training jobs.

## Contents

- `augmentation.py`: model/data augmentation helpers for prep.
- `entrypoint.py`: model-prep container entrypoint.
- `env_stats.py`: environment-task baseline and sidecar stats collection.
- `stats.py`: general model and dataset statistics collection.
- `__init__.py`: package marker.

Environment baselines run until `MODEL_PREP_ENV_TIME_BUDGET_SECONDS` expires, defaulting
to 420 seconds per environment. PvP game baselines run in-harness; individual environments
run through their sidecar.
