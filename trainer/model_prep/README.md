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

## LoRA continuation bases

When the task base is a LoRA adapter (e.g. continuous-SFT carrying last week's winner),
model-prep republishes a flat full-weight copy at a deterministic `merged-<hash>` repo
and returns it as `augmented_model_id`. The hash is keyed on the original source HF repo
id (`--source-model-id`), not the anonymized local cache path, so retries reuse one
publish target. Training and eval then prefer that flat base, so the lineage head is never
a raw adapter after prep and chain depth stays one hop per week.
