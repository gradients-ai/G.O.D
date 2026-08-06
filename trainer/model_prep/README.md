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

Both prep images use Transformers v5 and PEFT 0.19.1. Environment prep uses the SGLang
0.5.14 CUDA 12.9 image and launches through `core.pvp.sglang_server`, which adds the OLMo tool-call parser.
The exact Gemma 4 E2B and Ministral 3 Base tournament checkpoints have no upstream chat
template, so prep supplies the checked-in tool-capable templates under `core/pvp/chat_templates/`.
