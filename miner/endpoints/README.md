# Miner Endpoints

Miner API routes.

## Contents

- `training_repo.py`: implements `GET /training_repo/{task_type}` and returns `TrainingRepoResponse`.
- `__init__.py`: package marker.

This is the miner's main public contract with validators. Keep response fields compatible with `core.models.payload_models.TrainingRepoResponse`.
