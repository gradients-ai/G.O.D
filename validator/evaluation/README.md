# Validator Evaluation

Evaluation runtimes and helpers used after miner training completes.

## Contents

- `comfy_workflows/`: ComfyUI workflow JSON files for image evaluation.
- `evaluators/`: task-specific evaluator entrypoints for text, DPO, GRPO, and image tasks.
- `basilica.py`: Basilica client integration.
- `basilica_deployments.py`: Basilica deployment helpers.
- `common.py`: shared text evaluation helpers (base param counts via hub metadata, no weight reload).
- `constants.py`: evaluation constants (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on all remote eval containers).
- `remote_runner.py`: HTTP wrapper for Basilica/Runpod eval containers; serves `/result` even if a post-eval step exits non-zero when results already exist.
- `runpod.py`: Runpod evaluation backend via dstack services.
- `dataset_configs.py`: default dataset config discovery.
- `db_utils.py`: evaluation persistence helpers.
- `docker_evaluation.py`: Docker/Basilica evaluation orchestration.
- `evaluation_logging.py`: evaluation logging helpers.
- `image_io.py`: image loading and encoding helpers.
- `local_evaluation.py`: local evaluation runner.
- `model_checks.py`: model architecture and LoRA checks.
- `models.py`: evaluation payload models.
- `reconcile.py`: evaluation deployment and GPU-reservation reconciliation, including release of leftover PvP reservations from historical environment tasks.
- `result_processing.py`: conversion of raw eval output into persisted results.
- `runtime.py`: evaluation runtime helpers.
- `utils.py`: evaluation utility helpers.

Environment, InterCode, SWE Infinite, and PvP evaluation runtimes have been removed. `TaskType.ENVIRONMENTTASK` and stored PvP score rows remain readable for historical tournaments.
