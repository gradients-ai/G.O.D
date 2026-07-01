# Validator Evaluation

Evaluation runtimes and helpers used after miner training completes.

## Contents

- `comfy_workflows/`: ComfyUI workflow JSON files for image evaluation.
- `evaluators/`: task-specific evaluator entrypoints.
- `pvp/`: PvP environment evaluation runtime.
- `basilica.py`: Basilica client integration.
- `basilica_deployments.py`: Basilica deployment helpers.
- `common.py`: shared text evaluation helpers.
- `constants.py`: evaluation constants.
- `dataset_configs.py`: default dataset config discovery.
- `db_utils.py`: evaluation persistence helpers.
- `docker_evaluation.py`: Docker/Basilica evaluation orchestration.
- `evaluation_logging.py`: evaluation logging helpers.
- `image_io.py`: image loading and encoding helpers.
- `intercode_server.py`: InterCode server wrapper.
- `local_evaluation.py`: local evaluation runner.
- `model_checks.py`: model architecture and LoRA checks.
- `models.py`: evaluation payload models.
- `result_processing.py`: conversion of raw eval output into persisted results.
- `runtime.py`: evaluation runtime helpers.
- `utils.py`: evaluation utility helpers.

SWE Infinite evaluation runs as an individual environment tournament eval. The
candidate model is served by SGLang inside Basilica, and the evaluator calls an
external Affinetes SWE Infinite server configured by `SWE_INFINITE_SERVER_BASE_URL`.
