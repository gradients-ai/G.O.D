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
LoRA candidates use SGLang's native LoRA loader over their materialized base,
matching PvP evaluation even when the adapter repository contains tokenizer or
added-token artifacts. This preserves the base tokenizer/EOS serving contract and
avoids constructing a SWE-only merged model with divergent tokenizer metadata.

Each SWE task contributes one term to the final average. A task that still fails
or exceeds the overall session timeout contributes `0.0`. Affinetes TCP connection
setup failures are retried by default up to three total attempts, with exponential
backoff starting at one second. Errors after connection setup are not retried,
because the server may already have started that task and a second submission could
duplicate the evaluation. The retry count and initial delay can be overridden with
`connect_max_attempts` and `connect_retry_backoff_seconds` in
`SWE_INFINITE_EVAL_CONFIG_JSON`.
