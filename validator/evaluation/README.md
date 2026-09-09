# Validator Evaluation

Evaluation runtimes and helpers used after miner training completes.

## Contents

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
- `image_flow_adapter.py`: family-specific encoders, deterministic VAEs, strict LoRA loading and model contracts.
- `image_denoising.py`: direct sampler for shared prediction cases.
- `image_test_data.py`: nested dataset readers, deterministic preprocessing and duplicate screening.
- `denoising_mse.py`: shared rectified-flow cases and prediction MSE.
- `image_artifacts.py`: pinned base-model and LoRA artifact loading.
- `image_encoder.py`: deterministic float32 VAE encoding.
- `intercode_server.py`: InterCode server wrapper.
- `local_evaluation.py`: local evaluation runner.
- `model_checks.py`: model architecture and LoRA checks.
- `models.py`: evaluation payload models.
- `result_processing.py`: conversion of raw eval output into persisted results.
- `runtime.py`: evaluation runtime helpers.
- `utils.py`: evaluation utility helpers.

Production image evaluation uses **flow-prediction L2**, averaging captioned and
empty-caption losses equally across held-out images and shared 16 × 16 noise cases.
Z-Image, FLUX, Qwen-Image, Krea-2 Raw and Ideogram-4 are supported; SDXL is excluded.
`evaluators/diffusion.py` loads Comfy libraries directly, with no HTTP server,
image reconstruction or pixel-scoring path. Training data is required only for
duplicate screening. Every candidate uses the same deterministic VAE, base model
and noise cases; full LoRA loading is checked. Results include the version
`image-denoising-l2-v1`, component losses and a case fingerprint. The validator
checks the scalar equals the 50/50 mean and rejects incompatible comparisons.

See `ops/docker/README.md` for the Docker build and production environment contract.

SWE Infinite evaluation runs as an individual environment tournament eval. The
candidate model is served by SGLang inside Basilica, and the evaluator calls an
external Affinetes SWE Infinite server configured by `SWE_INFINITE_SERVER_BASE_URL`.
LoRA candidates use SGLang's native LoRA loader over their materialized base,
matching PvP evaluation even when the adapter repository contains tokenizer or
added-token artifacts. This preserves the base tokenizer/EOS serving contract and
avoids constructing a SWE-only merged model with divergent tokenizer metadata.

Environment, InterCode, SWE Infinite, and PvP evaluation do not opt into
Hugging Face remote model code. Transformer merge loads explicitly disable
`trust_remote_code`, downloaded snapshots exclude Python modules, and SGLang
commands that contain `--trust-remote-code` are rejected. Models that require
custom repository code therefore fail closed during evaluation.

Each SWE task contributes one term to the final average. A task that still fails
or exceeds the overall session timeout contributes `0.0`. Affinetes TCP connection
setup failures are retried by default up to three total attempts, with exponential
backoff starting at one second. Errors after connection setup are not retried,
because the server may already have started that task and a second submission could
duplicate the evaluation. The retry count and initial delay can be overridden with
`connect_max_attempts` and `connect_retry_backoff_seconds` in
`SWE_INFINITE_EVAL_CONFIG_JSON`.
