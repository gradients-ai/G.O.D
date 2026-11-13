# Core Constants

Shared constants that are safe for multiple runtimes to import.

## Contents

- `credentials.py`: environment-variable names and credential-related constants, including the tournament HF org
  (`RAYONLABS_HF_USERNAME`, from `TOURNAMENT_HF_USERNAME`, default `gradients-io-tournaments`; the one definition,
  re-exported by `validator/infrastructure/service_constants.py`).
- `datasets.py`: image/text pair limits and dataset split constants.
- `docker.py`: shared Docker image names and container defaults, and `TEXT_DOCKERFILE_PATHS`, the text training Dockerfile the trainer builds and the validator's registration lint reads.
- `environments.py`: supported environment names, environment images, and environment runtime configuration.
- `network.py`: subnet, chain, and network-level constants.
- `paths.py`: shared container/cache path constants.
- `quant.py`: quant-task constants: the decoder limits (directory name, xz size cap, decode-sandbox timeout, CPUs,
  memory multiplier and pids limit, and the `decode.py` entrypoint and 64 MiB extracted-size cap that both the trainer's
  decode and registration enforce), the decoded tensor dtype, the HF config filenames used to tell a full model from a
  LoRA adapter, the base-model size band (`QUANT_MODEL_SIZE_B`), the build-context review size
  (`QUANT_LARGE_FILE_REVIEW_BYTES`, flagged for manual review), the miner output contract (`QUANT_MANIFEST_NAME`,
  `QUANT_MANIFEST_MAX_FILES`, `QUANT_ARTIFACT_MAX_BYTES_MULTIPLIER`).
- `training.py`: training defaults and task-type training constants.
- `__init__.py`: aggregate import surface for core constants.

Add new constants to the narrowest file that owns the domain. Avoid turning `__init__.py` into the only place a constant exists.
