# Core

Shared code used by more than one runtime. Keep this package free of validator-only, trainer-only, or miner-only behavior unless the type or helper is part of a cross-process contract.

## Contents

- `constants/`: shared network, dataset, docker, path, training, credential, and environment constants.
- `datasets/`: approved SFT dataset whitelist and small dataset fixtures.
- `models/`: shared Pydantic/dataclass models used across service boundaries.
- `training_templates/`: base Axolotl and diffusion training templates copied into trainer containers.
- `build_context.py`: `find_large_files`, the scan for files over the quant review size (`QUANT_LARGE_FILE_REVIEW_BYTES`) in a training repo, used by the validator's registration lint and the trainer's per-job image build. It lists, never rejects.
- `downloads.py`: shared S3 download helper.
- `git.py`: Git repository URL and clone helpers.
- `logging.py`: common logger setup, and `redact_url_queries` (replaces every http(s) URL query with `***`, so a presigned URL is never logged with its signature).
- `training_config.py`: shared Axolotl dataset entry construction used by trainer and validator evaluation flows.
