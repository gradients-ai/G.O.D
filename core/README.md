# Core

Shared code used by more than one runtime. Keep this package free of validator-only, trainer-only, or miner-only behavior unless the type or helper is part of a cross-process contract.

## Contents

- `constants/`: shared network, dataset, docker, path, training, credential, and environment constants.
- `datasets/`: approved SFT dataset whitelist and small dataset fixtures.
- `models/`: shared Pydantic/dataclass models used across service boundaries.
- `training_templates/`: base Axolotl and diffusion training templates copied into trainer containers.
- `downloads.py`: shared S3 download helper.
- `git.py`: Git repository URL and clone helpers.
- `logging.py`: common logger setup.
- `training_config.py`: shared Axolotl dataset entry construction used by trainer and validator evaluation flows.
