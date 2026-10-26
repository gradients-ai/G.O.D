# Docker

Dockerfiles and container support files for miner, trainer, validator, evaluation, model prep, uploads, and environment tasks.

## Contents

- `environment_functions/`: rollout/reward function files copied into environment training images.
- `patches/`: patches applied to upstream training/evaluation dependencies.
- `requirements/`: Docker-specific requirements files.
- `cache-cleanup.dockerfile`: cache cleanup container image.
- `hf-uploader.dockerfile`: Hugging Face upload container.
- `miner.dockerfile`: miner service image.
- `model-prep.dockerfile`: model-prep image.
- `standalone-image-toolkit-trainer.dockerfile`: ai-toolkit image trainer expected in miner repos.
- `standalone-text-trainer.dockerfile`: text trainer image expected in miner repos.
- `trainer-downloader.dockerfile`: model/dataset downloader image, including image-model cache validation and legacy FLUX
  checkpoint layout normalization; see `trainer/containers/README.md` for its cache and deployment contract.
- `validator.dockerfile`: text evaluator image (`gradientsio/text-evaluator`): CUDA 12.8 + torch 2.9.1 + pinned HF stack
  from `requirements/text-evaluator-pins.txt`, no axolotl; a build-time smoke asserts the pinned versions and that every
  text evaluator entrypoint imports.
- `validator-diffusion.dockerfile`: image evaluation validator image.
