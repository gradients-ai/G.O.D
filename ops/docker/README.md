# Docker

Dockerfiles and container support files for miner, trainer, validator, evaluation, model prep, uploads, and environment tasks.

## Contents

- `environment_functions/`: rollout/reward function files copied into environment training images.
- `patches/`: patches applied to upstream training/evaluation dependencies.
- `requirements/`: Docker-specific requirements files.
- `cache-cleanup.dockerfile`: cache cleanup container image.
- `hf-uploader.dockerfile`: Hugging Face upload container.
- `intercode_build_fs.sh`: InterCode filesystem build helper.
- `miner.dockerfile`: miner service image.
- `model-prep-env.dockerfile`: environment model prep and baseline image, aligned with the SGLang 0.5.14 CUDA 12.9 evaluation stack.
- `model-prep-text.dockerfile`: text model prep image, pinned to Transformers 5.12.1 and PEFT 0.19.1.
- `pvp-eval.dockerfile`: PvP evaluation image.
- `standalone-image-toolkit-trainer.dockerfile`: ai-toolkit image trainer expected in miner repos.
- `standalone-text-trainer.dockerfile`: text trainer image expected in miner repos.
- `trainer-downloader.dockerfile`: model/dataset downloader image, including image-model cache validation and legacy FLUX
  checkpoint layout normalization; see `trainer/containers/README.md` for its cache and deployment contract.
- `validator.dockerfile`: base validator image.
- `validator-diffusion.dockerfile`: image evaluation validator image.
- `validator-env.dockerfile`: environment evaluation validator image. It shares the SGLang/PEFT model-family smoke checks with the other environment evaluators.
- `validator-intercode.dockerfile`: InterCode evaluation image, including structured tool parsing for the round-one model catalog.
- `validator-swe-infinite.dockerfile`: SWE Infinite model-serving evaluation image. Its SGLang, Pydantic/core, and PEFT
  versions are pinned together, and the build imports both the SGLang launcher and LoRA merge path to catch dependency skew.
