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
- `quant-decoder-build.dockerfile`: quant decoder sandbox build phase (run image plus gcc/g++/make).
- `quant-decoder-run.dockerfile`: quant decoder sandbox run phase (pinned numpy, CPU torch, safetensors, zstd, xz; no compiler).
- `quant-evaluator.dockerfile`: quant-task scorer image on a CUDA 13.0 base (no nvcc: Humming JITs through NVRTC from
  the pip wheels). Pins vLLM 0.31.0 and humming-kernels 0.1.16, sets the in-process engine, FlashInfer-sampler-off
  and Humming cache env, and checks those pins and the scorer imports at build time, plus
  `scripts/check_quant_packed_layout.py` (compressed-tensors is exactly vLLM's pin, and vLLM allocates the dense
  packed layout the structural pre-check expects). Ships gcc, libc6-dev and python3-dev because Triton builds its
  CUDA launcher stub with the host cc inside `LLM(...)`. No kernel pre-warm and no SM target: the image runs
  unchanged on H100 SXM and PCIe.
- `standalone-image-toolkit-trainer.dockerfile`: ai-toolkit image trainer expected in miner repos.
- `standalone-text-trainer.dockerfile`: text trainer image expected in miner repos.
- `trainer-downloader.dockerfile`: model/dataset downloader image, including image-model cache validation and legacy FLUX
  checkpoint layout normalization; see `trainer/containers/README.md` for its cache and deployment contract.
- `validator.dockerfile`: base validator image.
- `validator-diffusion.dockerfile`: image evaluation validator image.
