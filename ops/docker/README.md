# Docker

Dockerfiles and build support files for the trainer, model prep, decoder sandbox and evaluator images.

## Contents

| Path | Purpose |
| --- | --- |
| `patches/` | `comfyui-fp8-lora.patch`, applied to ComfyUI in the image evaluator build. |
| `requirements/` | Requirements files used only inside Docker builds. |
| `scripts/` | Build checks and entrypoints: `check_quant_packed_layout.py`, `image_eval_entrypoint.sh`, `image_toolkit_entrypoint.py`. |
| `cache-cleanup.dockerfile` | Trainer cache cleanup image. |
| `hf-uploader.dockerfile` | Hugging Face upload image run after training. |
| `miner.dockerfile` | Axolotl training image with config, output and input data directories. |
| `model-prep-text.dockerfile` | Model prep image for text tasks. |
| `quant-decoder-build.dockerfile` | Decoder sandbox build phase: the run image plus a C/C++ toolchain. |
| `quant-decoder-run.dockerfile` | Decoder sandbox run phase: pinned numpy, CPU torch, safetensors, zstd and xz, with no compiler. |
| `quant-evaluator.dockerfile` | Compression task scorer image. The build checks its pinned versions and the packed tensor layout. |
| `standalone-image-toolkit-trainer.dockerfile` | Reference ai-toolkit image trainer. |
| `trainer-downloader.dockerfile` | Model and dataset downloader image. |
| `validator.dockerfile` | Text evaluator image (`gradientsio/text-evaluator`), also the devcontainer base. |
| `validator-diffusion.dockerfile` | Image evaluator image. |

## See also

- [Trainer containers](../../trainer/containers/README.md): the downloader, uploader and cache cleanup containers.
- [Decoder sandbox](../../trainer/quant_decode/README.md): the decoder sandbox.
- [Training repository](../../docs/miner.md#training-repository): the Dockerfiles a training repository provides.
