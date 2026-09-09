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
- `model-prep.dockerfile`: model-prep image.
- `pvp-eval.dockerfile`: PvP evaluation image.
- `standalone-image-toolkit-trainer.dockerfile`: ai-toolkit image trainer expected in miner repos.
- `standalone-text-trainer.dockerfile`: text trainer image expected in miner repos.
- `trainer-downloader.dockerfile`: model/dataset downloader image, including image-model cache validation and legacy FLUX
  checkpoint layout normalization; see `trainer/containers/README.md` for its cache and deployment contract.
- `validator.dockerfile`: base validator image.
- `validator-diffusion.dockerfile`: direct flow-prediction L2 image evaluator, pinned to the validated Comfy revision; no Comfy server or tooling nodes.
- `validator-env.dockerfile`: environment evaluation validator image.
- `validator-intercode.dockerfile`: InterCode evaluation image.
- `validator-swe-infinite.dockerfile`: SWE Infinite model-serving evaluation image. Its SGLang, Pydantic/core, and PEFT
  versions are pinned together, and the build imports both the SGLang launcher and LoRA merge path to catch dependency skew.

## Production image evaluation

Build from the repository root:

```bash
docker build -f ops/docker/validator-diffusion.dockerfile \
  -t gradientsio/image-evaluator:latent-l2-v1 .
```

Create an environment file with `MODEL_TYPE` (`z-image`, `flux`, `qwen-image`,
`krea2` or `ideogram4`), `ORIGINAL_MODEL_REPO`, comma-separated submission repos in
`MODELS`, `DATASET=/inputs/test.zip`, and `TRAINING_DATASET=/inputs/train.zip`.
Directories also work. Remote archives use `TEST_SPLIT_URL` and `TRAIN_SPLIT_URL`
instead. Standard Hugging Face cache/auth variables apply.

```bash
mkdir -p /path/to/results
docker run --rm --gpus all \
  --env-file /path/to/image-eval.env \
  -v /path/to/datasets:/inputs:ro \
  -v /path/to/results:/aplp \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  gradientsio/image-evaluator:latent-l2-v1
```

Results are written to `/aplp/evaluation_results.json`; override with
`EVALUATION_RESULTS_PATH`. `/app/start.sh` invokes the evaluator directly.
The score equally averages captioned and empty-caption flow-prediction MSE over
16 strata × 16 noises per held-out test image per mode. Training images are used
for duplicate screening. SDXL is excluded.

`IMAGE_EVAL_AUDIT_DIR` optionally saves per-case results and artifact provenance.
`IMAGE_EVAL_STRATA`, `IMAGE_EVAL_NOISES`, `IMAGE_EVAL_MAX_IMAGES` (0 means all), and
`IMAGE_EVAL_NOISE_BATCH_SIZE` (default 2) allow manual overrides. They change the
case fingerprint; production orchestration uses the full defaults. Smaller runs
must not replace production task scores. Deploy this image and the validator
together: results use the `image-denoising-l2-v1` contract defined in
`core/models/image_models.py`.
