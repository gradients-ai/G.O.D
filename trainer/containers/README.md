# Trainer containers

Code that runs inside the support containers the trainer launches around a training job.

## Contents

| Path | Purpose |
| --- | --- |
| `cache_cleanup.py` | Removes checkpoints, datasets and models of old jobs from the shared volumes. |
| `dataset_cache.py` | Downloads approved miner-requested Hugging Face datasets into the cache volume. |
| `downloader.py` | Downloads a job's model and dataset before training. |
| `uploader.py` | Uploads a finished job's output to Hugging Face. |

## Verbatim uploads

With `UPLOAD_VERBATIM=1`, set by the runtime for compression tasks, the uploader uploads the output root as it is. It skips checkpoint-subfolder detection, `README.md` and `adapter_config.json` patching, tokenizer-config sanitising, and the training-state ignore patterns. The uploaded bytes then match `quant_manifest.json`.

## Image model cache layouts

The downloader keeps two layouts for image models.

| Layout | Applies to | Cache content |
| --- | --- | --- |
| Standalone checkpoint | a FLUX repo with root `.safetensors` files, no `model_index.json`, no Diffusers component directories and no sharded-weight index | one checkpoint as the only top-level file |
| Full snapshot | Diffusers repos, and the Z-Image, Qwen-Image, Ideogram4 and Krea2 model types | the complete repository, with component files in place |

- When a FLUX repo has several root checkpoints, the downloader keeps one full-precision file: a bf16 or fp16 name first, else an unmarked file.
- For every FLUX task it also places `t5xxl_fp16.safetensors`, `clip_l.safetensors` and `ae.safetensors` in `/cache/hf_cache`. Training containers receive their paths as `FLUX_T5_PATH`, `FLUX_CLIP_PATH` and `FLUX_VAE_PATH`.
- Before reusing a cache, the downloader compares it with the repository metadata and checks sharded-index references.
- An incomplete cache is replaced from a validated temporary sibling directory. A download is promoted only after validation.

This behaviour lives in the downloader image. A change takes effect once `ops/docker/trainer-downloader.dockerfile` is rebuilt and the trainer hosts pull the image.
