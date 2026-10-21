# Trainer Containers

Helpers for support containers launched by the trainer runtime.

## Contents

- `cache_cleanup.py`: removes stale cache entries and old trainer artifacts.
- `dataset_cache.py`: downloads approved miner-requested Hugging Face datasets into the shared cache volume.
- `downloader.py`: pre-downloads models and task datasets before training.
- `uploader.py`: uploads completed model artifacts to Hugging Face.
- `__init__.py`: package marker.

## Image model cache layouts

The downloader preserves two image-model layouts. A FLUX repository is treated as a standalone checkpoint only when its
repository tree has exactly one root `.safetensors` file, no root `model_index.json`, no Diffusers component directories,
and no sharded-weight index. Its final cache directory has that checkpoint as its only top-level file. This contract keeps
pinned miner repositories compatible with their legacy resolver, which selects a checkpoint only when it is the sole
top-level file.

For every FLUX task, the downloader also places the exact auxiliary artifacts used by evaluation in `/cache/hf_cache`:
`t5xxl_fp16.safetensors`, `clip_l.safetensors`, and `ae.safetensors`. Trainer containers receive their paths as
`FLUX_T5_PATH`, `FLUX_CLIP_PATH`, and `FLUX_VAE_PATH`. Keeping these outside the model directory preserves the standalone
checkpoint contract while making caption and image encoding available to offline trainer containers.

Diffusers repositories and the Z-Image, Qwen-Image, Ideogram4, and Krea2 AI Toolkit model types retain complete snapshots;
their component files are never flattened or relocated. Before reusing any image-model cache, the downloader compares it
with Hugging Face repository metadata and checks sharded-index references. A complete standalone FLUX cache that still has
root metadata such as `README.md` or `.gitattributes` is normalized automatically. Incomplete caches are replaced from a
validated temporary sibling directory, and new downloads are promoted only after validation succeeds.

This behavior lives in the downloader image, not in miner training repositories. To deploy a change, rebuild
`ops/docker/trainer-downloader.dockerfile`, publish and configure an actually built versioned image tag, pull it on trainer
hosts, and restart trainer services as needed. Do not rely solely on the mutable `latest` tag; already-running services and
locally cached downloader images do not acquire the fix automatically.
