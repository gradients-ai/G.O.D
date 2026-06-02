# Trainer Containers

Helpers for support containers launched by the trainer runtime.

## Contents

- `cache_cleanup.py`: removes stale cache entries and old trainer artifacts.
- `dataset_cache.py`: downloads approved miner-requested Hugging Face datasets into the shared cache volume.
- `downloader.py`: pre-downloads models and task datasets before training.
- `uploader.py`: uploads completed model artifacts to Hugging Face.
- `__init__.py`: package marker.
