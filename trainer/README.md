# Trainer

Service that receives work from the validator, clones miner repositories, builds training Docker images, runs jobs on GPUs, tracks job state, and uploads completed models.

## Contents

- `asgi.py`: FastAPI app factory, startup cleanup, and service entrypoint.
- `cleanup.py`: background cleanup loop for stale trainer state and cache.
- `constants.py`: trainer Docker images, volumes, resource defaults, and container paths.
- `containers/`: downloader, uploader, cache cleanup, and miner dataset cache helpers.
- `endpoints.py`: trainer API routes called by validators.
- `host.py`: host/GPU inspection, repository cloning, and Docker host helpers.
- `job_state.py`: persisted trainer task/model-prep state.
- `dataset_adapters.py`: text dataset column adapters used by training entrypoints.
- `diffusion_dataset.py`: diffusion image dataset extraction/arrangement helpers.
- `model_artifacts.py`: model artifact path and metadata helpers.
- `model_prep/`: baseline/model-prep routines.
- `runtime.py`: core Docker orchestration for downloader, model prep, training, env sidecars, and upload containers.
- `telemetry.py`: trainer logging setup.
- `training_config.py`: trainer-only config file writing and reward function materialization helpers.
- `training_paths.py`: canonical paths used inside training containers.
