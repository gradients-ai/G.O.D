# Trainer

Service that receives work from the validator, clones miner repositories, builds training Docker images, runs jobs on GPUs, tracks job state, and uploads completed models.

## Contents

- `asgi.py`: FastAPI app factory, startup cleanup, GPU throttle monitor, and service entrypoint.
- `cleanup.py`: background cleanup loop for stale trainer state and cache (training stale timeout uses `started_at`, refreshed when the train container starts).
- `constants.py`: trainer Docker images, volumes, resource defaults, and container paths.
- `containers/`: downloader, uploader, cache cleanup, and miner dataset cache helpers.
- `endpoints.py`: trainer API routes called by validators.
- `monitoring/`: GPU monitor started with the trainer. `monitor.py` posts Discord alerts, fails trainings on the bad GPU, and withdraws that GPU. `blacklist.py` keeps withdrawn GPUs out for the life of the process. `fault.py` stops their containers. `health.py` is the NVML ECC, NVLink, and Xid fallback. `dcgm.py` parses `dcgmi health` after `ops/trainer_ops/install_dcgm.sh` installs DCGM. `throttle.py` is the clock-throttle rules.
- `host.py`: host/GPU inspection, repository cloning, and Docker host helpers.
- `job_state.py`: persisted trainer task/model-prep state (`started_at` is set on accept and refreshed at container start for training jobs).
- `dataset_adapters.py`: text dataset column adapters used by training entrypoints.
- `diffusion_dataset.py`: diffusion image dataset extraction/arrangement helpers.
- `model_artifacts.py`: model artifact path and metadata helpers.
- `model_prep/`: baseline/model-prep routines.
- `runtime.py`: core Docker orchestration for downloader, model prep, training, env sidecars, and upload containers.
- `telemetry.py`: trainer logging setup.
- `training_config.py`: trainer-only config file writing and reward function materialization helpers.
- `training_paths.py`: canonical paths used inside training containers.
