# Trainer

Service that receives work from the validator, clones miner repositories, builds training Docker images, runs jobs on GPUs, tracks job state, and uploads completed models.

## Contents

- `asgi.py`: FastAPI app factory, startup cleanup, GPU throttle monitor, and service entrypoint.
- `cleanup.py`: background cleanup loop for stale trainer state and cache (training stale timeout uses `started_at`, refreshed when the train container starts).
- `constants.py`: trainer Docker images, volumes, resource defaults, and container paths.
- `containers/`: downloader, uploader, cache cleanup, and miner dataset cache helpers.
- `endpoints.py`: trainer API routes called by validators.
- `gpu_monitor.py`: GPU throttle monitor started with the trainer; posts Discord alerts, fails trainings on the bad GPU, and withdraws that GPU from new assignments.
- `gpu_blacklist.py`: GPUs withdrawn after a throttle fault for the life of the trainer process.
- `gpu_fault.py`: marks those trainings failed and stops their containers.
- `gpu_health.py`: NVML ECC, NVLink, and Xid rules used when DCGM is not available.
- `gpu_dcgm.py`: `dcgmi health` parser. DCGM is installed by `ops/trainer_ops/install_dcgm.sh` when the trainer starts. If that install fails, the NVML checks are used.
- `gpu_throttle.py`: pure throttle detection and alert formatting used by `gpu_monitor.py`.
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
