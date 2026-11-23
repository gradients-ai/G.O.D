# Trainer

The service that takes jobs from the validator, clones miner repositories, builds their images, runs them on GPUs, tracks job state and uploads the results.

## Contents

| Path | Purpose |
| --- | --- |
| `asgi.py` | FastAPI app factory, startup cleanup, GPU monitor start and service entrypoint. Loads `.trainer.env` before any project import. |
| `cleanup.py` | Background loop that fails overdue jobs, runs the cache cleaner and removes the trainer's stale containers. |
| `constants.py` | Docker images, volumes, resource defaults, container paths and compression-task limits. |
| `containers/` | Downloader, uploader, cache cleanup and requested-dataset cache. |
| `dataset_adapters.py` | Column adapters for text datasets. |
| `diffusion_dataset.py` | Arranges image datasets for diffusion training. |
| `endpoints.py` | API routes the validator calls. |
| `host.py` | GPU inspection, repository cloning and Docker host helpers. |
| `host_resources.py` | Splits the host's cores and RAM into one equal share per GPU; every job container gets its GPUs' shares. |
| `job_state.py` | Persisted job and model-prep state (`task_history.json`). |
| `model_artifacts.py` | Anonymous model directory names and identity scrubbing. |
| `model_prep/` | Model preparation and baseline statistics. |
| `monitoring/` | GPU monitor started with the trainer. `monitor.py` posts Discord alerts, fails trainings on the bad GPU and withdraws that GPU. `blacklist.py` keeps withdrawn GPUs out for the life of the process. `fault.py` stops their containers. `health.py` is the NVML ECC, NVLink and Xid fallback. `dcgm.py` parses `dcgmi health` after `ops/trainer_ops/install_dcgm.sh` installs DCGM. `throttle.py` holds the clock-throttle rules. |
| `output_reclaim.py` | Gives a finished job's output directory to the trainer user before it is checked. |
| `job_checkpoints.py` | Each job's own directory on the checkpoints volume, the only part of it the job's containers mount. |
| `quant_calibration.py` | Fetches a task's calibration set and writes it into the cache volume for the miner. |
| `quant_decode/` | Sandbox that runs a freeform decoder, the decoded-output check and the re-run CLI. |
| `quant_upload_check.py` | Checks a compression task's output against its `quant_manifest.json` before upload. |
| `runtime.py` | Docker orchestration: download, model prep, image build, training, upload, decode. |
| `telemetry.py` | Logging setup. |
| `training_config.py` | Writes training config files and reward function files. |
| `training_paths.py` | Paths used inside training containers. |

## Image builds

- Miner images are built with no cache, so one entrant's build never reuses another's layers.
- A build whose step failed is detected by the missing image tag. The job fails at the build stage with `Docker image build failed: no image ...`.
- Before every compression-task build, files over `QUANT_LARGE_FILE_REVIEW_BYTES` in the checkout are logged as `[REVIEW] ...`. This never fails the job.
- After each job the image is removed. Dangling images are pruned after each job and by the periodic cleanup, but only while no image build is in progress on the trainer (a build abandoned on timeout counts until the daemon ends it).
- The host-wide BuildKit cache prune runs while `TRAINER_PRUNE_BUILD_CACHE` is `true`, the default. Set it to `false` on a host that shares Docker with other builds.

## CPU and memory shares

`host_resources.py` reads the host once per trainer process: the CPUs the trainer may use, their hyperthread siblings, RAM, and each GPU's local CPUs (`nvidia-smi` bus id, then `/sys/bus/pci/devices/<bdf>/local_cpulist`). If `nvidia-smi` fails, the GPU count comes from `/proc/driver/nvidia/gpus` and the split is even.

- Each GPU owns an equal set of whole physical cores. GPUs on separate NUMA domains take cores from their own domain; when the affinity is missing, overlapping or too uneven for equal shares, the cores are split evenly in order.
- `HOST_CPU_RESERVE_FRACTION` of the cores (at least one) and `HOST_MEMORY_RESERVE_FRACTION` of RAM (at least `HOST_MEMORY_RESERVE_MIN_GB`) stay with the host. The lowest cores are the ones kept.
- A job with N GPUs gets N shares: `cpuset_cpus` for the union of their cores, N memory shares as `mem_limit`, and the thread-pool variables (`THREAD_ENV_VARS`) set to one GPU's share (one process per GPU). A pinned job has no CFS quota on top.
- Training, image training and model prep get the full share. The job's downloader, requested-dataset download, uploader and freeform decode run on the same cores, inside the job's GPU reservation (decode keeps its `DECODE_CPUS` quota). Cache cleanup and output reclaim are not pinned.
- The host reserve is only cores no job is given: the trainer, dockerd and the unpinned containers may still run on any core.
- When the GPUs' NUMA domains cannot give equal local shares (e.g. both GPUs on one socket of a two-socket host), the even split is fair but not local.
- A host with fewer physical cores than GPUs is shared by quota (`nano_cpus`) instead.

The split is logged once at the first job (`Host split: ...`) and per container (`Allocating GPUs [...]`).

## Cleanup

Every 10 minutes `cleanup.py` removes created or exited containers older than `STALE_CONTAINER_MAX_AGE_HOURS`, with their anonymous volumes. It touches only containers whose names start with one of `TRAINER_CONTAINER_NAME_PREFIXES`. Other containers and all named volumes are left alone.

A training job's stale timeout runs from `started_at`, which is set when the job is accepted and refreshed when its container starts.

## Compression jobs

A compression job runs a `QuantFixedTask` or a `QuantFreeformTask`. Its uploaded bytes are its score (freeform) or its served model (fixed). The trainer uploads exactly the files the miner declared and changes none of them. The miner-facing contract is in the [compression guide](../docs/miner_compression.md).

| Step | What happens |
| --- | --- |
| Inputs | Fixed tasks get `--quant-spec` and `QUANT_SPEC`. `QUANT_TARGET` and `QUANT_CALIBRATION_PATH` are set on every compression job; the calibration file carries the reference losses. |
| Reclaim | `reclaim_output_dir` chowns the output directory to the trainer user. |
| Check | `check_quant_output` validates the manifest, sizes and file types, and deletes unlisted files. |
| Upload | The uploader runs with `UPLOAD_VERBATIM=1` and uploads the output root unmodified. |
| Decode | Freeform only: `quant_decode/job.py` runs after the upload, inside the job's GPU slot. |

Any rejection by the check (`quant output rejected: ...`) is a training failure with no upload and no decode. The size cap is `QUANT_ARTIFACT_MAX_BYTES_MULTIPLIER` x the base model's bf16 bytes. It is skipped, with a warning, when the base model is unreadable.

The decode outcome is stored on `TrainerTaskLog.quant_decode` and never changes training success. `cleanup.py` extends the stale deadline of such jobs by `QUANT_DECODE_GRACE_MINUTES`. See [`quant_decode/README.md`](quant_decode/README.md).

<details>
<summary>Output ownership</summary>

The miner's container runs as root with every capability dropped, so its output is root-owned and a non-root trainer cannot prune it.

- `reclaim_output_dir` runs one container from the uploader image as root, with only `CHOWN`, `DAC_READ_SEARCH` and `FOWNER`, no network and a read-only root.
- Its fixed command is `chown -R -P` to the trainer's uid:gid, then `chmod -R u+rwX`. Symlinks are never followed, and chown clears setuid and setgid bits.
- Only the job's own directory of the checkpoints volume is mounted. An output or task directory that is a symlink is rejected.
- A reclaim that cannot run (image missing, or a timeout after `OUTPUT_RECLAIM_TIMEOUT_SECONDS`) fails the job without upload.
- A file the trainer cannot reach after the reclaim makes the check reject the output (`output not accessible to the trainer`).

</details>

## See also

- [Running a trainer](../docs/developer.md#running-a-trainer): setup and configuration.
- [How your code runs](../docs/miner.md#how-your-code-runs): the container contract miners see.
