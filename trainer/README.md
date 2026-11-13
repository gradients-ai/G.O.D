# Trainer

The service that takes jobs from the validator, clones miner repositories, builds their images, runs them on GPUs, tracks job state and uploads the results.

## Contents

| Path | Purpose |
| --- | --- |
| `asgi.py` | FastAPI app factory, startup cleanup and service entrypoint. Loads `.trainer.env` before any project import. |
| `cleanup.py` | Background loop that fails overdue jobs, runs the cache cleaner and removes the trainer's stale containers. |
| `constants.py` | Docker images, volumes, resource defaults, container paths and compression-task limits. |
| `containers/` | Downloader, uploader, cache cleanup and requested-dataset cache. |
| `dataset_adapters.py` | Column adapters for text datasets. |
| `diffusion_dataset.py` | Arranges image datasets for diffusion training. |
| `endpoints.py` | API routes the validator calls. |
| `host.py` | GPU inspection, repository cloning and Docker host helpers. |
| `job_state.py` | Persisted job and model-prep state (`task_history.json`). |
| `model_artifacts.py` | Anonymous model directory names and identity scrubbing. |
| `model_prep/` | Model preparation and baseline statistics. |
| `output_reclaim.py` | Gives a finished job's output directory to the trainer user before it is checked. |
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
- After each job the image is removed and dangling images are pruned.
- The host-wide BuildKit cache prune runs while `TRAINER_PRUNE_BUILD_CACHE` is `true`, the default. Set it to `false` on a host that shares Docker with other builds.

## Cleanup

Every 10 minutes `cleanup.py` removes created or exited containers older than `STALE_CONTAINER_MAX_AGE_HOURS`, with their anonymous volumes. It touches only containers whose names start with one of `TRAINER_CONTAINER_NAME_PREFIXES`. Other containers and all named volumes are left alone.

A training job's stale timeout runs from `started_at`, which is set when the job is accepted and refreshed when its container starts.

## Compression jobs

A compression job runs a `QuantFixedTask` or a `QuantFreeformTask`. Its uploaded bytes are its score (freeform) or its served model (fixed). The trainer uploads exactly the files the miner declared and changes none of them. The miner-facing contract is in the [compression guide](../docs/miner_compression.md).

| Step | What happens |
| --- | --- |
| Inputs | Fixed tasks get `--quant-spec` and `QUANT_SPEC`. `QUANT_TARGET` and `QUANT_CALIBRATION_PATH` are set when the request carries them. |
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
- The checkpoints volume is mounted by name. An output or task directory that is a symlink is rejected.
- A reclaim that cannot run (image missing, or a timeout after `OUTPUT_RECLAIM_TIMEOUT_SECONDS`) fails the job without upload.
- A file the trainer cannot reach after the reclaim makes the check reject the output (`output not accessible to the trainer`).

</details>

## See also

- [Running a trainer](../docs/developer.md#running-a-trainer): setup and configuration.
- [How your code runs](../docs/miner.md#how-your-code-runs): the container contract miners see.
