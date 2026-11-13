# Trainer

Service that receives work from the validator, clones miner repositories, builds training Docker images, runs jobs on GPUs, tracks job state, and uploads completed models.

## Contents

- `asgi.py`: FastAPI app factory, startup cleanup, and service entrypoint. It loads `.trainer.env` before any project import, because modules such as `minio_client.py` read their settings at import; variables already in the environment (`uvicorn --env-file`) take precedence.
- `cleanup.py`: background cleanup loop for stale trainer state and cache (training stale timeout uses `started_at`, refreshed when the train container starts). Every 10 minutes it removes created/exited containers older than `STALE_CONTAINER_MAX_AGE_HOURS` whose names start with one of `TRAINER_CONTAINER_NAME_PREFIXES` (the trainer's own), with their anonymous volumes; other containers on the host and all named volumes are left alone, and there is no host-wide volume prune. Containers the trainer removes itself also go with `v=True`, and image builds pass `rm`/`forcerm` so no intermediate build container is left behind.
- `constants.py`: trainer Docker images, volumes, resource defaults, and container paths.
- `containers/`: downloader, uploader, cache cleanup, and miner dataset cache helpers.
- `endpoints.py`: trainer API routes called by validators.
- `host.py`: host/GPU inspection, repository cloning, and Docker host helpers.
- `job_state.py`: persisted trainer task/model-prep state (`started_at` is set on accept and refreshed at container start for training jobs).
- `dataset_adapters.py`: text dataset column adapters used by training entrypoints.
- `diffusion_dataset.py`: diffusion image dataset extraction/arrangement helpers.
- `model_artifacts.py`: model artifact path and metadata helpers.
- `model_prep/`: baseline/model-prep routines.
- `quant_decode/`: quant decoder sandbox (two-phase build/run container specs and runner), the decoded-output check,
  the freeform decode job run after a freeform job's HF upload, and its ops re-run CLI (`rerun.py`).
- `output_reclaim.py`: `reclaim_output_dir`, a short fixed-command root container (`CHOWN`, `DAC_READ_SEARCH`,
  `FOWNER` only) that chowns a finished job's output dir to the trainer before the quant upload check.
- `quant_upload_check.py`: pre-upload check of a quant output dir against its `quant_manifest.json`: identity,
  listed sizes, size cap, prune of unlisted files, then the artifact file check.
- `runtime.py`: core Docker orchestration for downloader, model prep, training, env sidecars, and upload containers.
  Miner images are always built with no cache, so no entrant's build reuses another's layers. A build whose step
  failed (the build stream reports an error chunk without raising) is detected by the missing image tag and fails the
  job at the build stage with `Docker image build failed: no image ...`. After each job the
  image is removed and dangling images pruned; the host-wide BuildKit cache prune (`docker builder prune`) runs only
  while `TRAINER_PRUNE_BUILD_CACHE` is `true` (the default). Set it to `false` on a box that shares Docker with other
  builds: the prune reclaims nothing the trainer built but wipes their caches.
- `telemetry.py`: trainer logging setup.
- `training_config.py`: trainer-only config file writing and reward function materialization helpers.
- `training_paths.py`: canonical paths used inside training containers.

## Quant jobs

A quant job's uploaded bytes are its score (freeform) or its served model (fixed), so the trainer uploads exactly
the files the miner declared and changes none of them.

- Fixed track: the miner container gets `--task-type QuantFixedTask --quant-spec <QuantFixedSpec json>` and the same
  JSON in `QUANT_SPEC`.
- Reference bar: when the train request carries a `quant_target` (`core/models/quant_models.py:QuantTarget`), the miner
  container gets the QuantTarget JSON in the `QUANT_TARGET` env var only (never a flag: an older miner's strict argparse would
  reject it). Without one, it is not set.
- Output ownership: the miner's container runs as root with every capability dropped, so its output is root-owned
  (0o755 dirs, or any mode the miner set) and a non-root trainer could not delete anything in it. Before the check,
  `reclaim_output_dir` runs one container from the uploader image (`OUTPUT_RECLAIM_DOCKER_IMAGE`, already on every
  trainer) as root with only `CHOWN`, `DAC_READ_SEARCH` and `FOWNER`, no network and a read-only root: `chown -R -P`
  the output dir to the trainer's uid:gid, then `chmod -R u+rwX`. The checkpoints volume is mounted by name, so no
  host path is resolved through a miner's symlink; an output or task dir that is a symlink is rejected
  (`quant output rejected: output dir or its task dir is a symlink`); symlinks inside are never followed; chown clears
  setuid/setgid bits. A reclaim that cannot run (image missing, timeout after `OUTPUT_RECLAIM_TIMEOUT_SECONDS`) fails
  the job without upload, like a failed upload container. Only the job's own dir is walked; a hardlink in it to another
  file on the volume changes that inode too, which grants nothing new, since every training container already mounts
  the whole checkpoints volume read-write. Should a file still be out of the trainer's reach, the check rejects the
  output (`output not accessible to the trainer`) rather than crash or upload it unchecked.
- Upload: after the reclaim, `check_quant_output` reads `quant_manifest.json` at the output root, checks
  `task_id`, track and every listed file's size, caps the total at `QUANT_ARTIFACT_MAX_BYTES_MULTIPLIER` x the base
  model's bf16 bytes (skipped, with a warning, when the base model is unreadable), deletes everything unlisted and
  runs `check_artifact_files`. Any rejection (`quant output rejected: ...`, including a missing manifest after a
  timeout) makes the job a training failure with no upload and no decode. An accepted output is uploaded with
  `UPLOAD_VERBATIM=1`: the uploader skips subfolder detection, README/adapter_config patching and the training-state
  ignore patterns, since any rewrite would change the counted bytes. That flag needs a rebuilt `gradientsio/trainer-uploader:latest`; an old image ignores it.
- Build-context review: before every text-task image build, `flag_large_build_context_files` lists files over `QUANT_LARGE_FILE_REVIEW_BYTES` in the checkout (`.git` skipped, LFS objects included) to the log and the task log as `[REVIEW] ...`. It never fails the job; weights-derived payloads over about 1 MB are banned and enforced by manual review.
- Freeform track: when `TrainRequestQuant.decode` is set, `runtime.py` runs `quant_decode/job.py` after the HF upload
  and before `complete_task`, still inside the job's GPU slot. The outcome lands on `TrainerTaskLog.quant_decode`;
  it never changes training success. `cleanup.py` extends the stale deadline by `QUANT_DECODE_GRACE_MINUTES` for
  such jobs. After an `infra_error`, `python -m trainer.quant_decode.rerun <task_id> <hotkey>` re-runs the decode on
  the trainer and stores the new outcome (see `quant_decode/README.md`).
- `QUANT_DECODE_WORK_ROOT` defaults to a path under `Path.home()`, which assumes the trainer runs on the host (pm2 as the
  trainer user): the sandbox containers bind-mount that path from the Docker host, so it must exist there. A containerised
  trainer must set `QUANT_DECODE_WORK_ROOT` to a host-visible path (the same path on host and in the container), or the
  sandbox mounts a directory that is not the one the trainer wrote.
- The decode uploads to MinIO, so `.trainer.env` needs `S3_COMPATIBLE_ENDPOINT`, `S3_COMPATIBLE_ACCESS_KEY`,
  `S3_COMPATIBLE_SECRET_KEY` and `S3_REGION` (read at import by `validator/infrastructure/minio_client.py`).
