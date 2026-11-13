# Trainer Tests

Tests for trainer behavior, model prep, dataset/cache flow, and anonymization.

## Contents

- `model_prep/`: tests for model-prep statistics, sidecars, and roundtrips.
- `test_build_context_review.py`: build-time scan flags build-context files over `QUANT_LARGE_FILE_REVIEW_BYTES` (never `.git`) and stays silent otherwise.
- `test_dataset_and_anonymizer_flow.py`: trainer dataset download/cache and model anonymizer flow tests.
- `test_downloader_image_cache.py`: network-free image-model cache classification, migration, validation, and atomicity tests.
- `test_uploader_tokenizer_sanitize.py`: the HF uploader pushes task outputs with a v4-loadable tokenizer config.
- `test_image_build_cleanup.py`: miner image builds keep `nocache` and remove their intermediate containers; the
  post-job host build-cache prune follows `TRAINER_PRUNE_BUILD_CACHE` (default on).
- `test_quant_decode_output_check.py`: decoded-output check on hand-built safetensors files.
- `test_quant_decode_sandbox.py`: decoder sandbox container specs and phase runner against a fake Docker client.
- `test_quant_decode_job.py`: freeform decode job against a local HTTP snapshot server, fake HF download, fake Docker
  and fake MinIO: happy path, unsafe archives, executable artifacts, phase failures, infra errors, work-dir cleanup
  (a decoder's 0o000 subdir is removed and a re-run succeeds), idempotent `_WorkDirs.create` (empties a stale dir,
  moves aside one it cannot remove), and both phases running as the trainer's uid:gid with owner-only work dirs.
- `test_quant_decode_workdir_docker.py`: `sandbox.run_phase` with the real specs against real containers: with alpine
  standing in, the sandbox runs as the trainer's uid with `HOME=/tmp`, writes `/build` and `/output` but no ro mount,
  and leaves no dir the trainer cannot remove or re-create; with the pinned decoder images (when present), `build.sh`
  compiles into `/build` and `decode.py` imports torch, loads it and writes `/output`. Skipped without a docker daemon
  or the images.
- `test_output_reclaim.py`: the output reclaim container spec (root, only `CHOWN`/`DAC_READ_SEARCH`/`FOWNER`, no
  network, read-only root, the checkpoints volume by name) and runner against a fake Docker client: symlink rejection,
  non-zero exit, container error and timeout.
- `test_output_reclaim_docker.py`: a root `--cap-drop ALL` alpine "miner" writes nested, 0o000, setuid and symlink
  entries; without the reclaim a non-root trainer's check rejects it, after the real reclaim (uploader image) it owns
  everything, setuid is cleared and the prune succeeds; a symlinked output dir is rejected and never followed. Skipped
  without a docker daemon, alpine or the uploader image.
- `test_quant_train_contract.py`: `--quant-spec`/`QUANT_SPEC` and `--quant-target`/`QUANT_TARGET` reach the miner container; a quant output without a
  valid manifest (including after a timeout) fails with no upload or decode; an accepted one uploads with
  `UPLOAD_VERBATIM=1` and instruct jobs never do; the output is reclaimed before the check (never for instruct jobs),
  and a symlink rejection or a failed reclaim fails the job without upload; a freeform decode outcome is stored on the job and never fails training; a large build-context file is logged to the task for review before the build and never fails it.
- `test_quant_upload_check.py`: pre-upload manifest check on tmp output dirs: missing/invalid manifest, identity,
  sizes, symlinks (a symlinked output dir included), size cap, pruning of unlisted files, an unremovable unlisted file
  as a clean rejection, artifact file check, and the bf16-based cap.
- `test_quant_decode_rerun.py`: decode re-run CLI against a tmp `task_history.json` and a fake decode: stored outcome
  and exit codes.
- `test_uploader_verbatim.py`: uploader `main()` with fake `HfApi`: `UPLOAD_VERBATIM=1` uploads the root unpatched;
  without it subfolder detection and metadata patching still run.
- `test_model_prep_quant_extractor.py`: model-prep extractor choice for quant tasks (chat records vs instruct records);
  marked `gpu` because `stats.py` imports torch.
- `test_trainer_cleanup_scope.py`: the stale-container cleanup removes only the trainer's own prefixed containers (with
  their anonymous volumes), never other host containers or volumes, and every container name the trainer assigns is
  under a cleanup prefix.
- `test_trainer_env_load_order.py`: importing `trainer.asgi` loads `.trainer.env` before the S3 settings are read, and the
  process environment still wins over the file.
- `test_trainer_get_task_type.py`: task-type routing for quant, text and image requests.
- `__init__.py`: package marker.
