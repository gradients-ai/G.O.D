# Trainer tests

Tests for the trainer service: job runtime, containers, cleanup, model prep and the compression upload and decode path.

## Contents

| Path | Covers |
| --- | --- |
| `model_prep/` | Model-prep statistics, device probes, merged-repo naming and round trips. |
| `test_build_context_review.py` | Large build-context files are flagged for review, `.git` excluded, and never fail a build. |
| `test_container_resources.py` | Training container CPU limits never exceed the host. |
| `test_dataset_and_anonymizer_flow.py` | Requested-dataset whitelist validation, field propagation and model anonymisation. |
| `test_downloader_image_cache.py` | Image-model cache classification, validation and atomic replacement, without network. |
| `test_image_build_cleanup.py` | Miner image builds use no cache and leave no intermediate containers. The build-cache prune follows `TRAINER_PRUNE_BUILD_CACHE`. |
| `test_job_state_started_at.py` | `started_at` is refreshed when a training container starts. |
| `test_model_prep_quant_extractor.py` | Model-prep extractor choice for compression tasks. Marked `gpu`. |
| `test_output_reclaim.py` | Output reclaim container spec and runner against a fake Docker client. |
| `test_output_reclaim_docker.py` | Output reclaim against real containers. Skipped without Docker or the images. |
| `test_quant_calibration.py` | The calibration file is written for the miner, and any failure drops it without failing the job. |
| `test_quant_decode_job.py` | Freeform decode job against fake snapshot, Hugging Face, Docker and storage backends. |
| `test_quant_decode_output_check.py` | Decoded-output check on hand-built safetensors files. |
| `test_quant_decode_rerun.py` | Decode re-run CLI: stored outcome and exit codes. |
| `test_quant_decode_sandbox.py` | Decoder sandbox container specs and phase runner against a fake Docker client. |
| `test_quant_decode_workdir_docker.py` | Sandbox phases against real containers: ownership, mounts and cleanup. Skipped without Docker or the images. |
| `test_quant_train_contract.py` | Compression inputs reach the miner container, and the check, upload and decode sequence holds. |
| `test_quant_upload_check.py` | Pre-upload manifest check: identity, sizes, symlinks, size cap and pruning. |
| `test_telemetry_vector_handler.py` | The Vector log handler never raises when its sink is absent. |
| `test_trainer_cleanup_scope.py` | Stale-container cleanup removes only the trainer's prefixed containers. |
| `test_trainer_env_load_order.py` | `.trainer.env` is loaded before settings are read, and the process environment wins. |
| `test_trainer_get_task_type.py` | Task-type routing for text, image and compression requests. |
| `test_uploader_tokenizer_sanitize.py` | The uploader pushes outputs with a loadable tokenizer config. |
| `test_uploader_verbatim.py` | `UPLOAD_VERBATIM=1` uploads the root unpatched. Without it, subfolder detection and patching run. |

## See also

- [Trainer](../../trainer/README.md): the module under test.
- [Testing and linting](../../docs/developer.md#testing-and-linting): how to run the tests.
