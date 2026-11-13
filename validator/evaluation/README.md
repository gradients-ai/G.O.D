# Validator evaluation

Evaluation of trained submissions: dispatch to remote GPUs, the evaluators, model checks and result processing.

## Contents

| Path | Purpose |
| --- | --- |
| `evaluators/` | Evaluator entrypoints, one per task type. |
| `comfy_workflows/` | Location for ComfyUI workflow templates. |
| `basilica.py` | Basilica backend: deploys an evaluator, polls its result, retries and cleans up. |
| `basilica_deployments.py` | Deployment health checks, deletion and the runner source a deployment starts. |
| `chat_labels.py` | The chat labelling rule: which tokens of a conversation are scored. Standard library and pydantic only. |
| `chat_templates.py` | Chat template resolution by name, and the tokenizer adapter for `chat_labels`. |
| `common.py` | Shared text evaluation helpers: model and tokenizer loading, parameter counts, eval-set fingerprint. |
| `constants.py` | Evaluation constants: GPU SKUs, engine settings, deployment timings and container env. |
| `dataset_configs.py` | Picks a dataset's default config name. |
| `dataset_utils.py` | Builds the placeholder train dataset an evaluation run needs. |
| `db_utils.py` | Evaluation helpers that need the database. |
| `denoising_mse.py` | Shared flow-prediction cases and the L2 error for image evaluation. |
| `docker_evaluation.py` | Remote evaluation dispatch per task type, and presigned URLs for decoded models. |
| `errors.py` | Failure classes, `is_infra_error` and `describe_error`. Standard library only. |
| `evaluation_logging.py` | Evaluation step logging and cleaning of relayed deployment logs. |
| `image_artifacts.py` | Pinned model artifact loading and atomic result writing for image evaluation. |
| `image_denoising.py` | Prediction evaluation on the shared test cases. |
| `image_encoder.py` | Deterministic float32 VAE encoding. |
| `image_flow_adapter.py` | Per-family image encoders and the flow and LoRA contracts. |
| `image_io.py` | Image loading, resizing and base64 helpers. |
| `image_test_data.py` | Deterministic test image loading and preparation. |
| `local_evaluation.py` | Evaluation in Docker on local GPUs. |
| `model_checks.py` | Whether a model is a finetune of its base, and LoRA checks. |
| `models.py` | Payload models of the evaluation runtimes. |
| `notifications.py` | Alerts for evaluation exceptions. |
| `quant_artifacts.py` | Artifact byte count from the hub, and the sha256-verified decoded freeform model. |
| `quant_config_check.py` | Fixed-track check of a checkpoint's `config.json` against the spec and the input model. |
| `quant_gpu.py` | Prints the evaluation GPU as `QuantEvalGpu` JSON. Runs as a subprocess in the scorer image. |
| `quant_loss.py` | Test loss from prompt logprobs, the eval-set fingerprint and the kernel check. No torch or vLLM. |
| `quant_structure.py` | Fixed-track structural pre-check from hub metadata, before any GPU is used. |
| `quant_vllm.py` | vLLM prompt-logprob engine and the per-layer spec and kernel check. Imported only in the scorer image. |
| `reconcile.py` | Reconciles live deployments with GPU reservations. |
| `remote.py` | Selects the remote backend from `EVAL_BACKEND`. |
| `remote_runner.py` | HTTP wrapper that runs inside a remote evaluator container and serves `/result`. |
| `result_processing.py` | Converts raw evaluator output into result models. |
| `runpod.py` | Runpod backend, implemented as dstack services. |
| `runtime.py` | Process stop and deployment health wait helpers. |
| `utils.py` | Re-exports of commonly used evaluation helpers. |

## Compression evaluation path

A compression task is scored through vLLM. The serving kernel decides the loss.

| Rule | Detail |
| --- | --- |
| Loss per example | `quant_loss.example_loss` turns prompt logprobs into shifted cross-entropy. A supervised first token is dropped. |
| `eval_loss` | The flat mean over examples, one value per example. |
| One deployment per submission | Each deployment scores one repo. |
| Scorer imports | `evaluators/quant.py` and `single_quant.py` import neither `common.py` nor `core.logging`. |

### Fixed track checks

A fixed-track submission passes three checks. Failing any is a scored fail.

| Check | When | What it requires |
| --- | --- | --- |
| Structure (`quant_structure`) | Before any GPU | The tensor set (name, dtype, shape) equals what the base model implies under the spec. Hub metadata only. |
| Config (`quant_config_check`) | Before loading | `quantization_config` describes exactly the spec, and every other field equals the input model's. |
| Layers (`quant_vllm`) | After loading | Every linear layer is quantised to the spec and runs a kernel in `QUANT_EXPECTED_KERNEL_CLASSES`. |

The contract a checkpoint must meet is in [Fixed track: the checkpoint](../../docs/miner_compression.md#fixed-track-the-checkpoint).

<details>
<summary>Config check internals</summary>

- Both configs are compared as vLLM resolves them (`quant_vllm.resolved_model_config`). A key missing on one side is a difference.
- Only `CONFIG_KEYS_FREE_TO_DIFFER` may differ: `quantization_config`, `transformers_version`, `_name_or_path`, `dtype` and `torch_dtype`.
- The input model's config failing to resolve is an evaluation error. The submission's failing to resolve is a scored fail.
- The layer check covers `LinearBase` layers, so it covers dense base models.

</details>

### What reaches the engine

| Track | Loaded from | Read from the submission |
| --- | --- | --- |
| Fixed | The miner's Hugging Face repo | Weights (safetensors only) and `config.json` (HF format only). |
| Freeform | The decoded bf16 model in object storage | Nothing. The model dir is rebuilt from the base model's config and tokenizer plus the decoded shards. |

On both tracks the tokenizer is the input model's, prompts enter as token ids, and `generation_config.json` supplies no sampling defaults (`QUANT_VLLM_*` in `constants.py`). Miner code never runs at scoring time.

### Freeform decoded models

1. The trainer decode job uploads bf16 safetensors and `index.json`, written last, under `quant-decoded/<task_id>/<hotkey>/`.
2. `decoded_model_urls_for` presigns each complete prefix. Each deployment receives only its repo's URLs in `DECODED_MODEL_URLS`.
3. The evaluator downloads the shards the index lists and checks each by byte count and sha256.

| Situation | Outcome |
| --- | --- |
| A shard fails its byte count or sha256 | Scored fail |
| A download fails | Evaluation error |
| No complete prefix exists | Tournament task: deferred, returned to `pending`, not counted. Other freeform tasks: evaluation error. The repo is not sent to a GPU. |

`artifact_bytes` is measured by the validator from the Hugging Face repo. Every file counts.

> [!IMPORTANT]
> Decoded models are materialised under `TMPDIR` in the scorer image. `TMPDIR` must point at a disk-backed path, because a tmpfs holds the whole model in RAM.

## Whose fault an evaluation failure is

Only a verdict on the repo is scored against the miner. `errors.py` defines the classes.

For compression tasks:

| Outcome | Examples | Handling |
| --- | --- | --- |
| Scored fail | `quant_failure` (config mismatch, load error, failed layer check, malformed `config.json`), a non-finite loss, the hub answering that the repo does not exist | Scored against the entry |
| Evaluation error (`EvaluationInfraError`) | See the list below | Re-queued, counted per row, never scored |
| Deployment never ready (`DeploymentNotReadyError`) | A pod that does not become healthy | Re-queued and counted |
| Capacity wait (`EvaluationCapacityUnavailable`) | No GPU available | Re-queued, not counted |
| Decode outcome pending | A freeform tournament row whose decode has not settled, or settled with no decoded model in object storage | Deferred: returned to `pending`, not counted |

Evaluation errors on a compression task:

- A deployment that fails or times out.
- A results file without the repo's entry, or an `{"evaluation_error": ...}` entry.
- An error string the evaluator reports for the repo.
- The hub, object storage or the database failing.
- A missing decoded model, or a missing reference model on a common-input task.
- Any exception out of the evaluation call.

On a seed task that no round has attached, an evaluation error goes to the seed-time retry in `validator/lifecycle/boss_seed_eval.py`, not to the counted path.

For text and image tasks:

| Outcome | Handling |
| --- | --- |
| Any per-repo error in the results: an evaluator error report, an `evaluation_error` entry or a missing result | Scored as a failure, with `Evaluation failed: ...` |
| Any failure of the hub repo lookup | Scored as a failure, with `Model repo not found on HuggingFace` |
| Any `EvaluationRetryableError` | Re-queued, not counted |
| Any other exception out of the evaluation call | Alert, and the rows are marked `failure` |

`is_infra_error` recognises transport errors, timeouts and HTTP 5xx, 408 and 429 anywhere in an exception's cause chain. It labels the alert and lets an evaluator report a hub or network failure as `{"evaluation_error": "<Type>: <message>"}`. `describe_error` adds the exception type to every message.

### Counted errors and the hold

`validator/lifecycle/eval_infra_errors.py` handles counted errors.

| Step | Behaviour |
| --- | --- |
| Each error | The row returns to `pending` and `evaluations.infra_error_attempts` increases. The first error alerts. |
| Backoff | A re-queued row is not dispatched for `EVAL_INFRA_REQUEUE_BACKOFF_SECONDS` (10 minutes). |
| At `MAX_EVAL_INFRA_ATTEMPTS` (3) | The row is held in `evaluation_status='infra_held'`. It is never dispatched or scored, and the task does not finalise. |
| While held | The alert carries the SQL to re-queue or fail the rows and repeats every `EVAL_INFRA_HELD_REMINDER_SECONDS` (4 hours). |

Only rows in `pending` or `evaluating` are counted. Settled rows are never reopened.

## Compression eval GPU SKUs

A compression entry is scored on the H100 family. Both SKUs are allowed and neither is preferred, so a task never waits for one SKU. `QUANT_EVAL_ALLOWED_SKUS` in `constants.py` lists them:

| Label | torch device name | SMs | Runpod GPU type |
| --- | --- | --- | --- |
| H100 SXM | `NVIDIA H100 80GB HBM3` | 132 | `NVIDIA H100 80GB HBM3` |
| H100 PCIe | `NVIDIA H100 PCIe` | 114 | `NVIDIA H100 PCIe` |

| Rule | Detail |
| --- | --- |
| Dispatch | For an entry, the dispatcher requests `QUANT_EVAL_GPU_MODELS` and narrows Runpod offers to `QUANT_EVAL_RUNPOD_INSTANCE_TYPES`. |
| Check | Before loading a model, `evaluators/quant.check_gpu` requires the device name and SM count to match one entry. |
| Mismatch | `QuantGpuMismatch`: the container exits non-zero. This is an evaluation error, never a scored fail. |
| Record | Every result carries `eval_gpu` (`name`, `sm_count`, `sku`, `override`), stored in `task_nodes.eval_gpu`. |
| Override | `QUANT_EVAL_ALLOW_ANY_GPU=1` admits any GPU for testing. The result carries `override=true` and no `sku`. |

### Reference-only runs

A bare bf16 reference measurement may also run on an 80 GB A100. `QUANT_REFERENCE_EXTRA_SKUS` lists them:

| Label | torch device name | SMs |
| --- | --- | --- |
| A100 SXM | `NVIDIA A100-SXM4-80GB` | 108 |
| A100 PCIe | `NVIDIA A100 80GB PCIe` | 108 |

The dispatcher passes `reference_only=True`. It requests `QUANT_REFERENCE_GPU_MODELS`, narrows Runpod offers to `QUANT_REFERENCE_RUNPOD_INSTANCE_TYPES` and sets `QUANT_REFERENCE_ONLY=1`. The evaluator accepts the flag only when `MODELS` is exactly `QUANT_REFERENCE_MODEL` with no `QUANT_SPEC` and no `DECODED_MODEL_URLS`. Any other run with the flag exits non-zero (`QuantReferenceOnlyMisuse`), so an entry is never scored off the H100 family.

A reference is stored under the SKU it ran on, and a score is only ever compared with the reference on the entry's SKU.

### Same-SKU bf16 reference

Scoring compares compression losses only as multiples of a bf16 reference scored on the same SKU. The scoring rule is in [Compression scoring](../scoring/README.md#compression-scoring). The reference is measured inside the evaluation deployments, so a submission and its reference share a GPU.

1. Per deployment, the dispatcher (`run_evaluation_basilica_quant`) sets `QUANT_REFERENCE_MODEL` and `QUANT_CACHED_REFERENCES`, a JSON map of `{reference_key: loss}` for the SKUs already cached.
2. After `check_gpu`, the evaluator scores the reference first when its SKU is not in the map. The reference uses the same test set and engine as the submission, with no spec. When the reference is itself one of the repos, it is scored once and its result is copied.
3. The result is written under `QUANT_REFERENCE_RESULT_KEY` and attached to the repo's result as `EvaluationResultText.quant_reference`.
4. A reference that errored or came back without a GPU record is logged and dropped. The submission's result stands, and finalise re-evaluates the row while its SKU has no reference.

<details>
<summary>Why the two H100 SKUs can differ in the low decimals</summary>

Kernel tiling, scheduling and reduction order read the GPU's SM count, and the two SKUs have different counts. The same weights can therefore score slightly differently on each, including a bf16 model. The scorer image targets no SM count. Normalising by a same-SKU reference removes the difference from rankings.

</details>

## See also

- [Compression scoring](../scoring/README.md#compression-scoring): how compression losses are normalised and ranked.
- [Evaluation](../../docs/miner_compression.md#evaluation): the miner-facing description.
- [Evaluation flow and debugging](../../ops/runbooks/evaluation_flow_debugging.md): the operator runbook.
- [Decoder sandbox](../../trainer/quant_decode/README.md): the decoder sandbox.
