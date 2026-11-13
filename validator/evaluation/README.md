# Validator Evaluation

Evaluation runtimes and helpers used after miner training completes.

## Contents

- `comfy_workflows/`: ComfyUI workflow JSON files for image evaluation.
- `evaluators/`: task-specific evaluator entrypoints for text, DPO, GRPO, and image tasks.
- `basilica.py`: Basilica client integration.
- `basilica_deployments.py`: Basilica deployment helpers.
- `chat_labels.py`: the chat labelling rule (which tokens of a conversation are scored: every assistant turn's content plus the template's turn end). Self-contained (stdlib + pydantic) because miners copy it verbatim.
- `chat_templates.py`: chat template resolution (`tokenizer_default`, `chatml`, `tokenizer_default_fallback_chatml`) and the transformers adapter for `chat_labels`.
- `common.py`: shared text evaluation helpers (base param counts via hub metadata, no weight reload).
- `constants.py`: evaluation constants (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on all remote eval containers).
- `remote_runner.py`: HTTP wrapper for Basilica/Runpod eval containers; serves `/result` even if a post-eval step exits non-zero when results already exist.
- `runpod.py`: Runpod evaluation backend via dstack services.
- `dataset_configs.py`: default dataset config discovery.
- `db_utils.py`: evaluation persistence helpers.
- `docker_evaluation.py`: Docker/Basilica evaluation orchestration.
- `errors.py`: whose fault an evaluation failure is. `EvaluationRetryableError` and its subclasses
  (`EvaluationCapacityUnavailable`, `DeploymentNotReadyError`, `EvaluationInfraError`), `RepoEvaluationFailed`,
  the one infra predicate `is_infra_error` and `describe_error`. Standard library only, because the evaluator images
  import it too. See "Whose fault an evaluation failure is" below.
- `evaluation_logging.py`: evaluation logging helpers.
- `image_io.py`: image loading and encoding helpers.
- `local_evaluation.py`: local evaluation runner.
- `model_checks.py`: model architecture and LoRA checks.
- `models.py`: evaluation payload models.
- `reconcile.py`: evaluation deployment and GPU-reservation reconciliation, including release of leftover PvP reservations from historical environment tasks.
- `quant_artifacts.py`: quant artifact byte count from HF and the sha256-verified decoded freeform model (no torch).
- `quant_gpu.py`: reads the quant eval GPU (name, SM count) with torch; run as a subprocess in the quant image.
- `quant_config_check.py`: fixed-track check of a checkpoint's config.json: `quantization_config` against the spec, and every
  other field against the input model's config, compared as vLLM resolves both (pure).
- `quant_loss.py`: quant test loss from prompt logprobs, eval-set fingerprint and kernel check (no torch/vllm).
- `quant_vllm.py`: vLLM prompt-logprob engine (input model's tokenizer, HF config format, safetensors only), the
  resolved-config view the drift check compares, Humming stream-K patch and per-layer spec/kernel check; imported only in
  the quant image.
- `result_processing.py`: conversion of raw eval output into persisted results.
- `runtime.py`: evaluation runtime helpers.
- `utils.py`: evaluation utility helpers.

Environment, InterCode, SWE Infinite, and PvP evaluation runtimes have been removed. `TaskType.ENVIRONMENTTASK` and stored PvP score rows remain readable for historical tournaments.

Quant tasks are scored through vLLM, not HF: the serving kernel is the ground truth, and a checkpoint that HF can
dequantise may still be served differently. `quant_loss.example_loss` turns vLLM prompt logprobs
into HF's shifted per-example cross-entropy (a supervised first token is dropped), and `eval_loss`
is the flat mean over examples, not over tokens, so a few long completions cannot dominate and the paired boss gate
gets one value per example. The GPU rule is in "Quant eval GPU SKUs" below. A fixed-track submission is checked
twice, because the kernel class alone does not pin the format (W6A16 and the W2A16 formats route to Humming like W3A16, W4A8-int8 to
Machete like W4A16):

- before loading, `quant_config_check` requires config.json to describe exactly the spec:
  compressed-tensors `pack-quantized`, one config group targeting `Linear`, the spec's weight and
  activation args, bf16 (or unstated) scales, only `lm_head` ignored, no `kv_cache_scheme`; and
  `config_drift_failure` requires every other field to be the input model's. vLLM builds the model from the
  submission's config, so `rms_norm_eps`, RoPE, `max_position_embeddings`, `tie_word_embeddings`, vocab and
  architecture fields would otherwise be free knobs. Both configs are compared as vLLM resolves them
  (`quant_vllm.resolved_model_config`, vLLM's `get_config` in HF format), because transformers 5 re-spells a v4
  config on save (`rope_theta` into `rope_parameters`, derived `layer_types`); a key missing on one side is a
  difference, so dropping a field to reach a class default fails too. Only `CONFIG_KEYS_FREE_TO_DIFFER` may
  differ: `quantization_config` (checked above), `transformers_version`, `_name_or_path` (the trainer scrubs it)
  and `dtype`/`torch_dtype` (the engine is always bf16). The input model's config failing to resolve is an
  evaluation error; the submission's is a scored fail;
- after loading, `quant_vllm` reads every vLLM linear layer through `LLM.apply_model`. A layer left
  unquantised (`UnquantizedLinearMethod`), a scheme whose bits, group size, symmetry or activation
  differ from the spec, or a kernel outside `QUANT_EXPECTED_KERNEL_CLASSES` fails the check.
  `FusedMoE` experts are not `LinearBase`, so this covers dense bases only (all bases in the pool are).

The config check catches a mislabelled checkpoint before any GPU time; the layer check catches a config that
describes one thing while the weights load as another.

A config mismatch (including a config.json that is missing, malformed or not a JSON object), load
error, failed layer check or non-finite loss is a scored fail (`eval_loss` null plus
`quant_failure`), not an evaluation error; failing to reach the hub for config.json is an evaluation
error. Humming (W3A16, W2A16) is made deterministic by
`disable_humming_stream_k`, which turns off its bf16 atomicAdd stream-K reduction before any engine
is built; it patches a private function of humming-kernels 0.1.16, and a missing target is an
evaluation error. `quant.py` and `single_quant.py` run in the quant image, which has no validator
stack: they import neither `common.py` (axolotl/peft) nor `core.logging` (fiber).

Fixed-track repos are loaded straight from HF, and only their weights and config.json reach the engine: the
tokenizer is the input model's (quant prep tokenised the test set with it and prompts go in as token ids, so a
miner's tokenizer files, chat template or special tokens are never read), the config is parsed in HF format only
(a `params.json` plus `consolidated*.safetensors` would otherwise switch vLLM to its Mistral parser and loader,
past the config check and the structural check's shards), weights load as safetensors only, and
`generation_config.json` supplies no sampling defaults (`QUANT_VLLM_*` in `constants.py`). Freeform repos are never loaded from the miner's
artifact, so miner code never runs at scoring time and the scorer only ever sees plain bf16 tensors under the base
config and tokenizer: the trainer decode job uploads bf16 safetensors plus `index.json` (written last) under
`quant-decoded/<task_id>/<hotkey>/` in MinIO, `decoded_model_urls_for` presigns each complete
prefix, and each deployment gets only its own repo's urls in `DECODED_MODEL_URLS`. The evaluator
rebuilds the model dir from the base model's config/tokenizer plus the shards the index lists,
checked by byte count and sha256; a mismatch is a scored fail, a failed download (our object storage, an
expired url) an evaluation error. A freeform repo with no complete prefix is not sent to a GPU and is an
evaluation error. `artifact_bytes` is
measured on the validator from the HF repo (every file counts), never taken from the miner.

Image-config requirement: decoded models are materialised under `TMPDIR` inside the quant
evaluator image (`tempfile.mkdtemp`), and an 8B bf16 model is ~16 GB. `TMPDIR` must point at a
disk-backed path; a tmpfs `/tmp` would hold the model in RAM.

## Whose fault an evaluation failure is

Only a verdict on the repo is ever scored against the miner. Everything else is an evaluation error, which is re-queued
and never scored (`errors.py`):

- **The miner's (scored).** A result model that says so (`quant_failure`, a non-finite loss, `is_finetune=False`),
  the evaluator's own per-repo error string (a text/image evaluator could not load or run the repo:
  `RepoEvaluationFailed`, kept as `Evaluation failed: <Type>: <message>`), and the hub answering that the repo does
  not exist (`RepositoryNotFoundError` from `repo_info`).
- **Infrastructure or ours (`EvaluationInfraError`, re-queued).** Anything else: a deployment that fails, dies, stops
  answering or times out, attempts exhausted on either backend, an exception escaping the backend runner, a results
  file without the repo's entry (the evaluator subprocess died), an evaluator's `{"evaluation_error": ...}` entry, a
  quant evaluator error string (the quant evaluator turns every submission fault into `quant_failure`, so whatever
  escapes it is ours), the hub/S3/DB failing on the validator (repo_info, structural pre-check, artifact bytes,
  decoded-model listing), a missing decoded model or reference model on a task-common task, a missing expected repo
  name, and any exception out of the evaluation call. `is_infra_error` (transport errors, timeouts including
  empty-message ones, HTTP 5xx/408/429, through the exception's cause/context chain) labels the alert
  ("infrastructure error" vs "evaluation error (likely ours)") and lets an evaluator mark a repo's hub/network failure
  as `{"evaluation_error": "<Type>: <message>"}` instead of a verdict; both kinds are re-queued, counted per row and,
  at `MAX_EVAL_INFRA_ATTEMPTS`, held (see Lifecycle below), never scored. The quant evaluator uses it on the config.json
  read as well: a transport error there is an evaluation error, a malformed or oversized file a scored fail.

Lifecycle (`validator/lifecycle/eval_infra_errors.py`) puts the rows back to `pending`, counts each in
`evaluations.infra_error_attempts` and alerts on the first one. At `MAX_EVAL_INFRA_ATTEMPTS` the rows are held in
`evaluation_status='infra_held'`: never dispatched, never scored, and the task does not finalise while one is held. The
alert carries the SQL to re-queue them once the cause is fixed. `EvaluationCapacityUnavailable` and
`DeploymentNotReadyError` are the GPU queue, not errors: re-queued, never counted; any other `EvaluationRetryableError`
(a deployment whose deletion cannot be verified) counts like an `EvaluationInfraError`. Every failure message carries the
exception type (`describe_error`), so an empty-message exception is never reported blank.

## Quant eval GPU SKUs

Rule: a quant eval runs on the H100 family, either SKU, with nothing forced either way and no per-task pin, so a
task never waits for one SKU. `QUANT_EVAL_ALLOWED_SKUS` (`validator/evaluation/constants.py`) lists them as
`QuantEvalSku` entries:

| Label | torch device name | SMs | Runpod GPU type |
| --- | --- | --- | --- |
| H100 SXM | `NVIDIA H100 80GB HBM3` | 132 | `NVIDIA H100 80GB HBM3` |
| H100 PCIe | `NVIDIA H100 PCIe` | 114 | `NVIDIA H100 PCIe` |

- The dispatcher asks for `QUANT_EVAL_GPU_MODELS` (`H100`) and narrows Runpod offers to the listed GPU types
  (`QUANT_EVAL_RUNPOD_INSTANCE_TYPES`). It passes no GPU env: the rule lives in the evaluator's own constants.
- Before loading any model, `evaluators/quant.check_gpu` requires the device name and the SM count to match one
  entry. Anything else (A100, L40S, H100 NVL, a MIG slice) raises `QuantGpuMismatch`: the container exits
  non-zero, an evaluation error that is retried, never a scored fail. It fails closed: an unknown SKU is refused.
- `QUANT_EVAL_ALLOW_ANY_GPU=1` (test boxes only; any other value, including empty, is off) admits any GPU with a
  `!!! ... OVERRIDE` warning in the container log; the result carries `eval_gpu.override=true` and `sku=null`, and
  the dispatcher logs it as an error.
- Every result carries `eval_gpu` (`QuantEvalGpu`: `name`, `sm_count`, `sku`, `override`), persisted to
  `task_nodes.eval_gpu`; every bf16 reference records its own in `quant_sku_references.eval_gpu`.
- A bare bf16 reference measurement (the pre-pass in `validator/tournament/quant_reference_prepass.py`: own-model,
  task-common and calibration references) may also run on an 80 GB A100. The dispatcher passes
  `reference_only=True`, which asks for `QUANT_REFERENCE_GPU_MODELS` (`H100`, `A100`) narrowed to
  `QUANT_REFERENCE_RUNPOD_INSTANCE_TYPES` and sets `QUANT_REFERENCE_ONLY=1`; the evaluator then accepts
  `QUANT_REFERENCE_ALLOWED_SKUS` (the H100 SKUs plus `A100 SXM` / `A100 PCIe`, 108 SMs, 80 GB only). It refuses the
  flag (`QuantReferenceOnlyMisuse`, container exits non-zero) unless `MODELS` is exactly `QUANT_REFERENCE_MODEL` with
  no `QUANT_SPEC` and no `DECODED_MODEL_URLS`, so an entry is never scored off the H100 family.
  Why it is safe: only quant kernels differ by GPU class; a bf16 model scores within the low decimals anywhere. An
  A100 reference is stored under its own SKU label and every score divides by the reference on its entry's own SKU
  (`reference_on`), so an A100 row never becomes the denominator of an H100 score. It only sets the target published
  to the miner (`QuantTarget.reference_sku` says which GPU); the scoring deployment measures its own SKU's reference
  in place when none is cached. Why it matters: the pre-pass gates own-model training, and Runpod H100 capacity can
  vanish for hours while A100s are available.

Where the SKU reaches the numbers (vLLM 0.31.0, humming-kernels 0.1.16), so what differs between 114 and 132 SMs:

- Humming (W3A16 and W2A16): `humming.tune` sizes its SM90 tiles, row/token partitioning and grid from `device.sm_count`
  (`tune/process_input.py`, `tune/sm90_policies.py`). Stream-K, whose bf16 partial-sum reduction is
  order-dependent, is off (`quant_vllm.disable_humming_stream_k`), but tile and partition choices still depend on the
  SM count, and its JIT cache is keyed by SKU. Expect agreement to roughly the third decimal, not bit-exactness.
- Machete (W4A16): every default schedule uses CUTLASS's `StreamKScheduler`, whose data-parallel / split / stream-K
  decomposition is chosen from the SM count at launch, so the fp32 reduction order over K can differ by SKU.
- CutlassW4A8 (W4A8fp8): `KernelTmaWarpSpecializedCooperative` with the default persistent data-parallel scheduler;
  each output tile reduces its whole K in one CTA, so the SKU changes scheduling, not the arithmetic.
- The bf16 parts (attention, `lm_head`, a freeform decoded model): FlashAttention's split-KV heuristic and cuBLAS
  algorithm selection both read the SM count, so even the bf16 reference is not bit-identical across SKUs.
- The image has no SM target: no `TORCH_CUDA_ARCH_LIST`, no pre-warmed kernels, vLLM's wheel ships sm_90 kernels for
  both, Humming JITs per SKU through NVRTC. Nothing is hard-coded to 132 SMs.

### Same-SKU bf16 reference

Scoring compares quant losses only as multiples of a bf16 reference scored on the same SKU
(`validator/scoring/README.md`, "Quant SKU normalisation"). The reference is measured inside the evaluation
deployments themselves, so a submission and its reference always share a pod:

- The dispatcher (`run_evaluation_basilica_quant`, given `reference_models` and `cached_references` keyed by repo)
  sets, per repo's deployment, `QUANT_REFERENCE_MODEL` and `ORIGINAL_MODEL` (that repo's bf16 reference: its miner's
  own model on an own-model task, else the common input) and
  `QUANT_CACHED_REFERENCES` (JSON `{reference_key: loss}` of the SKUs already cached for that reference). The
  in-container evaluator is per deployment and reads these single values unchanged.
- After `check_gpu`, the evaluator scores the reference first when its own `eval_gpu.reference_key` is not in the map,
  through the same `single_quant` path, test set and engine as the submissions but with no spec (bf16 is never
  config- or kernel-checked) and never from a decoded model. The result goes under `QUANT_REFERENCE_RESULT_KEY`
  (`quant_reference`) in the results file. A cached SKU costs nothing. When the reference is itself one of the
  repos, it is scored once and its result copied under that key.
- The dispatcher pops that entry and attaches it as the repo's `EvaluationResultText.quant_reference`
  (`QuantReferenceScore`). A reference that errored, scored a fail or came back without a vector, fingerprint or GPU
  is logged as an error and dropped; the submission's own result stands, and finalize re-evaluates the row while its
  SKU still has no reference.
- Cost: one extra bf16 eval per deployment that starts before its SKU is cached. A deployment cannot know its SKU
  before it lands, and a task's per-hotkey evaluations dispatch together, so on a fresh task every first-wave
  deployment scores the reference; later ones (retries, late hotkeys) find it cached. The duplicates are the
  cross-check: the first writer is kept and any later measurement must match it exactly.
