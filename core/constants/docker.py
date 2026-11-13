VALIDATOR_DOCKER_IMAGE = "gradientsio/text-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_DIFFUSION = "gradientsio/image-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_QUANT = "gradientsio/quant-evaluator:basilica"

# Env vars used to signal KL-regularized instruct training to miner containers and evaluators.
USE_KL_ENV = "USE_KL"
KL_COEF_ENV = "KL_COEF"
# Set only for boss-round tasks: makes the instruct/DPO evaluators emit a per-example loss vector
# for the paired comparison. Off elsewhere so ordinary tasks do not pay the extra forward passes.
EMIT_PER_EXAMPLE_LOSSES_ENV = "EMIT_PER_EXAMPLE_LOSSES"

# Immutable lineage seed for every continuous-SFT lineage; eval pins tokenizer + chat template here,
# not original_model (the carried winner).
CONTINUOUS_SFT_TOKENIZER_REPO_ENV = "CONTINUOUS_SFT_TOKENIZER_REPO"

# Quant evaluator inputs. QUANT_SPEC is the JSON of the task's QuantFixedSpec and is absent for the
# freeform track; QUANT_FORMAT is the QuantFormat value and selects the expected kernel set.
QUANT_SPEC_ENV = "QUANT_SPEC"
# The JSON of the miner's QuantTarget (its bf16 reference loss, and for freeform the bar); set on every compression job.
QUANT_TARGET_ENV = "QUANT_TARGET"
# Path (in the cache volume mount) of the calibration-set JSON (core.models.quant_models.QuantCalibrationSet: the examples
# with the bf16 reference's per-example losses); set on every compression job.
QUANT_CALIBRATION_PATH_ENV = "QUANT_CALIBRATION_PATH"
QUANT_FORMAT_ENV = "QUANT_FORMAT"
# The bf16 reference every quant loss on the task is normalised by (the miner's own model on an own-model task,
# else the common input), and the JSON {reference_key: loss} of the SKUs whose reference is already cached. The
# evaluator scores the reference in the same deployment only when its own SKU is missing from the map.
QUANT_REFERENCE_MODEL_ENV = "QUANT_REFERENCE_MODEL"
QUANT_CACHED_REFERENCES_ENV = "QUANT_CACHED_REFERENCES"
# Set by the dispatcher for a bare bf16 reference measurement (the pre-pass): "1" lets the evaluator run on
# QUANT_REFERENCE_ALLOWED_SKUS (adds 80 GB A100s). The evaluator refuses it unless MODELS is exactly the reference
# model with no QUANT_SPEC and no DECODED_MODEL_URLS, so an entry can never be scored outside QUANT_EVAL_ALLOWED_SKUS.
QUANT_REFERENCE_ONLY_ENV = "QUANT_REFERENCE_ONLY"
# Test boxes only, never set by the dispatcher: "1" lets the evaluator score on a GPU outside
# QUANT_EVAL_ALLOWED_SKUS (logged loudly, recorded as eval_gpu.override). Unset, any other GPU fails the container.
QUANT_EVAL_ALLOW_ANY_GPU_ENV = "QUANT_EVAL_ALLOW_ANY_GPU"
# Freeform track only: JSON {repo: {filename: presigned GET url}} for the decoded bf16 model the
# trainer uploaded; the evaluator scores those weights instead of the miner's HF artifact.
DECODED_MODEL_URLS_ENV = "DECODED_MODEL_URLS"
# Written last by the decode job, so its presence marks a complete upload.
DECODED_MODEL_INDEX_NAME = "index.json"

# A text entrant's training Dockerfile, first match wins. The trainer builds it per job and the validator's
# registration lint (validator/tournament/registration_lint.py) checks it, so both read this one tuple.
DEFAULT_TEXT_DOCKERFILE_PATH = "ops/docker/standalone-text-trainer.dockerfile"
LEGACY_TEXT_DOCKERFILE_PATH = "dockerfiles/standalone-text-trainer.dockerfile"
TEXT_DOCKERFILE_PATHS = (DEFAULT_TEXT_DOCKERFILE_PATH, LEGACY_TEXT_DOCKERFILE_PATH)
