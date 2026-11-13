from core.constants.datasets import GRPO_DEFAULT_FIELD_PROMPT
from core.models.quant_models import QuantActivationFormat
from core.models.quant_models import QuantEvalSku
from core.models.quant_models import QuantFormat


DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT = 0.5


DOCKER_EVAL_HF_CACHE_DIR = "/root/.cache/huggingface"

# DPO evaluation
TRL_DPO_FIELD_PROMPT = "prompt"
TRL_DPO_FIELD_CHOSEN = "chosen"
TRL_DPO_FIELD_REJECTED = "rejected"

# GRPO evaluation
TRL_GRPO_FIELD_PROMPT = GRPO_DEFAULT_FIELD_PROMPT

# Default, fixed Hyperparameters
BETA_DPO = 0.1
BETA_GRPO = 0.5

# GRPO evaluation
GRPO_INITIAL_BATCH_SIZE = 16
GRPO_KL_BATCH_SIZE = 1
# Per-example loss extraction must run one example per forward. The reported eval_loss is produced
# at eval batch size 1, so this is what makes mean(per_example_losses) == eval_loss exact rather
# than an approximation, and it avoids any dependence on padding being neutral across a batch.
# Do not raise this.
PER_EXAMPLE_LOSS_BATCH_SIZE = 1
# Sequence-position chunk size for the per-example CE pass. Keeps the transient fp32
# logits×vocab footprint bounded on long 70B examples instead of materialising the full
# shifted logits tensor in float32 at once.
PER_EXAMPLE_LOSS_CE_CHUNK_SIZE = 512
GRPO_DEFAULT_NUM_GENERATIONS = 2
GRPO_KL_SEQUENCE_LENGTH = 512

BASILICA_GPU_MODELS = ["A100"]
BASILICA_SGLANG_MIN_GPU_MEMORY_GB = 80

# Quant evaluation (vLLM prompt-logprob scorer). vLLM picks the kernel per GPU class (Machete and
# CutlassW4A8 on H100, Marlin on A100), so quant evals run on the H100 family only: SXM and PCIe are
# both allowed, and neither is preferred, so a task never waits for one SKU. Each submission is scored
# in its own deployment. Within the family, Humming's W3 tiling and Machete's stream-K decomposition
# read the SM count, so the two SKUs can differ in the low decimals; every result records its SKU
# (eval_gpu). The evaluator reads its GPU before loading any model and refuses (an evaluation error,
# never a scored fail) one whose device name and SM count match no entry here, unless
# QUANT_EVAL_ALLOW_ANY_GPU is set (test boxes only; logged loudly and recorded as override).
QUANT_EVAL_GPU_MODELS = ["H100"]
QUANT_EVAL_ALLOWED_SKUS = (
    QuantEvalSku(label="H100 SXM", device_name="NVIDIA H100 80GB HBM3", sm_count=132, runpod_gpu_type="NVIDIA H100 80GB HBM3"),
    QuantEvalSku(label="H100 PCIe", device_name="NVIDIA H100 PCIe", sm_count=114, runpod_gpu_type="NVIDIA H100 PCIe"),
)
# dstack's "H100" names the GPU class, not a SKU (H100 NVL is not listed above), so Runpod offers are
# narrowed by instance type (the Runpod GPU type id) to the allowed SKUs.
QUANT_EVAL_RUNPOD_INSTANCE_TYPES = [sku.runpod_gpu_type for sku in QUANT_EVAL_ALLOWED_SKUS]
# A bf16 reference measured on its own (the pre-pass: own-model, task-common and calibration references) may also
# run on an 80 GB A100. Only quant kernels differ by GPU class; a bf16 model scores within the low decimals anywhere.
# Such a reference is stored under its own SKU label, and every score is normalised by the reference on the SKU its
# entry was scored on (reference_on), so an A100 row is never a denominator for an H100 entry: it only sets the
# target published to the miner, and the scoring deployment measures its own SKU's reference in place.
# Entries themselves stay on QUANT_EVAL_ALLOWED_SKUS.
QUANT_REFERENCE_EXTRA_SKUS = (
    QuantEvalSku(label="A100 SXM", device_name="NVIDIA A100-SXM4-80GB", sm_count=108, runpod_gpu_type="NVIDIA A100-SXM4-80GB"),
    QuantEvalSku(label="A100 PCIe", device_name="NVIDIA A100 80GB PCIe", sm_count=108, runpod_gpu_type="NVIDIA A100 80GB PCIe"),
)
QUANT_REFERENCE_ALLOWED_SKUS = QUANT_EVAL_ALLOWED_SKUS + QUANT_REFERENCE_EXTRA_SKUS
QUANT_REFERENCE_GPU_MODELS = ["H100", "A100"]
QUANT_REFERENCE_RUNPOD_INSTANCE_TYPES = [sku.runpod_gpu_type for sku in QUANT_REFERENCE_ALLOWED_SKUS]
QUANT_VLLM_MAX_NUM_SEQS = 1
QUANT_VLLM_LOGPROBS_MODE = "raw_logprobs"
QUANT_VLLM_SEED = 0
# What the quant engine reads besides the repo's weights and config.json (see quant_vllm): the HF tokenizer, as
# quant prep used; HF-format config only, never params.json; safetensors only, so consolidated*.safetensors cannot
# switch the loader to Mistral format; and no sampling defaults from the repo's generation_config.json.
QUANT_VLLM_TOKENIZER_MODE = "hf"
QUANT_VLLM_CONFIG_FORMAT = "hf"
QUANT_VLLM_LOAD_FORMAT = "safetensors"
QUANT_VLLM_GENERATION_CONFIG = "vllm"
# Set in-process before vLLM is imported. The engine stays in this process (so apply_model needs no
# pickling), and FlashInfer's top-k/top-p sampler is off: it JIT-compiles with nvcc, which the image lacks.
QUANT_VLLM_ENV = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}
# Results-file key the quant evaluator writes the same-deployment bf16 reference under, next to the repo's result.
QUANT_REFERENCE_RESULT_KEY = "quant_reference"
# Reported quant_failure reasons are cut to this length.
QUANT_FAILURE_REASON_MAX_CHARS = 350
# A repo's config.json over this size is not parsed (an honest one is a few KB; a huge one is a memory and disk
# attack on the shared eval container, so it is a scored fail before json.loads sees it).
QUANT_CONFIG_MAX_BYTES = 1 << 20
# vLLM 0.31.0 kernel classes each fixed-track format must load through on H100.
QUANT_EXPECTED_KERNEL_CLASSES: dict[QuantFormat, frozenset[str]] = {
    QuantFormat.W4A16_SYM_G128: frozenset({"MacheteLinearKernel"}),
    QuantFormat.W4A8_FP8_G128: frozenset({"CutlassW4A8LinearKernel"}),
    QuantFormat.W3A16_ASYM_G128: frozenset({"HummingLinearKernel"}),
    QuantFormat.W2A16_ASYM_G64: frozenset({"HummingLinearKernel"}),
}
# vLLM 0.31.0 compressed-tensors scheme classes and the activation format each one runs. The kernel
# class alone does not pin it: a W4A8-int8 checkpoint routes to Machete like W4A16.
QUANT_SCHEME_ACTIVATIONS: dict[str, QuantActivationFormat] = {
    "CompressedTensorsWNA16": QuantActivationFormat.BF16,
    "CompressedTensorsW4A8Fp8": QuantActivationFormat.FP8,
    "CompressedTensorsWNA8Int": QuantActivationFormat.INT8,
}

EVAL_BASILICA_CPU = "4"
EVAL_BASILICA_MEMORY = "64Gi"
EVAL_BASILICA_TTL_SECONDS = 28800
EVAL_BASILICA_TIMEOUT = 14400
EVAL_BASILICA_DEPLOYMENT_NAME_PREFIX = "god-eval-"
EVAL_BASILICA_CREATE_SPACING_SECONDS = 5
EVAL_BASILICA_MAX_RETRIES = 3
EVAL_BASILICA_RETRY_DELAY_SECONDS = 30
EVAL_BASILICA_POLL_INTERVAL_SECONDS = 300
EVAL_BASILICA_MAX_POLL_SECONDS = 28800
EVAL_BASILICA_MAX_CONSECUTIVE_POLL_FAILURES = 5
EVAL_BASILICA_FAILED_POLL_RECHECK_SECONDS = 30
EVAL_DEPLOYMENT_READY_TIMEOUT_SECONDS = 1800
EVAL_RESUME_HEALTH_GRACE_SECONDS = 60
EVAL_DB_MAX_CONCURRENT_WRITES = 2
EVAL_DB_RETRY_ATTEMPTS = 4
EVAL_DB_RETRY_BASE_DELAY_SECONDS = 1.0

# HuggingFace container env vars (shared across all eval containers)
_HF_CONTAINER_ENV_BASE = {
    "HF_HOME": "/root/.cache/huggingface",
    "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
    "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
    "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
    # Avoid CUDA allocator fragmentation OOMs on long-sequence 70B multi-GPU evals when
    # materialising per-example logits (boss-round paired loss vectors).
    # PyTorch 2.9+ prefers PYTORCH_ALLOC_CONF; keep the legacy name for older images.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
}


def _hf_auth_env_from_host() -> dict[str, str]:
    """Pass HF_READ_ONLY_TOKEN into Basilica/eval containers.

    huggingface_hub authenticates via HF_TOKEN (or HUGGING_FACE_HUB_TOKEN). Our
    evaluators also read HUGGINGFACE_TOKEN. Only the read-only token is used.
    """
    import os

    token = (os.environ.get("HF_READ_ONLY_TOKEN") or "").strip()
    if not token:
        return {}
    return {
        "HF_TOKEN": token,
        "HUGGING_FACE_HUB_TOKEN": token,
        "HUGGINGFACE_TOKEN": token,
    }


def hf_container_env(*, enable_hf_transfer: bool = True) -> dict[str, str]:
    return {
        **_HF_CONTAINER_ENV_BASE,
        "HF_HUB_ENABLE_HF_TRANSFER": "1" if enable_hf_transfer else "0",
        **_hf_auth_env_from_host(),
    }


def __getattr__(name: str):
    # Resolve at access time so .vali.env tokens loaded after import are picked up.
    if name == "HF_CONTAINER_ENV":
        return hf_container_env(enable_hf_transfer=True)
    if name == "HF_CONTAINER_ENV_IMAGE":
        return hf_container_env(enable_hf_transfer=False)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Eval-deployment reconciler: a live Basilica deployment with no backing eval row, or an
# active eval row whose deployment is gone, must be older/staler than this grace window
# before the reconciler acts on it. Protects the brief reserve -> deploy -> persist window
# (and any list() staleness) so an in-flight eval is never reaped mid-startup.
EVAL_ORPHAN_GRACE_SECONDS = 7200

# Grace for RELEASING a ghost reservation whose deployment is provably absent from a fresh Basilica
# list(). This is the opposite direction from orphan reaping: a reservation only becomes a ghost
# once its deployment_id was stamped post-readiness (Basilica already knew it), so a still-booting
# deployment (deployment_id NULL) can never be false-positived here — it is covered by the long
# orphan grace instead. So this can be short: release a dead deployment's GPUs in ~2 reconcile
# cycles rather than making the cap wait the full 30 min.
EVAL_GHOST_GRACE_SECONDS = 240

EVAL_RUNPOD_ORPHAN_GRACE_SECONDS = 2100
EVAL_RUNPOD_MAX_DURATION_SECONDS = 7200
