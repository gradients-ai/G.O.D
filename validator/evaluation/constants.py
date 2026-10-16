from core.constants.datasets import GRPO_DEFAULT_FIELD_PROMPT


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

EVAL_BASILICA_CPU = "4"
EVAL_BASILICA_MEMORY = "64Gi"
EVAL_BASILICA_TTL_SECONDS = 28800
EVAL_BASILICA_TIMEOUT = 14400
EVAL_BASILICA_DEPLOYMENT_NAME_PREFIX = "god-eval-"
EVAL_BASILICA_CREATE_SPACING_SECONDS = 5
EVAL_BASILICA_MAX_RETRIES = 1
EVAL_BASILICA_RETRY_DELAY_SECONDS = 0
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
