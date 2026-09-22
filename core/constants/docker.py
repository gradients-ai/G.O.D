VALIDATOR_DOCKER_IMAGE = "gradientsio/text-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_DIFFUSION = "gradientsio/image-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_ENV = "gradientsio/env-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_INTERCODE = "gradientsio/env-eval-intercode:basilica"
VALIDATOR_DOCKER_IMAGE_SWE_INFINITE = "gradientsio/env-eval-swe-infinite:basilica"
VALIDATOR_DOCKER_IMAGE_PVP = "gradientsio/pvp-evaluator:basilica"
MCTS_API_DOCKER_IMAGE = "gradientsio/mcts-api:latest"
# Shared SGLang runtime for PvP / env / InterCode / SWE eval. CUDA 12.9 image so
# --enable-deterministic-inference is available (added in 0.5.3; no official cu126).
SGLANG_EVAL_IMAGE = "lmsysorg/sglang:v0.5.3-cu129"

# Env vars used to signal KL-regularized instruct training to miner containers and evaluators.
USE_KL_ENV = "USE_KL"
KL_COEF_ENV = "KL_COEF"
# Set only for boss-round tasks: makes the instruct/DPO evaluators emit a per-example loss vector
# for the paired comparison. Off elsewhere so ordinary tasks do not pay the extra forward passes.
EMIT_PER_EXAMPLE_LOSSES_ENV = "EMIT_PER_EXAMPLE_LOSSES"

# Immutable lineage seed for every continuous-SFT lineage; eval pins tokenizer + chat template here,
# not original_model (the carried winner).
CONTINUOUS_SFT_TOKENIZER_REPO_ENV = "CONTINUOUS_SFT_TOKENIZER_REPO"
