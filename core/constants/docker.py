VALIDATOR_DOCKER_IMAGE = "gradientsio/text-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_DIFFUSION = "gradientsio/image-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_ENV = "gradientsio/env-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_INTERCODE = "gradientsio/env-eval-intercode:basilica"
VALIDATOR_DOCKER_IMAGE_PVP = "gradientsio/pvp-evaluator:basilica"
MCTS_API_DOCKER_IMAGE = "gradientsio/mcts-api:latest"

# Env vars used to signal KL-regularized instruct training to miner containers and evaluators.
USE_KL_ENV = "USE_KL"
KL_COEF_ENV = "KL_COEF"

# Audited seed mirror for custom-arch continuous-SFT lineages (e.g. quasar). Its presence gates
# trust_remote_code; eval pins the custom *.py to this repo so miner code never runs (RCE guard).
CONTINUOUS_SFT_REMOTE_CODE_REPO_ENV = "CONTINUOUS_SFT_REMOTE_CODE_REPO"

# Immutable lineage seed for every continuous-SFT lineage; eval pins tokenizer + chat template here,
# not original_model (the carried winner).
CONTINUOUS_SFT_TOKENIZER_REPO_ENV = "CONTINUOUS_SFT_TOKENIZER_REPO"
