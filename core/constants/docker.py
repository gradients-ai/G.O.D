VALIDATOR_DOCKER_IMAGE = "gradientsio/text-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_DIFFUSION = "gradientsio/image-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_ENV = "gradientsio/env-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_INTERCODE = "gradientsio/env-eval-intercode:basilica"
VALIDATOR_DOCKER_IMAGE_PVP = "gradientsio/pvp-evaluator:basilica"
MCTS_API_DOCKER_IMAGE = "gradientsio/mcts-api:latest"

# Env vars used to signal KL-regularized instruct training to miner containers and evaluators.
USE_KL_ENV = "USE_KL"
KL_COEF_ENV = "KL_COEF"

# Set ONLY for continuous-SFT lineages whose model ships custom architecture code (e.g. quasar):
# value is the audited seed mirror. Its presence is the per-lineage gate that turns on
# trust_remote_code, and the evaluator pins the custom *.py to this repo so a miner-supplied repo's
# code is never executed (RCE guard). Absent for standard-arch lineages (e.g. qwen) → no remote code.
CONTINUOUS_SFT_REMOTE_CODE_REPO_ENV = "CONTINUOUS_SFT_REMOTE_CODE_REPO"

# Set for every continuous-SFT lineage: the immutable lineage seed. Eval pins the tokenizer + chat
# template to this, not original_model (the carried winner). Absent for non-continuous tasks.
CONTINUOUS_SFT_TOKENIZER_REPO_ENV = "CONTINUOUS_SFT_TOKENIZER_REPO"
