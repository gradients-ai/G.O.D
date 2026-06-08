VALIDATOR_DOCKER_IMAGE = "gradientsio/text-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_DIFFUSION = "gradientsio/image-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_ENV = "gradientsio/env-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_INTERCODE = "gradientsio/env-eval-intercode:basilica"
VALIDATOR_DOCKER_IMAGE_PVP = "gradientsio/pvp-evaluator:basilica"
MCTS_API_DOCKER_IMAGE = "gradientsio/mcts-api:latest"

# Env vars used to signal KL-regularized instruct training to miner containers and evaluators.
USE_KL_ENV = "USE_KL"
KL_COEF_ENV = "KL_COEF"
