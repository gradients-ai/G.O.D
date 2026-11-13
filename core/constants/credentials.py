import os


BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_TOKEN = os.getenv("WANDB_TOKEN")

HUGGINGFACE_USERNAME = os.getenv("HUGGINGFACE_USERNAME")
# HF org the validator reads tournament uploads from (<org>/<expected_repo_name>); the trainer's HUGGINGFACE_USERNAME
# must be the same org. Defaults to the prod org; a test box points both at a scratch org.
RAYONLABS_HF_USERNAME = os.getenv("TOURNAMENT_HF_USERNAME", "gradients-io-tournaments")

YARN_HUGGINGFACE_USERNAME = os.getenv("YARN_HUGGINGFACE_USERNAME", "gradients-io")
YARN_HUGGINGFACE_TOKEN = os.getenv("YARN_HUGGINGFACE_TOKEN")
