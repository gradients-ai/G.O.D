import os

from dotenv import load_dotenv


load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_TOKEN = os.getenv("WANDB_TOKEN")

HUGGINGFACE_USERNAME = os.getenv("HUGGINGFACE_USERNAME")
RAYONLABS_HF_USERNAME = "gradients-io-tournaments"

YARN_HUGGINGFACE_USERNAME = os.getenv("YARN_HUGGINGFACE_USERNAME", "gradients-io")
YARN_HUGGINGFACE_TOKEN = os.getenv("YARN_HUGGINGFACE_TOKEN")
