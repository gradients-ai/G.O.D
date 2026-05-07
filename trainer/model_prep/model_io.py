import hashlib
import json
import os
import tempfile

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def generate_anonymous_repo_name(model_id: str, seed: int) -> str:
    """Generate an opaque repo name that doesn't leak the original model identity."""
    hf_username = os.environ.get("HUGGINGFACE_USERNAME", "gradients-io")
    hash_input = f"{model_id}:{seed}"
    repo_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    return f"{hf_username}/augmented-{repo_hash}"


def load_model_and_tokenizer(model_id: str, hf_token: str):
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Multi-GPU detected ({n_gpus}), using device_map=auto", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, token=hf_token, device_map="auto",
        )
    elif torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, token=hf_token,
        )
        model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    return model, tokenizer


def upload_augmented_model(model, tokenizer, repo_id: str, hf_token: str) -> None:
    """Upload prepared model to HuggingFace, scrubbing identity."""
    print(f"Uploading augmented model to {repo_id}")

    model.config._name_or_path = repo_id
    model.push_to_hub(repo_id, token=hf_token, private=False)
    tokenizer.push_to_hub(repo_id, token=hf_token, private=False)

    api = HfApi(token=hf_token)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = api.hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=tmp, token=hf_token)
        with open(config_path, "r") as f:
            config = json.load(f)
        if "_name_or_path" in config:
            del config["_name_or_path"]
            modified_path = os.path.join(tmp, "config_clean.json")
            with open(modified_path, "w") as f:
                json.dump(config, f, indent=2)
            api.upload_file(
                path_or_fileobj=modified_path,
                path_in_repo="config.json",
                repo_id=repo_id,
            )

    print(f"Upload complete: {repo_id}")
