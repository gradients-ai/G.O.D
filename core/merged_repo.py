"""Naming of the published flat merge of a LoRA adapter repo."""

import hashlib


def merged_repo_name(model_id: str, hf_username: str) -> str:
    """Opaque, deterministic repo name for a published LoRA-merge; namespaced apart from augmented-*."""
    repo_hash = hashlib.sha256(f"{model_id}:lora-merge".encode()).hexdigest()[:16]
    return f"{hf_username}/merged-{repo_hash}"
