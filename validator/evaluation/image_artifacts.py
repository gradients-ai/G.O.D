"""Pinned model artifact loading and atomic evaluation results."""

import hashlib
import json
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def select_lora(files):
    files = sorted(f for f in files if f.startswith(("checkpoints/", "checkpoint/")) and f.endswith(".safetensors"))
    lasts = [f for f in files if Path(f).name == "last.safetensors"]
    if len(lasts) == 1:
        return lasts[0]
    numbered = [(int(m.group(1)), f) for f in files if (m := re.search(r"[-_](\d+)\.safetensors$", f))]
    if numbered:
        return max(numbered)[1]
    if len(files) == 1:
        return files[0]
    raise ValueError("No unambiguous LoRA checkpoint")


def materialize_model(api, repo, filename, directory):
    from huggingface_hub import hf_hub_download

    info = api.model_info(repo, files_metadata=True)
    revision = info.sha
    if filename is None:
        files = api.list_repo_files(repo, revision=revision)
        filename = select_lora(files)
    name = hashlib.sha256(f"{repo}@{revision}/{filename}".encode()).hexdigest()[:24] + ".safetensors"
    target = directory / name
    directory.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        existing = directory / Path(filename).name
        metadata = next((f for f in info.siblings if f.rfilename == filename), None)
        if existing.exists() and metadata and metadata.lfs and sha256_file(existing) == metadata.lfs.sha256:
            source = str(existing)
        else:
            source = hf_hub_download(repo, filename, revision=revision)
        target.symlink_to(Path(source).resolve())
    return name, {"repo": repo, "revision": revision, "filename": filename}


def prepare_base(api, task, root):
    model_type = task["model_type"]
    if model_type in ("ideogram4", "krea2"):
        repo, filename = (
            ("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_fp8_scaled.safetensors")
            if model_type == "ideogram4"
            else ("Comfy-Org/Krea-2", "diffusion_models/krea2_raw_fp8_scaled.safetensors")
        )
    else:
        repo = task["model_id"]
        info = api.model_info(repo, files_metadata=True)
        candidates = [
            f for f in info.siblings
            if f.rfilename.endswith(".safetensors") and "/" not in f.rfilename and (f.size or 0) > 5 * 1024**3
        ]
        if len(candidates) != 1:
            raise ValueError("Base model requires one unambiguous root-level ComfyUI checkpoint")
        filename = candidates[0].rfilename
    folder = "unet" if model_type == "flux" else "diffusion_models"
    return materialize_model(api, repo, filename, root / "models" / folder)
