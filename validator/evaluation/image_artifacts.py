"""Pinned model artifact loading and atomic evaluation results."""

import hashlib
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _fmt_bytes(size: int | None) -> str:
    if not size:
        return "unknown size"
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GiB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MiB"
    return f"{size} B"


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

    kind = "lora" if filename is None else "artifact"
    logger.info("resolving %s repo=%s filename=%s dest=%s", kind, repo, filename or "(auto)", directory)
    info = api.model_info(repo, files_metadata=True)
    revision = info.sha
    if filename is None:
        files = api.list_repo_files(repo, revision=revision)
        filename = select_lora(files)
        logger.info("selected lora checkpoint repo=%s file=%s revision=%s", repo, filename, revision[:12])
    metadata = next((f for f in info.siblings if f.rfilename == filename), None)
    size = getattr(metadata, "size", None)
    name = hashlib.sha256(f"{repo}@{revision}/{filename}".encode()).hexdigest()[:24] + ".safetensors"
    target = directory / name
    directory.mkdir(parents=True, exist_ok=True)
    if target.exists():
        logger.info("using cached %s repo=%s file=%s size=%s", kind, repo, filename, _fmt_bytes(size))
    else:
        existing = directory / Path(filename).name
        if existing.exists() and metadata and metadata.lfs and sha256_file(existing) == metadata.lfs.sha256:
            logger.info("reusing local %s file=%s size=%s", kind, existing.name, _fmt_bytes(size))
            source = str(existing)
        else:
            logger.info("downloading %s repo=%s file=%s size=%s", kind, repo, filename, _fmt_bytes(size))
            source = hf_hub_download(repo, filename, revision=revision)
            logger.info("downloaded %s repo=%s file=%s", kind, repo, filename)
        target.symlink_to(Path(source).resolve())
    return name, {"repo": repo, "revision": revision, "filename": filename}


def prepare_base(api, task, root):
    model_type = task["model_type"]
    logger.info("preparing base model family=%s repo=%s", model_type, task.get("model_id"))
    if model_type == "ideogram4":
        repo = "Comfy-Org/Ideogram-4"
        filename = "diffusion_models/ideogram4_fp8_scaled.safetensors"
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
        logger.info("selected base checkpoint repo=%s file=%s size=%s", repo, filename, _fmt_bytes(candidates[0].size))
    folder = "unet" if model_type == "flux" else "diffusion_models"
    return materialize_model(api, repo, filename, root / "models" / folder)
