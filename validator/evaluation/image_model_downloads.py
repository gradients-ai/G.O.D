import shutil
import urllib.request
from pathlib import Path


COMFY_MODELS_ROOT = Path("/app/validator/evaluation/ComfyUI/models")

IMAGE_SUPPORT_FILES = {
    "flux": [
        {
            "type": "hf",
            "repo_id": "comfyanonymous/flux_text_encoders",
            "filename": "clip_l.safetensors",
            "target": "text_encoders/clip_l.safetensors",
        },
        {
            "type": "hf",
            "repo_id": "comfyanonymous/flux_text_encoders",
            "filename": "t5xxl_fp16.safetensors",
            "target": "text_encoders/t5xxl_fp16.safetensors",
        },
        {
            "type": "url",
            "url": "https://huggingface.co/Albert-zp/flux-vaesft/resolve/main/fluxVaeSft_aeSft.sft",
            "target": "vae/ae.safetensors",
        },
    ],
    "z-image": [
        {
            "type": "hf",
            "repo_id": "Comfy-Org/z_image_turbo",
            "filename": "split_files/text_encoders/qwen_3_4b.safetensors",
            "target": "text_encoders/qwen_3_4b.safetensors",
        },
        {
            "type": "url",
            "url": "https://huggingface.co/Albert-zp/flux-vaesft/resolve/main/fluxVaeSft_aeSft.sft",
            "target": "vae/ae.safetensors",
        },
    ],
    "qwen-image": [
        {
            "type": "hf",
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "target": "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        },
        {
            "type": "hf",
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/vae/qwen_image_vae.safetensors",
            "target": "vae/qwen_image_vae.safetensors",
        },
    ],
    "krea2": [
        {
            "type": "hf",
            "repo_id": "Comfy-Org/Krea-2",
            "filename": "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
            "target": "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        },
        {
            "type": "hf",
            "repo_id": "Comfy-Org/Krea-2",
            "filename": "vae/qwen_image_vae.safetensors",
            "target": "vae/qwen_image_vae.safetensors",
        },
    ],
    "ideogram4": [
        {
            "type": "hf",
            "repo_id": "Comfy-Org/Ideogram-4",
            "filename": "text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
            "target": "text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
        },
        {
            "type": "hf",
            "repo_id": "Comfy-Org/Ideogram-4",
            "filename": "vae/flux2-vae.safetensors",
            "target": "vae/flux2-vae.safetensors",
        },
        {
            "type": "hf",
            "repo_id": "Comfy-Org/Ideogram-4",
            "filename": "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
            "target": "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
        },
    ],
}


def _log(message: str) -> None:
    print(f"[image_model_prep] {message}", flush=True)


def _target_model_path(relative_path: str) -> Path:
    return COMFY_MODELS_ROOT / relative_path


def _target_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MB"
    return f"{num_bytes} B"


def _download_hf_file(repo_id: str, filename: str, target_path: Path) -> None:
    if _target_exists(target_path):
        size = target_path.stat().st_size
        _log(f"skip existing {target_path} ({_format_size(size)})")
        return

    _log(f"downloading hf://{repo_id}/{filename} -> {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    shutil.copyfile(downloaded_path, target_path)
    size = target_path.stat().st_size
    _log(f"saved {target_path} ({_format_size(size)})")


def _download_url(url: str, target_path: Path) -> None:
    if _target_exists(target_path):
        size = target_path.stat().st_size
        _log(f"skip existing {target_path} ({_format_size(size)})")
        return

    _log(f"downloading {url} -> {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target_path)
    size = target_path.stat().st_size
    _log(f"saved {target_path} ({_format_size(size)})")


def prepare_required_image_models(model_type: str) -> None:
    specs = IMAGE_SUPPORT_FILES.get(model_type, [])
    _log(f"start model_type={model_type!r} files={len(specs)} root={COMFY_MODELS_ROOT}")
    if not specs:
        _log(f"no support files configured for model_type={model_type!r}")
        return

    for index, spec in enumerate(specs, start=1):
        target_path = _target_model_path(spec["target"])
        _log(f"step {index}/{len(specs)} type={spec['type']} target={spec['target']}")
        if spec["type"] == "hf":
            _download_hf_file(spec["repo_id"], spec["filename"], target_path)
        elif spec["type"] == "url":
            _download_url(spec["url"], target_path)
        else:
            raise ValueError(f"Unknown image support file type: {spec['type']}")

    _log(f"complete model_type={model_type!r}")
