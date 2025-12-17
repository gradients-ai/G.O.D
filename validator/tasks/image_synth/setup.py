import os
import shutil
import subprocess

from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download


def clone_repo(repo_url: str, target_dir: str, commit_hash: str | None = None) -> None:
    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        subprocess.run(["git", "clone", "--depth", "1", repo_url, target_dir], check=True)
        if commit_hash:
            subprocess.run(["git", "fetch", "--depth", "1", "origin", commit_hash], cwd=target_dir, check=True)
            subprocess.run(["git", "checkout", commit_hash], cwd=target_dir, check=True)


def download_file(*, repo_id: str, filename: str, local_dir: str, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)

    is_file_path = os.path.splitext(local_dir)[1] != ""
    if is_file_path:
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    else:
        os.makedirs(local_dir, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
    )
    real_downloaded_path = os.path.realpath(downloaded_path)

    if is_file_path:
        target_path = local_dir
    else:
        target_path = os.path.join(local_dir, os.path.basename(filename))

    shutil.copyfile(real_downloaded_path, target_path)
    print(f"Downloaded and copied to: {target_path}")

    if not is_file_path and os.path.sep in filename:
        subdir = os.path.join(local_dir, os.path.dirname(filename))
        try:
            while subdir != local_dir:
                os.rmdir(subdir)
                subdir = os.path.dirname(subdir)
        except OSError:
            pass
    try:
        os.remove(real_downloaded_path)
        print(f"Removed from cache: {real_downloaded_path}")
    except OSError as e:
        print(f"Warning: Failed to remove cache file: {e}")



def snapshot_repo(repo_id: str, local_dir: str) -> None:
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading snapshot of {repo_id} into {local_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=os.path.abspath(local_dir),   
        local_dir_use_symlinks=False,           
        resume_download=True,
        ignore_patterns=["*.msgpack", "*.h5"],  
    )
    print(f"Snapshot of {repo_id} downloaded to {local_dir}")


def main():
    print("Cloning ComfyUI...")
    clone_repo(
        repo_url="https://github.com/comfyanonymous/ComfyUI.git",
        target_dir="ComfyUI",
        commit_hash="887143854bb2ae1e0f975e4461f376844a1628c8",
    )

    print("Cloned ComfyUI, now downloading all models... (this might take a while)")

    print("Downloading t5xxl_fp16.safetensors")
    download_file(
        repo_id="comfyanonymous/flux_text_encoders",
        filename="t5xxl_fp16.safetensors",
        local_dir="ComfyUI/models/text_encoders",
        cache_dir="ComfyUI/models/caches",
    )

    print("Downloading clip_l.safetensors")
    download_file(
        repo_id="comfyanonymous/flux_text_encoders",
        filename="clip_l.safetensors",
        local_dir="ComfyUI/models/text_encoders",
        cache_dir="ComfyUI/models/caches",
    )

    print("Downloading ae.safetensors")
    download_file(
        repo_id="tripathiarpan20/FLUX.1-schnell",
        filename="ae.safetensors",
        local_dir="ComfyUI/models/vae",
        cache_dir="ComfyUI/models/caches",
    )

    print("flux1-kontext-dev.safetensors")
    download_file(
        repo_id="diagonalge/kontext",
        filename="flux1-kontext-dev.safetensors",
        local_dir="ComfyUI/models/diffusion_models",
        cache_dir="ComfyUI/models/caches",
    )

    print("Downloading full repo liuhaotian/llava-v1.5-7b...")
    snapshot_repo(
        repo_id="liuhaotian/llava-v1.5-7b",
        local_dir="/app/validator/tasks/image_synth/cache/llava-v1.5-7b",
    )

    print("Downloading mistral_3_small_flux2_fp8.safetensors")
    download_file(
        repo_id="Comfy-Org/flux2-dev",
        filename="split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors",
        local_dir="ComfyUI/models/text_encoders",
        cache_dir="ComfyUI/models/caches",
    )

    print("Downloading flux2_dev_fp8mixed.safetensors")
    download_file(
        repo_id="Comfy-Org/flux2-dev",
        filename="split_files/diffusion_models/flux2_dev_fp8mixed.safetensors",
        local_dir="ComfyUI/models/diffusion_models",
        cache_dir="ComfyUI/models/caches",
    )

    print("Downloading flux2-vae.safetensors")
    download_file(
        repo_id="Comfy-Org/flux2-dev",
        filename="split_files/vae/flux2-vae.safetensors",
        local_dir="ComfyUI/models/vae",
        cache_dir="ComfyUI/models/caches",
    )

    print("Setup completed successfully.")


if __name__ == "__main__":
    main()
