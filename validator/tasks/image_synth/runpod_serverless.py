import json
import os
import socket
import subprocess
import tempfile
import time
import uuid
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

import runpod
from minio import Minio


COMFY_DIR = Path("/app/validator/tasks/image_synth/ComfyUI")
COMFY_MAIN = COMFY_DIR / "main.py"
COMFY_PORT = 8188
COMFY_PROCESS: subprocess.Popen | None = None


def _log(message: str) -> None:
    print(f"[runpod-image-synth] {message}", flush=True)


def _wait_for_port(host: str, port: int, timeout_seconds: int = 180) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_seconds:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {host}:{port} to become available")


def _ensure_comfy_running() -> None:
    global COMFY_PROCESS
    if COMFY_PROCESS and COMFY_PROCESS.poll() is None:
        return

    if not COMFY_MAIN.exists():
        raise FileNotFoundError(f"ComfyUI entrypoint not found: {COMFY_MAIN}")

    env = os.environ.copy()
    _log("Starting ComfyUI background process")
    COMFY_PROCESS = subprocess.Popen(
        ["/envs/comfyui/bin/python", str(COMFY_MAIN)],
        cwd=str(COMFY_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    _wait_for_port("127.0.0.1", COMFY_PORT, timeout_seconds=180)
    _log("ComfyUI is ready")


def _collect_pairs(save_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for png_path in sorted(save_dir.glob("*.png")):
        txt_path = png_path.with_suffix(".txt")
        if txt_path.exists() and txt_path.stat().st_size > 0:
            pairs.append((png_path, txt_path))
    return pairs


def _build_minio_client() -> tuple[Minio, str]:
    endpoint_raw = os.environ["S3_COMPATIBLE_ENDPOINT"]
    access_key = os.environ["S3_COMPATIBLE_ACCESS_KEY"]
    secret_key = os.environ["S3_COMPATIBLE_SECRET_KEY"]
    region = os.environ.get("S3_REGION", "us-east-1")
    bucket = os.environ["S3_BUCKET_NAME"]

    parsed = urlparse(endpoint_raw if "://" in endpoint_raw else f"https://{endpoint_raw}")
    endpoint = parsed.netloc or parsed.path
    secure = parsed.scheme != "http"

    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
        region=region,
    )
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    return client, bucket


def _upload_pairs(save_dir: Path) -> list[dict[str, str]]:
    pairs = _collect_pairs(save_dir)
    if not pairs:
        raise RuntimeError(f"No generated image/text pairs found in {save_dir}")

    client, bucket = _build_minio_client()
    presign_expiry = timedelta(days=7)
    uploaded: list[dict[str, str]] = []

    for png_path, txt_path in pairs:
        image_object = f"{uuid.uuid4()}.png"
        text_object = f"{uuid.uuid4()}.txt"
        client.fput_object(bucket, image_object, str(png_path))
        client.fput_object(bucket, text_object, str(txt_path))
        uploaded.append(
            {
                "image_url": client.presigned_get_object(bucket, image_object, expires=presign_expiry),
                "text_url": client.presigned_get_object(bucket, text_object, expires=presign_expiry),
            }
        )
    return uploaded


def _run_style_generation(payload: dict, save_dir: Path) -> None:
    env = os.environ.copy()
    env["SAVE_DIR"] = f"{save_dir}/"
    env["MODE"] = "style"
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts):
        raise ValueError("For style mode, input.prompts must be a non-empty list of strings")
    env["PROMPTS"] = json.dumps(prompts)
    _log("Running style generation via start.sh")
    result = subprocess.run(
        ["/app/start.sh"],
        cwd="/app",
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Style generation failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _run_person_generation(payload: dict, save_dir: Path) -> None:
    env = os.environ.copy()
    env["SAVE_DIR"] = f"{save_dir}/"
    env["MODE"] = "person"
    num_prompts = int(payload.get("num_prompts", 15))
    env["NUM_PROMPTS"] = str(num_prompts)
    _log("Running person generation via start.sh")
    result = subprocess.run(
        ["/app/start.sh"],
        cwd="/app",
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Person generation failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def handler(job):
    _ensure_comfy_running()

    payload = job.get("input", {}) or {}
    mode = payload.get("mode", "person")

    with tempfile.TemporaryDirectory(prefix=f"runpod_{mode}_") as tmp_dir:
        save_dir = Path(tmp_dir)
        if mode == "style":
            _run_style_generation(payload, save_dir)
        elif mode == "person":
            _run_person_generation(payload, save_dir)
        else:
            raise ValueError("mode must be one of: style, person")
        image_text_pairs = _upload_pairs(save_dir)

    return {
        "status": "ok",
        "mode": mode,
        "num_pairs": len(image_text_pairs),
        "image_text_pairs": image_text_pairs,
    }


if __name__ == "__main__":
    _log("Starting RunPod serverless worker")
    runpod.serverless.start({"handler": handler})
