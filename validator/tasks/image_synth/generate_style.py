import datetime
import json
import os
import socket
import subprocess
import tempfile
import time
import uuid
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse

import runpod
from minio import Minio

import validator.tasks.image_synth.constants as cst
import validator.utils.comfy_api_gate as api_gate


COMFY_MAIN_PATH = "/app/validator/tasks/image_synth/ComfyUI/main.py"
COMFY_DIR = "/app/validator/tasks/image_synth/ComfyUI"
COMFY_PROCESS = None
COMFY_PORT = 8188

with open(cst.STYLE_WORKFLOW_PATH, "r") as file:
    style_template = json.load(file)


def _wait_for_port(host: str, port: int, timeout_seconds: int = 180) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout_seconds:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def _ensure_comfy_running() -> None:
    global COMFY_PROCESS
    if COMFY_PROCESS and COMFY_PROCESS.poll() is None:
        return
    COMFY_PROCESS = subprocess.Popen(
        ["/envs/comfyui/bin/python", COMFY_MAIN_PATH],
        cwd=COMFY_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    _wait_for_port("127.0.0.1", COMFY_PORT, timeout_seconds=180)


def _create_minio_client() -> tuple[Minio, str]:
    endpoint_raw = os.environ["S3_COMPATIBLE_ENDPOINT"]
    parsed = urlparse(endpoint_raw if "://" in endpoint_raw else f"https://{endpoint_raw}")
    endpoint = parsed.netloc or parsed.path
    secure = parsed.scheme != "http"
    bucket = os.environ["S3_BUCKET_NAME"]
    client = Minio(
        endpoint,
        access_key=os.environ["S3_COMPATIBLE_ACCESS_KEY"],
        secret_key=os.environ["S3_COMPATIBLE_SECRET_KEY"],
        region=os.environ.get("S3_REGION", "us-east-1"),
        secure=secure,
    )
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    return client, bucket


def _upload_pairs(save_dir: str) -> list[dict[str, str]]:
    client, bucket = _create_minio_client()
    uploaded = []
    expires = datetime.timedelta(seconds=604800)
    for image_path in sorted(Path(save_dir).glob("*.png")):
        text_path = image_path.with_suffix(".txt")
        if not text_path.exists():
            continue
        image_object = f"{uuid.uuid4()}.png"
        text_object = f"{uuid.uuid4()}.txt"
        try:
            client.fput_object(bucket, image_object, str(image_path))
            client.fput_object(bucket, text_object, str(text_path))
        except Exception:
            continue
        image_url = client.presigned_get_object(bucket, image_object, expires=expires)
        text_url = client.presigned_get_object(bucket, text_object, expires=expires)
        uploaded.append({"image_url": image_url, "text_url": text_url})
    return uploaded


def run_style_generation(prompts: list[str], save_dir: str) -> list[dict[str, str]]:
    _ensure_comfy_running()
    api_gate.connect()

    os.makedirs(save_dir, exist_ok=True)
    for prompt in prompts:
        workflow = deepcopy(style_template)
        workflow["Prompt"]["inputs"]["text"] += prompt
        image = api_gate.generate(workflow)[0]
        image_id = uuid.uuid4()
        image.save(f"{save_dir}{image_id}.png")
        with open(f"{save_dir}{image_id}.txt", "w") as file:
            file.write(prompt)

    return _upload_pairs(save_dir)


def handler(job):
    payload = job.get("input", {}) or {}
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts):
        raise ValueError("input.prompts must be a non-empty list of strings")

    with tempfile.TemporaryDirectory(prefix="style_synth_") as tmp_dir:
        pairs = run_style_generation(prompts, f"{tmp_dir}/")
    return {"status": "ok", "mode": "style", "num_pairs": len(pairs), "image_text_pairs": pairs}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})