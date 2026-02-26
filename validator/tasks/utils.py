import asyncio
import json
import uuid

import basilica
import requests
from core.models.payload_models import ImageTextPair

import validator.core.constants as cst
from validator.utils.logging import get_logger


logger = get_logger(__name__)

SYNTH_RESULT_STATUS_PATH = "/result"


def create_basilica_synth_runner_source(command: list[str]) -> str:
    command_json = json.dumps(command)
    return f"""import datetime
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from minio import Minio

COMMAND = {command_json}
RESULT_STATUS_PATH = "{SYNTH_RESULT_STATUS_PATH}"
SAVE_DIR = os.environ.get("SAVE_DIR", "/app/images/")

_state = {{
    "status": "running",
    "result": None,
    "error": None,
}}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{{"status":"ok"}}')
            return
        if self.path == RESULT_STATUS_PATH:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(_state).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


def _collect_valid_pairs(root: str) -> list[tuple[str, str]]:
    pairs = []
    for name in os.listdir(root):
        if not name.endswith(".png"):
            continue
        png_path = os.path.join(root, name)
        txt_path = os.path.join(root, f"{{os.path.splitext(name)[0]}}.txt")
        if os.path.isfile(txt_path) and os.path.getsize(txt_path) > 0:
            pairs.append((png_path, txt_path))
    return pairs


def _create_minio_client() -> tuple[Minio, str]:
    endpoint = os.environ["S3_COMPATIBLE_ENDPOINT"]
    access_key = os.environ["S3_COMPATIBLE_ACCESS_KEY"]
    secret_key = os.environ["S3_COMPATIBLE_SECRET_KEY"]
    region = os.environ.get("S3_REGION", "us-east-1")
    bucket = os.environ["S3_BUCKET_NAME"]
    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
        region=region,
    )
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    return client, bucket


def _upload_pairs_and_get_urls(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    client, bucket = _create_minio_client()
    presign_expiry = datetime.timedelta(days=7)
    uploaded_pairs: list[dict[str, str]] = []

    for png_path, txt_path in pairs:
        image_object_name = f"{{os.urandom(8).hex()}}.png"
        text_object_name = f"{{os.urandom(8).hex()}}.txt"
        client.fput_object(bucket, image_object_name, png_path)
        client.fput_object(bucket, text_object_name, txt_path)
        uploaded_pairs.append(
            {{
                "image_url": client.presigned_get_object(bucket, image_object_name, expires=presign_expiry),
                "text_url": client.presigned_get_object(bucket, text_object_name, expires=presign_expiry),
            }}
        )

    return uploaded_pairs


def _run_synth():
    try:
        proc = subprocess.run(COMMAND, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Synth command failed with exit code {{proc.returncode}}")

        if not os.path.isdir(SAVE_DIR):
            raise RuntimeError(f"SAVE_DIR not found: {{SAVE_DIR}}")

        pairs = _collect_valid_pairs(SAVE_DIR)
        if len(pairs) == 0:
            raise RuntimeError("No valid image/text pairs generated")

        image_text_pairs = _upload_pairs_and_get_urls(pairs)
        _state["status"] = "completed"
        _state["result"] = {{"image_text_pairs": image_text_pairs, "num_pairs": len(image_text_pairs)}}
    except Exception as e:
        _state["status"] = "failed"
        _state["error"] = str(e)


def main():
    server = HTTPServer(("0.0.0.0", 8000), _Handler)
    worker = threading.Thread(target=_run_synth, daemon=True)
    worker.start()
    server.serve_forever()


if __name__ == "__main__":
    main()
"""


def parse_synth_image_text_pairs(synth_result: dict) -> list[ImageTextPair]:
    pairs_raw = synth_result.get("image_text_pairs", [])
    if not isinstance(pairs_raw, list) or not pairs_raw:
        raise RuntimeError(f"Basilica synth result missing image_text_pairs: {synth_result}")
    return [ImageTextPair.model_validate(pair) for pair in pairs_raw]


async def poll_basilica_synth_result(deployment, deployment_name: str) -> dict:
    started = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - started < cst.EVAL_BASILICA_MAX_POLL_SECONDS:
        try:
            response = await asyncio.to_thread(
                requests.get,
                f"{deployment.url}{SYNTH_RESULT_STATUS_PATH}",
                timeout=30,
            )
        except Exception as e:
            logger.warning(f"[{deployment_name}] Synth polling error: {e}")
            await asyncio.sleep(cst.EVAL_BASILICA_POLL_INTERVAL_SECONDS)
            continue

        if response.status_code == 200:
            payload = response.json()
            status = payload.get("status")
            if status == "completed":
                result = payload.get("result")
                if isinstance(result, dict) and isinstance(result.get("image_text_pairs"), list):
                    return result
                raise RuntimeError(f"Completed but invalid result payload: {result}")
            if status == "failed":
                raise RuntimeError(payload.get("error", "Basilica synth reported failure"))

        await asyncio.sleep(cst.EVAL_BASILICA_POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"[{deployment_name}] Timed out waiting for synth result")


async def run_basilica_synth(command: list[str], env: dict[str, str], task_label: str, image: str | None = None) -> dict:
    source = create_basilica_synth_runner_source(command)
    target_image = image or cst.IMAGE_SYNTH_DOCKER_IMAGE

    for attempt in range(1, cst.EVAL_BASILICA_MAX_RETRIES + 1):
        deployment = None
        deployment_name = f"synth-{uuid.uuid4()}"
        try:
            client = basilica.BasilicaClient()
            deployment = await asyncio.to_thread(
                client.deploy,
                name=deployment_name,
                source=source,
                image=target_image,
                port=8000,
                cpu=cst.EVAL_BASILICA_CPU,
                memory=cst.EVAL_BASILICA_MEMORY,
                ttl_seconds=cst.EVAL_BASILICA_TTL_SECONDS,
                timeout=cst.EVAL_BASILICA_TIMEOUT,
                env=env,
                gpu_count=1,
                gpu_models=cst.BASILICA_GPU_MODELS,
                min_gpu_memory_gb=cst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
            )
            logger.info(f"[{task_label}] Basilica synth deployment started: {deployment_name}")
            return await poll_basilica_synth_result(deployment, deployment_name)
        except Exception as e:
            logger.error(
                f"[{task_label}] attempt {attempt}/{cst.EVAL_BASILICA_MAX_RETRIES} failed: {e}",
                exc_info=True,
            )
            if attempt < cst.EVAL_BASILICA_MAX_RETRIES:
                await asyncio.sleep(cst.EVAL_BASILICA_RETRY_DELAY_SECONDS)
            else:
                raise
        finally:
            if deployment is not None:
                try:
                    await asyncio.to_thread(deployment.delete)
                except Exception as e:
                    logger.warning(f"[{task_label}] failed to cleanup deployment {deployment_name}: {e}")
