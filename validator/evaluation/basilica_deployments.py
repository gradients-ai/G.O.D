import asyncio
import json

import basilica
import requests

from core.logging import get_logger


logger = get_logger(__name__)

EVAL_RESULT_STATUS_PATH = "/result"


def deployment_is_healthy(deployment, health_path: str = "/health", timeout: int = 8) -> bool:
    try:
        response = requests.get(f"{deployment.url}{health_path}", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False

async def delete_deployment_if_exists(deployment_name: str) -> None:
    try:
        client = basilica.BasilicaClient()
        deployments = await asyncio.to_thread(client.list)
        for dep in deployments:
            if getattr(dep, "name", None) == deployment_name:
                await asyncio.to_thread(dep.delete)
                return
    except Exception:
        return

async def cleanup_basilica_deployments_by_name(deployment_names: set[str]) -> None:
    """Cleanup specific Basilica deployments by name."""
    if not deployment_names:
        return
    try:
        client = basilica.BasilicaClient()
        deployments = await asyncio.to_thread(client.list)
    except Exception as e:
        logger.warning(f"Failed to list deployments for final cleanup: {e}")
        return

    by_name = {getattr(dep, "name", None): dep for dep in deployments}
    cleaned = 0
    for name in deployment_names:
        dep = by_name.get(name)
        if dep is None:
            continue
        try:
            await asyncio.to_thread(dep.delete)
            cleaned += 1
        except Exception as e:
            logger.warning(f"Failed final cleanup for deployment {name}: {e}")

    if cleaned:
        logger.info(f"Final cleanup removed {cleaned} lingering deployments for this evaluation batch")


async def cleanup_all_basilica_deployments() -> None:
    """Delete every visible Basilica deployment after the evaluation queue drains."""
    try:
        client = basilica.BasilicaClient()
        deployments = await asyncio.to_thread(client.list)
    except Exception as e:
        logger.warning(f"Failed to list deployments for drained evaluation cleanup: {e}")
        return

    cleaned = 0
    for dep in deployments:
        deployment_name = getattr(dep, "name", "unknown")
        try:
            await asyncio.to_thread(dep.delete)
            cleaned += 1
        except Exception as e:
            logger.warning(f"Failed drained evaluation cleanup for deployment {deployment_name}: {e}")

    logger.info(f"Drained evaluation cleanup removed {cleaned}/{len(deployments)} Basilica deployments")


def create_basilica_eval_runner_source(
    command: list[str],
    result_path: str,
    prepare_image_models: bool = False,
) -> str:
    """Create a generic eval runner source with health and result endpoints.

    The runner executes a single eval command, then serves the parsed
    `evaluation_results.json` payload on `/result`.
    """
    command_json = json.dumps(command)
    result_path_json = json.dumps(result_path)
    prepare_image_models_literal = repr(prepare_image_models)
    return f"""import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

COMMAND = {command_json}
RESULT_PATH = {result_path_json}
RESULT_STATUS_PATH = "{EVAL_RESULT_STATUS_PATH}"
PREPARE_IMAGE_MODELS = {prepare_image_models_literal}

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

def _prepare_image_models():
    if not PREPARE_IMAGE_MODELS:
        return
    print("[eval_runner] preparing required image models...", flush=True)
    from validator.evaluation.image_model_downloads import prepare_required_image_models
    prepare_required_image_models(os.environ.get("MODEL_TYPE", ""))
    print("[eval_runner] image model prep complete", flush=True)

def _run_eval():
    try:
        _prepare_image_models()
        print("[eval_runner] starting eval command:", " ".join(COMMAND), flush=True)
        proc = subprocess.run(COMMAND, text=True)
        print(f"[eval_runner] eval command finished exit_code={{proc.returncode}}", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Eval command failed with exit code {{proc.returncode}}")
        with open(RESULT_PATH, "r", encoding="utf-8") as f:
            _state["result"] = json.load(f)
        _state["status"] = "completed"
    except Exception as e:
        if _state["status"] != "completed":
            _state["status"] = "failed"
            _state["error"] = str(e)

def main():
    server = HTTPServer(("0.0.0.0", 8000), _Handler)
    worker = threading.Thread(target=_run_eval, daemon=True)
    worker.start()
    server.serve_forever()

if __name__ == "__main__":
    main()
"""
