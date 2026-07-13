import asyncio
import json
from string import Template

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


def create_basilica_public_sglang_eval_runner_source(
    command: list[str],
    result_path: str,
    *,
    public_base_url_env: str = "SWE_INFINITE_MODEL_BASE_URL",
    model_api_key_env: str = "SWE_INFINITE_MODEL_API_KEY",
    sglang_base_url_env: str = "SGLANG_BASE_URL",
) -> str:
    """Create an eval runner that also exposes a token-protected SGLang `/v1` proxy.

    SWE Infinite runs its environment server outside the Basilica model deployment.
    The external Affinetes service therefore needs a public OpenAI-compatible URL
    for the candidate model, while the validator still needs `/health` and
    `/result` on the same exposed Basilica port.
    """
    return Template(
        r'''import http.client
import json
import os
import secrets
import subprocess
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMMAND = $command_json
RESULT_PATH = $result_path_json
RESULT_STATUS_PATH = "$result_status_path"
PUBLIC_BASE_URL_ENV = $public_base_url_env_json
MODEL_API_KEY_ENV = $model_api_key_env_json
SGLANG_BASE_URL_ENV = $sglang_base_url_env_json

_model_api_key = os.environ.get(MODEL_API_KEY_ENV) or secrets.token_urlsafe(32)
os.environ[MODEL_API_KEY_ENV] = _model_api_key
_state = {
    "status": "pending_start",
    "result": None,
    "error": None,
    "public_model_base_url": None,
}
_lock = threading.Lock()
_eval_started = False


def _infer_public_origin(handler):
    host = handler.headers.get("X-Forwarded-Host") or handler.headers.get("Host")
    if not host:
        return ""
    proto = handler.headers.get("X-Forwarded-Proto") or "https"
    return f"{proto}://{host}".rstrip("/")


def _run_eval():
    try:
        proc = subprocess.run(COMMAND, text=True, env=os.environ.copy())
        if proc.returncode != 0:
            raise RuntimeError(f"Eval command failed with exit code {proc.returncode}")
        with open(RESULT_PATH, "r", encoding="utf-8") as f:
            _state["result"] = json.load(f)
        _state["status"] = "completed"
    except Exception as e:
        if _state["status"] != "completed":
            _state["status"] = "failed"
            _state["error"] = str(e)


def _start_eval_if_needed(handler):
    global _eval_started
    with _lock:
        if _eval_started:
            return
        public_base_url = os.environ.get(PUBLIC_BASE_URL_ENV, "").rstrip("/")
        if not public_base_url:
            origin = _infer_public_origin(handler)
            if not origin:
                _state["status"] = "failed"
                _state["error"] = "Could not infer public Basilica deployment URL for model proxy"
                return
            public_base_url = f"{origin}/v1"
            os.environ[PUBLIC_BASE_URL_ENV] = public_base_url
        _state["status"] = "running"
        _state["public_model_base_url"] = public_base_url
        _eval_started = True
        worker = threading.Thread(target=_run_eval, daemon=True)
        worker.start()


def _proxy_authorized(handler):
    if not _model_api_key:
        return True
    auth = handler.headers.get("Authorization", "")
    return auth == f"Bearer {_model_api_key}" or auth == _model_api_key


def _proxy_to_sglang(handler):
    if not _proxy_authorized(handler):
        handler.send_response(401)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(b'{"error":"unauthorized"}')
        return

    local_base = os.environ.get(SGLANG_BASE_URL_ENV, "http://127.0.0.1:30000").rstrip("/")
    parsed = urllib.parse.urlparse(local_base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    base_path = parsed.path.rstrip("/")
    if base_path and handler.path.startswith(f"{base_path}/"):
        target_path = handler.path
    elif base_path:
        target_path = f"{base_path}{handler.path}"
    else:
        target_path = handler.path

    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    body = handler.rfile.read(content_length) if content_length else None
    headers = {
        key: value
        for key, value in handler.headers.items()
        if key.lower() not in {"host", "content-length", "connection", "accept-encoding"}
    }
    headers["Host"] = f"{host}:{port}"
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=1800)
    try:
        conn.request(handler.command, target_path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read()
        handler.send_response(response.status)
        for key, value in response.getheaders():
            if key.lower() in {"connection", "content-length", "transfer-encoding"}:
                continue
            handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
    except Exception as e:
        payload = json.dumps({"error": str(e)}).encode("utf-8")
        handler.send_response(502)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
    finally:
        conn.close()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        if self.path == RESULT_STATUS_PATH:
            _start_eval_if_needed(self)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            safe_state = dict(_state)
            safe_state.pop("public_model_base_url", None)
            self.wfile.write(json.dumps(safe_state).encode("utf-8"))
            return
        if self.path == "/v1" or self.path.startswith("/v1/"):
            _proxy_to_sglang(self)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/v1" or self.path.startswith("/v1/"):
            _proxy_to_sglang(self)
            return
        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        if self.path == "/v1" or self.path.startswith("/v1/"):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "authorization,content-type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer(("0.0.0.0", 8000), _Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
'''
    ).substitute(
        command_json=json.dumps(command),
        result_path_json=json.dumps(result_path),
        result_status_path=EVAL_RESULT_STATUS_PATH,
        public_base_url_env_json=json.dumps(public_base_url_env),
        model_api_key_env_json=json.dumps(model_api_key_env),
        sglang_base_url_env_json=json.dumps(sglang_base_url_env),
    )
