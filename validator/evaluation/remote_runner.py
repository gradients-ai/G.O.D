"""Provider-neutral HTTP wrapper for remote evaluator containers."""

import http.client
import json
import os
import secrets
import subprocess
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer


COMMAND_ENV = "EVAL_RUNNER_COMMAND"
RESULT_PATH_ENV = "EVAL_RUNNER_RESULT_PATH"
MODE_ENV = "EVAL_RUNNER_MODE"
CONTROL_TOKEN_ENV = "EVAL_RUNNER_CONTROL_TOKEN"
PORT_ENV = "EVAL_RUNNER_PORT"
PUBLIC_BASE_URL_ENV = "SWE_INFINITE_MODEL_BASE_URL"
MODEL_API_KEY_ENV = "SWE_INFINITE_MODEL_API_KEY"
SGLANG_BASE_URL_ENV = "SGLANG_BASE_URL"
RESULT_STATUS_PATH = "/result"
MODE_STANDARD = "standard"
MODE_PUBLIC_SGLANG = "public_sglang"


def _load_command() -> list[str]:
    raw = os.environ.get(COMMAND_ENV, "")
    command = json.loads(raw)
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise ValueError(f"{COMMAND_ENV} must be a non-empty JSON string list")
    return command


COMMAND = _load_command()
RESULT_PATH = os.environ[RESULT_PATH_ENV]
MODE = os.environ.get(MODE_ENV, MODE_STANDARD)
CONTROL_TOKEN = os.environ.get(CONTROL_TOKEN_ENV, "")
MODEL_API_KEY = os.environ.get(MODEL_API_KEY_ENV) or secrets.token_urlsafe(32)
os.environ[MODEL_API_KEY_ENV] = MODEL_API_KEY

_state = {
    "status": "pending_start" if MODE == MODE_PUBLIC_SGLANG else "running",
    "result": None,
    "error": None,
}
_lock = threading.Lock()
_eval_started = False


def _run_eval() -> None:
    try:
        print("[eval_runner] starting eval command:", " ".join(COMMAND), flush=True)
        proc = subprocess.run(COMMAND, text=True, env=os.environ.copy())
        print(f"[eval_runner] eval command finished exit_code={proc.returncode}", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Eval command failed with exit code {proc.returncode}")
        with open(RESULT_PATH, encoding="utf-8") as result_file:
            _state["result"] = json.load(result_file)
        _state["status"] = "completed"
    except Exception as exc:
        if _state["status"] != "completed":
            _state["status"] = "failed"
            _state["error"] = str(exc)
        print(f"[eval_runner] failed: {exc}", flush=True)


def _infer_public_origin(handler: BaseHTTPRequestHandler) -> str:
    host = handler.headers.get("X-Forwarded-Host") or handler.headers.get("Host")
    if not host:
        return ""
    proto = handler.headers.get("X-Forwarded-Proto") or "https"
    return f"{proto}://{host}".rstrip("/")


def _start_eval_if_needed(handler: BaseHTTPRequestHandler) -> None:
    global _eval_started
    with _lock:
        if _eval_started:
            return
        if MODE == MODE_PUBLIC_SGLANG:
            public_base_url = os.environ.get(PUBLIC_BASE_URL_ENV, "").rstrip("/")
            if not public_base_url:
                origin = _infer_public_origin(handler)
                if not origin:
                    _state["status"] = "failed"
                    _state["error"] = "Could not infer public deployment URL for model proxy"
                    return
                os.environ[PUBLIC_BASE_URL_ENV] = f"{origin}/v1"
        _state["status"] = "running"
        _eval_started = True
        threading.Thread(target=_run_eval, daemon=True).start()


def _authorized(handler: BaseHTTPRequestHandler, token: str) -> bool:
    if not token:
        return True
    auth = handler.headers.get("Authorization", "")
    return secrets.compare_digest(auth, f"Bearer {token}") or secrets.compare_digest(auth, token)


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _normalized_request_path(raw_path: str) -> str:
    path = urllib.parse.urlparse(raw_path).path
    for endpoint in ("/health", RESULT_STATUS_PATH):
        if path == endpoint or path.endswith(endpoint):
            return endpoint
    v1_index = path.find("/v1")
    if v1_index >= 0:
        return path[v1_index:]
    return path


def _proxy_to_sglang(handler: BaseHTTPRequestHandler, request_path: str) -> None:
    if not _authorized(handler, MODEL_API_KEY):
        _write_json(handler, 401, {"error": "unauthorized"})
        return

    local_base = os.environ.get(SGLANG_BASE_URL_ENV, "http://127.0.0.1:30000").rstrip("/")
    parsed = urllib.parse.urlparse(local_base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    base_path = parsed.path.rstrip("/")
    if base_path and (request_path == base_path or request_path.startswith(f"{base_path}/")):
        target_path = request_path
    elif base_path:
        target_path = f"{base_path}{request_path}"
    else:
        target_path = request_path
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    body = handler.rfile.read(content_length) if content_length else None
    headers = {
        key: value
        for key, value in handler.headers.items()
        if key.lower() not in {"host", "content-length", "connection", "accept-encoding"}
    }
    headers["Host"] = f"{host}:{port}"
    conn_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = conn_class(host, port, timeout=1800)
    try:
        connection.request(handler.command, target_path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        handler.send_response(response.status)
        for key, value in response.getheaders():
            if key.lower() not in {"connection", "content-length", "transfer-encoding"}:
                handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
    except Exception as exc:
        _write_json(handler, 502, {"error": str(exc)})
    finally:
        connection.close()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        request_path = _normalized_request_path(self.path)
        if request_path == "/health":
            _write_json(self, 200, {"status": "ok"})
            return
        if request_path == RESULT_STATUS_PATH:
            if not _authorized(self, CONTROL_TOKEN):
                _write_json(self, 401, {"error": "unauthorized"})
                return
            _start_eval_if_needed(self)
            _write_json(self, 200, dict(_state))
            return
        if MODE == MODE_PUBLIC_SGLANG and (
            request_path == "/v1" or request_path.startswith("/v1/")
        ):
            _proxy_to_sglang(self, request_path)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        request_path = _normalized_request_path(self.path)
        if MODE == MODE_PUBLIC_SGLANG and (
            request_path == "/v1" or request_path.startswith("/v1/")
        ):
            _proxy_to_sglang(self, request_path)
            return
        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self) -> None:
        request_path = _normalized_request_path(self.path)
        if MODE == MODE_PUBLIC_SGLANG and (
            request_path == "/v1" or request_path.startswith("/v1/")
        ):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "authorization,content-type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, _format: str, *_args) -> None:
        return


def main() -> None:
    if MODE == MODE_STANDARD:
        _start_eval_if_needed(None)  # type: ignore[arg-type]
    port = int(os.environ.get(PORT_ENV, "8000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
