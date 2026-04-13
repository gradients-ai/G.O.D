import json
import threading
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from pathlib import Path

from validator.core import constants as cst
from validator.evaluation.eval_service_http import eval_service_normalize_http_path
from validator.evaluation.eval_service_http import eval_service_path_is_health
from validator.evaluation.eval_service_http import eval_service_path_is_result
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def run_eval_with_result_server(run_eval_fn, host: str = "0.0.0.0", port: int | None = None) -> None:
    """Serve /health and /result while running eval in background."""
    if port is None:
        port = cst.EVAL_SERVICE_PORT
    state = {
        "status": "running",
        "result": None,
        "error": None,
    }

    def _run_eval():
        try:
            run_eval_fn()
            result_path = Path(cst.CONTAINER_EVAL_RESULTS_PATH)
            if not result_path.exists():
                raise FileNotFoundError(f"Result file not found: {result_path}")
            state["result"] = json.loads(result_path.read_text(encoding="utf-8"))
            state["status"] = "completed"
        except BaseException as exc:
            state["status"] = "failed"
            state["error"] = str(exc)
            logger.exception("Evaluation failed in service mode: %s", exc)

    class _Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = eval_service_normalize_http_path(self.path)
            if eval_service_path_is_health(p):
                self._send_json(200, b'{"status":"ok"}')
                return
            if eval_service_path_is_result(p):
                self._send_json(200, json.dumps(state).encode("utf-8"))
                return
            self.send_response(404)
            self.end_headers()

        def do_HEAD(self):
            p = eval_service_normalize_http_path(self.path)
            if eval_service_path_is_health(p) or eval_service_path_is_result(p):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            return

    worker = threading.Thread(target=_run_eval, daemon=True)
    worker.start()

    server = HTTPServer((host, port), _Handler)
    logger.info("Starting eval service mode server on %s:%s", host, port)
    server.serve_forever()
