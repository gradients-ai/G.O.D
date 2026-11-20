import json
import logging
import socket
import sys
import time

import requests

from trainer.constants import VECTOR_URL


HOSTNAME = socket.gethostname()
VECTOR_POST_TIMEOUT_SECONDS = 0.5
VECTOR_RETRY_SECONDS = 60.0


class VectorHandler(logging.Handler):
    """Ships trainer log records to the local Vector http_server.

    Best effort and silent: emit never raises and never logs through `logging` (no recursion). After a failed post it
    drops records for `retry_seconds` and prints one stderr line per outage, so a box without Vector on :8688 neither
    pays a blocking connect per record on the event loop nor fills the pm2 error log with a traceback per record.
    """

    def __init__(self, url: str, retry_seconds: float = VECTOR_RETRY_SECONDS, timeout: float = VECTOR_POST_TIMEOUT_SECONDS):
        super().__init__()
        self.url = url
        self.retry_seconds = retry_seconds
        self.timeout = timeout
        self._down_until = 0.0
        self._down = False

    def _entry(self, record: logging.LogRecord) -> dict:
        log_entry = {
            "message": record.getMessage(),
            "level": record.levelname,
            "logger": record.name,
            "timestamp": int(time.time() * 1000),
            "server": HOSTNAME,
        }
        for key, value in record.__dict__.items():
            if key not in log_entry and not key.startswith("_"):
                try:
                    json.dumps(value)
                    log_entry[key] = value
                except Exception:
                    log_entry[key] = str(value)
        return log_entry

    def _note(self, message: str) -> None:
        try:
            sys.stderr.write(f"{message}\n")
        except Exception:
            pass

    def emit(self, record: logging.LogRecord) -> None:
        now = time.monotonic()
        if now < self._down_until:
            return
        try:
            log_entry = self._entry(record)
        except Exception:  # a malformed record (bad format args) is dropped; it says nothing about the sink
            return
        try:
            requests.post(self.url, json=log_entry, timeout=self.timeout)
        except Exception as e:
            self._down_until = now + self.retry_seconds
            if not self._down:
                self._down = True
                self._note(
                    f"VectorHandler: {self.url} unreachable ({type(e).__name__}); "
                    f"dropping trainer telemetry, retrying every {self.retry_seconds:.0f}s"
                )
            return
        if self._down:
            self._down = False
            self._note(f"VectorHandler: {self.url} reachable again")


def setup_logger():
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    logger.addHandler(VectorHandler(VECTOR_URL))
    return logger


logger = setup_logger()
