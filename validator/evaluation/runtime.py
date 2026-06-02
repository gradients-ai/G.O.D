import os
import signal
import subprocess
import time

import requests

from core.logging import get_logger


logger = get_logger(__name__)


def stop_process(proc: subprocess.Popen | None, name: str) -> None:
    """Gracefully stop a subprocess, escalating to SIGKILL if needed."""
    if proc is None:
        return
    try:
        if proc.poll() is None:
            logger.info("Stopping %s (pid=%s)", name, proc.pid)
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=10)
    except Exception as exc:
        logger.warning("Failed to stop %s cleanly: %s", name, exc)

def wait_for_basilica_health(url: str, timeout: int = 3600, path: str = "/v1/models") -> bool:
    """Wait for Basilica service to be healthy."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}{path}", timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)

    error_msg = f"Service at {url} did not become healthy within {timeout} seconds"
    raise TimeoutError(error_msg)
