"""SGLang server lifecycle management for PvP evaluation.

Handles building launch commands, starting servers with GPU isolation,
stdout draining, and health checking.
"""

import asyncio
import logging
import os
import subprocess
import threading

from core.models.pvp_models import PreparedModel
from validator.core import constants as vcst
from validator.evaluation.eval_environment import _wait_for_health

logger = logging.getLogger(__name__)


def build_sglang_command(prepared: PreparedModel, port: int, seed: int) -> str:
    """Build SGLang launch command from a PreparedModel."""
    tensor_parallel = os.getenv("SGLANG_TENSOR_PARALLEL_SIZE", "1")
    dtype = os.getenv("SGLANG_DTYPE", "float16")
    cli_extra = (os.getenv("SGLANG_ENV_EVAL_EXTRA_CLI") or vcst.SGLANG_ENV_EVAL_EXTRA_CLI).strip()

    cmd = (
        "python3 -m sglang.launch_server "
        f"--model-path {prepared.sglang_model_path} "
        f"--host 0.0.0.0 --port {port} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--dtype {dtype} "
        f"--enable-deterministic-inference --random-seed {seed}"
    )
    if cli_extra:
        cmd = f"{cmd} {cli_extra}"
    if prepared.extra_sglang_args:
        cmd = f"{cmd} {prepared.extra_sglang_args}"
    return cmd


def start_sglang(prepared: PreparedModel, gpu_id: int, port: int, seed: int) -> subprocess.Popen:
    """Start an SGLang server on the specified GPU and port."""
    cmd = build_sglang_command(prepared, port, seed)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    logger.info("Starting SGLang on GPU %d port %d", gpu_id, port)
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
        env=env,
    )
    _drain_stdout(proc, f"sglang-gpu{gpu_id}")
    return proc


def _drain_stdout(proc: subprocess.Popen, name: str) -> None:
    """Drain subprocess stdout in a background thread to prevent pipe buffer deadlock."""

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            logger.info("[%s] %s", name, line.rstrip())
        proc.stdout.close()

    thread = threading.Thread(target=_reader, name=f"drain-{name}", daemon=True)
    thread.start()


async def wait_for_servers(port_a: int, port_b: int) -> None:
    """Wait for both SGLang instances to become healthy."""
    await asyncio.gather(
        _wait_for_health(
            f"http://{vcst.PVP_SGLANG_HOST}:{port_a}",
            vcst.PVP_SGLANG_HEALTH_PATH,
            vcst.PVP_SGLANG_HEALTH_TIMEOUT,
            service_name="sglang-a",
        ),
        _wait_for_health(
            f"http://{vcst.PVP_SGLANG_HOST}:{port_b}",
            vcst.PVP_SGLANG_HEALTH_PATH,
            vcst.PVP_SGLANG_HEALTH_TIMEOUT,
            service_name="sglang-b",
        ),
    )
