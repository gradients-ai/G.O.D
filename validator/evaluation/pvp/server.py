"""SGLang server lifecycle management for PvP evaluation.

Handles building launch commands, starting servers with GPU isolation,
stdout draining, and health checking.
"""

import asyncio
import logging
import os
import subprocess
import threading

import validator.evaluation.constants as vcst
from core.models.pvp_models import PreparedModel
from core.pvp.sglang_launch import build_base_command
from core.pvp.sglang_parsers import tool_call_parser_for
from validator.evaluation.evaluators.environment import _wait_for_health


logger = logging.getLogger(__name__)


def build_sglang_command(prepared: PreparedModel, port: int, seed: int) -> str:
    """Build SGLang launch command from a PreparedModel."""
    cli_extra = (os.getenv("SGLANG_ENV_EVAL_EXTRA_CLI") or vcst.SGLANG_ENV_EVAL_EXTRA_CLI).strip()

    cmd = build_base_command(prepared.sglang_model_path, port, seed)
    # No parser -> SGLang won't emit structured tool_calls and every turn forfeits.
    parser = prepared.tool_call_parser or tool_call_parser_for(prepared.sglang_model_path)
    if parser:
        cmd = f"{cmd} --tool-call-parser {parser}"
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


async def wait_for_servers(port_a: int, port_b: int, timeout: int = vcst.PVP_SGLANG_HEALTH_TIMEOUT) -> None:
    """Wait for both SGLang instances to become healthy.

    Used both for initial startup (long timeout) and for mid-eval recovery after
    a server goes unreachable (shorter timeout). Raises TimeoutError if either
    server is not healthy within `timeout`.
    """
    await asyncio.gather(
        _wait_for_health(
            f"http://{vcst.PVP_SGLANG_HOST}:{port_a}",
            vcst.PVP_SGLANG_HEALTH_PATH,
            timeout,
            service_name="sglang-a",
        ),
        _wait_for_health(
            f"http://{vcst.PVP_SGLANG_HOST}:{port_b}",
            vcst.PVP_SGLANG_HEALTH_PATH,
            timeout,
            service_name="sglang-b",
        ),
    )
