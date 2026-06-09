"""
Environment task stats: deploy model via SGLang, play baseline games against MCTS.
No validator imports (model-prep ships core/ only) — the in-harness MCTS baseline
lives in core.pvp. SGLang helpers inlined from eval_environment.py.
"""

import asyncio
import functools
import logging
import os
import signal
import socket
import statistics
import subprocess
import time

import aiohttp

from core.constants import EnvironmentName
from core.models.model_prep_models import EnvBaselineStats
from core.models.model_prep_models import EnvStats
from core.models.pvp_models import ChatCompletionConfig
from core.pvp.baseline import run_mcts_baseline
from core.pvp.sglang_parsers import tool_call_parser_for
from core.pvp.chat import chat_completion
from core.pvp.chat import create_client
from trainer.model_prep.stats import compute_weight_stats


logger = logging.getLogger(__name__)

# Default SGLang CLI flags (inlined from validator.core.constants)
SGLANG_EXTRA_CLI_DEFAULT = (
    "--attention-backend triton --prefill-attention-backend triton "
    "--decode-attention-backend triton --sampling-backend pytorch"
)
SGLANG_HEALTH_TIMEOUT = 600
ENV_EVAL_TEMPERATURE = 0.0


# --- SGLang process management (from eval_environment.py) ---

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


LOG_SGLANG_STDOUT = _env_bool("MODEL_PREP_LOG_SGLANG", False)


def build_sglang_command(model_path: str, seed: int) -> str:
    tensor_parallel = os.getenv("SGLANG_TENSOR_PARALLEL_SIZE", "1")
    dtype = os.getenv("SGLANG_DTYPE", "float16")
    port = os.getenv("SGLANG_PORT", "30000")
    base = (
        "python3 -m sglang.launch_server "
        f"--model-path {model_path} "
        f"--host 0.0.0.0 --port {port} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--dtype {dtype} "
        f"--enable-deterministic-inference --random-seed {seed}"
    )
    parser = tool_call_parser_for(model_path)
    if parser:
        base = f"{base} --tool-call-parser {parser}"
    extra = (os.getenv("SGLANG_ENV_EVAL_EXTRA_CLI") or SGLANG_EXTRA_CLI_DEFAULT).strip()
    return f"{base} {extra}" if extra else base


def start_process(command: str, name: str, *, capture_stdout: bool = False) -> subprocess.Popen:
    logger.info("Starting %s: %s", name, command)
    stdout = subprocess.PIPE if capture_stdout else subprocess.DEVNULL
    stderr = subprocess.STDOUT if capture_stdout else subprocess.DEVNULL
    return subprocess.Popen(
        command, shell=True,
        stdout=stdout, stderr=stderr,
        text=True, bufsize=1, preexec_fn=os.setsid,
    )


async def stream_process_logs(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.stdout is None:
        return
    while True:
        if proc.poll() is not None and proc.stdout.closed:
            return
        line = await asyncio.to_thread(proc.stdout.readline)
        if not line:
            if proc.poll() is not None:
                return
            await asyncio.sleep(0.2)
            continue
        logger.info("[%s] %s", name, line.rstrip())


def stop_process(proc: subprocess.Popen | None, name: str) -> None:
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


async def wait_for_health(
    url: str, path: str, timeout_seconds: int, *, service_name: str = "service",
) -> None:
    deadline = time.time() + timeout_seconds
    started = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(f"{url}{path}", timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status == 200:
                        logger.info("%s healthy after %.1fs", service_name, time.time() - started)
                        return
            except Exception:
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"{service_name} at {url}{path} not healthy within {timeout_seconds}s")


def _build_env_stats(scores: list[float]) -> EnvStats:
    if scores:
        return EnvStats(
            num_episodes=len(scores),
            mean_score=statistics.mean(scores),
            std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            min_score=min(scores),
            max_score=max(scores),
            median_score=statistics.median(scores),
        )
    return EnvStats(num_episodes=0)


def _mcts_baseline_stats(
    env_name: EnvironmentName,
    sglang_base_url: str,
    model_name: str,
    model_path: str,
    num_episodes: int,
    eval_payload_extra: dict | None,
) -> EnvStats:
    """Play num_episodes baseline games of the model vs in-harness MCTS.

    Uses the same tool-calling format as eval (core.pvp), so the baseline is
    measured consistently with how the model is evaluated — no external server.
    """
    extra = eval_payload_extra or {}
    mcts_simulations = extra.get("mcts_max_simulations")

    config = ChatCompletionConfig(
        inference_model=model_name,
        # Local weights dir holds the tokenizer, so slot budgets use real tokens
        # (model_name is only a basename and would fall back to word counting).
        tokenizer_repo=model_path,
        base_url=sglang_base_url,
        temperature=ENV_EVAL_TEMPERATURE,
    )
    client = create_client(config)
    chat_fn = functools.partial(chat_completion, client)

    print(f"  {env_name.value}: playing {num_episodes} games vs MCTS...", flush=True)
    result = run_mcts_baseline(
        env_name=env_name,
        chat_fn=chat_fn,
        config=config,
        num_games=num_episodes,
        mcts_simulations=mcts_simulations,
    )

    # Per-game scores (win=1, draw=0.5, loss=0) -> the usual EnvStats summary.
    scores = [1.0] * result.wins + [0.5] * result.draws + [0.0] * result.losses
    stats = _build_env_stats(scores)
    print(f"  {env_name.value}: {result.num_games} games, mean={stats.mean_score:.3f}", flush=True)
    return stats


# --- Main entry point ---

async def compute_env_stats(
    model_path: str,
    model,
    env_configs: dict[EnvironmentName, dict],
) -> EnvBaselineStats:
    """Compute env stats: deploy model via SGLang, play episodes against all environments.

    env_configs maps EnvironmentName to a dict with keys:
        url: str           — env server URL on bridge network
        task_id_min: int
        task_id_max: int
        num_episodes: int
        eval_payload_extra: dict | None
    """
    print("Computing weight stats...", flush=True)
    weight_stats = compute_weight_stats(model)

    sglang_cmd = build_sglang_command(model_path, seed=42)
    sglang_proc = start_process(sglang_cmd, "sglang", capture_stdout=LOG_SGLANG_STDOUT)
    sglang_log_task = None
    sglang_port = int(os.getenv("SGLANG_PORT", "30000"))
    sglang_local_url = f"http://localhost:{sglang_port}"
    container_ip = socket.gethostbyname(socket.gethostname())
    sglang_base_url = f"http://{container_ip}:{sglang_port}/v1"
    model_name = os.path.basename(model_path)

    all_stats: dict[EnvironmentName, EnvStats] = {}

    try:
        if LOG_SGLANG_STDOUT:
            sglang_log_task = asyncio.create_task(stream_process_logs(sglang_proc, "sglang"))

        await wait_for_health(sglang_local_url, "/v1/models", SGLANG_HEALTH_TIMEOUT, service_name="sglang")

        print(f"SGLang ready at {sglang_base_url}", flush=True)
        print(f"Evaluating {len(env_configs)} environments vs MCTS...", flush=True)

        for env_name, cfg in env_configs.items():
            all_stats[env_name] = _mcts_baseline_stats(
                env_name=env_name,
                sglang_base_url=sglang_base_url,
                model_name=model_name,
                model_path=model_path,
                num_episodes=cfg["num_episodes"],
                eval_payload_extra=cfg.get("eval_payload_extra"),
            )

    except TimeoutError:
        print("SGLang failed to start within timeout", flush=True)

    finally:
        stop_process(sglang_proc, "sglang")
        if sglang_log_task:
            sglang_log_task.cancel()

    # Fill in empty stats for any envs that weren't reached
    for env_name in env_configs:
        if env_name not in all_stats:
            all_stats[env_name] = EnvStats(num_episodes=0)

    return EnvBaselineStats(
        weights=weight_stats,
        env_stats=all_stats,
    )
