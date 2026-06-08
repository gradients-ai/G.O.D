"""
Environment task stats: deploy model via SGLang, play episodes against env servers.
Self-contained — no validator imports. SGLang helpers inlined from eval_environment.py.
"""

import asyncio
import logging
import os
import random
import signal
import socket
import statistics
import subprocess
import time

import aiohttp

from core.constants import EnvironmentName
from core.models.model_prep_models import EnvBaselineStats
from core.models.model_prep_models import EnvStats


logger = logging.getLogger(__name__)

# Default SGLang CLI flags (inlined from validator.core.constants)
SGLANG_EXTRA_CLI_DEFAULT = (
    "--attention-backend triton --prefill-attention-backend triton "
    "--decode-attention-backend triton --sampling-backend pytorch"
)
SGLANG_HEALTH_TIMEOUT = 600
ENV_EVAL_TEMPERATURE = 0.0
ENV_EVAL_TASK_TIMEOUT = 150
CONSECUTIVE_FAILURE_LIMIT = 5


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


def _sample_task_id(seed: int, task_id_min: int, task_id_max: int) -> int:
    return random.Random(seed).randint(task_id_min, task_id_max)


def _format_episode_error(error: object) -> str:
    if error is None:
        return ""
    message = str(error).strip()
    return message or type(error).__name__


async def _play_episodes(
    session: aiohttp.ClientSession,
    env_name: EnvironmentName,
    env_server_url: str,
    sglang_base_url: str,
    model_name: str,
    num_episodes: int,
    task_id_min: int,
    task_id_max: int,
    eval_payload_extra: dict | None,
) -> EnvStats:
    """Play episodes against a single environment and return summary stats.

    Stops early if CONSECUTIVE_FAILURE_LIMIT episodes fail in a row — the
    remaining episodes would almost certainly fail too (model hallucinating,
    timeouts), so there's no signal in continuing.
    """
    seed_rng = random.Random(42)
    scores: list[float] = []
    consecutive_failures = 0

    print(f"  {env_name.value}: playing {num_episodes} episodes...", flush=True)

    for i in range(num_episodes):
        seed = seed_rng.randint(1, 1_000_000)
        task_id = _sample_task_id(seed, task_id_min, task_id_max)

        payload: dict = {
            "model": model_name,
            "base_url": sglang_base_url,
            "task_id": task_id,
            "temperature": ENV_EVAL_TEMPERATURE,
            "seed": seed,
        }
        if eval_payload_extra:
            payload.update(eval_payload_extra)

        failed = False
        error_message = ""
        try:
            timeout = aiohttp.ClientTimeout(total=ENV_EVAL_TASK_TIMEOUT)
            async with session.post(
                f"{env_server_url}/evaluate", json=payload, timeout=timeout,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", data)
                    score = float(result.get("score", 0.0))
                    error_message = _format_episode_error(result.get("error"))
                    if error_message:
                        failed = True
                else:
                    raw_error = await resp.text()
                    error_message = f"HTTP {resp.status}"
                    if raw_error:
                        error_message = f"{error_message}: {raw_error[:500]}"
                    score = 0.0
                    failed = True
        except Exception as e:
            error_message = _format_episode_error(e)
            score = 0.0
            failed = True

        scores.append(score)

        if failed:
            print(
                f"  {env_name.value} episode {i+1}: error task_id={task_id} seed={seed}: "
                f"{error_message or 'unknown error'}",
                flush=True,
            )
            consecutive_failures += 1
            if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                print(
                    f"  {env_name.value}: {CONSECUTIVE_FAILURE_LIMIT} consecutive failures, "
                    f"stopping early at episode {i+1}/{num_episodes}",
                    flush=True,
                )
                break
        else:
            consecutive_failures = 0

    stats = _build_env_stats(scores)
    print(f"  {env_name.value}: {stats.num_episodes} episodes, mean={stats.mean_score:.3f}", flush=True)
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
    from trainer.model_prep.stats import compute_weight_stats

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

        print(f"SGLang ready, base_url for env servers: {sglang_base_url}", flush=True)
        print(f"Evaluating {len(env_configs)} environments...", flush=True)

        async with aiohttp.ClientSession() as session:
            for env_name, cfg in env_configs.items():
                stats = await _play_episodes(
                    session=session,
                    env_name=env_name,
                    env_server_url=cfg["url"],
                    sglang_base_url=sglang_base_url,
                    model_name=model_name,
                    num_episodes=cfg["num_episodes"],
                    task_id_min=cfg["task_id_min"],
                    task_id_max=cfg["task_id_max"],
                    eval_payload_extra=cfg.get("eval_payload_extra"),
                )
                all_stats[env_name] = stats

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
