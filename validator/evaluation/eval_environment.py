import asyncio
import glob
import json
import logging
import os
import signal
import importlib.util
import subprocess
import sys
import time
import random
from pathlib import Path

import aiohttp
from huggingface_hub import snapshot_download

from core import constants as cst
from core.models.utility_models import EnvironmentDatasetType
from validator.core import constants as vcst
from validator.evaluation.utils import (
    check_for_lora,
    check_lora_has_added_tokens,
)


logger = logging.getLogger(__name__)
_DEFAULT_AFFINETES_SERVER_CMD = vcst.ENV_SERVER_CMD_DEFAULT


def _download_model_with_retry(repo_id: str, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "eval_setup download base model (attempt %s/%s): %s",
                attempt,
                max_retries,
                repo_id,
            )
            start = time.time()
            path = snapshot_download(repo_id, local_files_only=False)
            elapsed = time.time() - start
            logger.info("eval_setup base model snapshot_download done in %.1fs -> %s", elapsed, path)
            return path
        except Exception as exc:
            logger.warning("Download attempt %s failed: %s", attempt, exc)
            if attempt < max_retries:
                wait = 30 * attempt
                logger.info("Retrying in %ss...", wait)
                time.sleep(wait)
            else:
                logger.error("All download attempts failed")
                raise


def _download_lora_with_retry(repo_id: str, local_dir: str, max_retries: int = 3) -> str:
    os.makedirs(local_dir, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "eval_setup download LoRA (attempt %s/%s): %s -> %s",
                attempt,
                max_retries,
                repo_id,
                local_dir,
            )
            start = time.time()
            snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
            elapsed = time.time() - start
            logger.info("eval_setup LoRA snapshot_download done in %.1fs", elapsed)
            return local_dir
        except Exception as exc:
            logger.warning("Download attempt %s failed: %s", attempt, exc)
            if attempt < max_retries:
                wait = 30 * attempt
                logger.info("Retrying in %ss...", wait)
                time.sleep(wait)
            else:
                logger.error("All download attempts failed")
                raise


def _merge_base_and_lora(base_model_path: str, lora_dir: str, output_dir: str = "/tmp/merged_model") -> str:
    needs_install = (
        importlib.util.find_spec("peft") is None
        or importlib.util.find_spec("accelerate") is None
    )
    if needs_install:
        logger.info("Installing merge dependencies at runtime...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "peft", "accelerate"],
            check=True,
        )
        logger.info("Merge dependencies installed")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    merge_t0 = time.time()
    logger.info(
        "eval_setup merge: start base=%s lora=%s out=%s",
        base_model_path,
        lora_dir,
        output_dir,
    )
    logger.info("eval_setup merge: loading tokenizers...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    lora_tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)

    t0 = time.time()
    logger.info("eval_setup merge: loading base weights (AutoModelForCausalLM.from_pretrained)...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    logger.info("eval_setup merge: base weights in memory in %.1fs", time.time() - t0)

    base_vocab_size = base.get_input_embeddings().weight.shape[0]
    target_tokenizer = lora_tokenizer if len(lora_tokenizer) >= base_vocab_size else base_tokenizer
    target_vocab_size = len(target_tokenizer)
    if target_vocab_size > base_vocab_size:
        logger.info("Resizing token embeddings from %s to %s", base_vocab_size, target_vocab_size)
        base.resize_token_embeddings(target_vocab_size)
    elif target_vocab_size < base_vocab_size:
        logger.info(
            "LoRA tokenizer smaller than base (%s < %s); keeping base vocab size.",
            target_vocab_size,
            base_vocab_size,
        )

    t1 = time.time()
    logger.info("eval_setup merge: attaching LoRA (PeftModel.from_pretrained)...")
    model = PeftModel.from_pretrained(base, lora_dir)
    logger.info("eval_setup merge: LoRA attached in %.1fs", time.time() - t1)

    t2 = time.time()
    logger.info("eval_setup merge: merge_and_unload...")
    merged = model.merge_and_unload(safe_merge=False)
    logger.info("eval_setup merge: merge_and_unload done in %.1fs", time.time() - t2)

    os.makedirs(output_dir, exist_ok=True)
    t3 = time.time()
    logger.info("eval_setup merge: saving merged model to disk...")
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    target_tokenizer.save_pretrained(output_dir)
    logger.info(
        "eval_setup merge: saved to %s in %.1fs (total merge wall %.1fs)",
        output_dir,
        time.time() - t3,
        time.time() - merge_t0,
    )
    return output_dir


def _configure_logging() -> None:
    """
    Ensure INFO logs reach stderr (Basilica/Docker). If root was configured earlier,
    basicConfig alone is a no-op; attach a stderr handler explicitly.
    """
    level_name = os.getenv("EVAL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root.addHandler(handler)
    logger.setLevel(level)


def _parse_environment_name() -> str:
    dataset_type_raw = os.getenv("DATASET_TYPE", "{}")
    env_name = os.getenv("ENVIRONMENT_NAME")

    if not env_name:
        try:
            dataset_type = EnvironmentDatasetType.model_validate_json(dataset_type_raw)
            env_name = dataset_type.environment_name
        except Exception:
            env_name = None

    if not env_name:
        raise ValueError("Missing environment name. Set ENVIRONMENT_NAME or DATASET_TYPE.")

    if env_name not in vcst.ENVIRONMENTS:
        raise ValueError(f"Unsupported environment '{env_name}'. Supported: {list(vcst.ENVIRONMENTS.keys())}")
    return env_name


def _build_sglang_command(model_path: str, seed: int) -> str:
    tensor_parallel = os.getenv("SGLANG_TENSOR_PARALLEL_SIZE", "1")
    dtype = os.getenv("SGLANG_DTYPE", "float16")
    port = os.getenv("SGLANG_PORT", "30000")
    return (
        "python3 -m sglang.launch_server "
        f"--model-path {model_path} "
        f"--host 0.0.0.0 --port {port} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--dtype {dtype} "
        f"--enable-deterministic-inference --random-seed {seed}"
    )


def _start_process(command: str, name: str) -> subprocess.Popen:
    logger.info("Starting %s: %s", name, command)
    return subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )


def _stop_process(proc: subprocess.Popen | None, name: str) -> None:
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


async def _wait_for_health(
    url: str,
    path: str,
    timeout_seconds: int,
    *,
    service_name: str = "service",
    log_interval_s: float = 30.0,
) -> None:
    deadline = time.time() + timeout_seconds
    started = time.time()
    last_log = started
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(f"{url}{path}", timeout=aiohttp.ClientTimeout(total=8)) as response:
                    if response.status == 200:
                        logger.info(
                            "eval_setup %s healthy after %.1fs (GET %s%s -> %s)",
                            service_name,
                            time.time() - started,
                            url,
                            path,
                            response.status,
                        )
                        return
            except Exception:
                pass
            now = time.time()
            if now - last_log >= log_interval_s:
                logger.info(
                    "eval_setup still waiting for %s: GET %s%s (elapsed=%.0fs / timeout=%ss)",
                    service_name,
                    url,
                    path,
                    now - started,
                    timeout_seconds,
                )
                last_log = now
            await asyncio.sleep(2)
    raise TimeoutError(
        f"{service_name} at {url}{path} did not become healthy within {timeout_seconds}s"
    )


async def _stream_logs(proc: subprocess.Popen | None, name: str) -> None:
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


async def _run_environment_evaluation(
    sglang_url: str,
    env_url: str,
    eval_seeds: list[int],
    task_id_max: int,
    task_id_min: int,
    inference_model_name: str,
    temperature: float,
    env_payload_extra: dict,
) -> float:
    eval_list = []
    for seed in eval_seeds:
        rng = random.Random(seed)
        task_id = rng.randint(task_id_min + 1, task_id_max)
        eval_list.append((seed, task_id))

    all_results = []
    total_tasks = len(eval_list)
    logger.info("eval_progress batch: %s tasks (concurrency=%s)", total_tasks, vcst.ENV_EVAL_MAX_CONCURRENT_REQUESTS)
    retry_statuses = {404, 500, 501, 503, 504}
    semaphore = asyncio.Semaphore(vcst.ENV_EVAL_MAX_CONCURRENT_REQUESTS)

    async def evaluate_single_task(
        session: aiohttp.ClientSession,
        seed: int,
        task_id: int,
        task_idx: int,
    ) -> dict | None:
        payload = {
            "model": inference_model_name,
            "base_url": f"{sglang_url}/v1",
            "task_id": task_id,
            "temperature": temperature,
            "seed": seed,
        }
        if env_payload_extra:
            payload.update(env_payload_extra)

        attempt = 0
        while True:
            attempt += 1
            start_ts = time.time()
            try:
                if attempt == 1:
                    logger.info(
                        "eval_progress %s/%s start task_id=%s seed=%s",
                        task_idx + 1,
                        total_tasks,
                        task_id,
                        seed,
                    )
                timeout = aiohttp.ClientTimeout(total=vcst.ENV_EVAL_TASK_TIMEOUT)
                async with session.post(
                    f"{env_url}/evaluate",
                    json=payload,
                    timeout=timeout,
                    headers={"Connection": "close"},
                ) as response:
                    raw_text = await response.text()
                    if response.status != 200:
                        error_detail = f": {raw_text[:500]}" if raw_text else ""
                        raise RuntimeError(f"HTTP {response.status}{error_detail}")

                    response_data = json.loads(raw_text)
                    result = response_data.get("result", response_data)
                    latency = result.get("time_taken", time.time() - start_ts)
                    score = result.get("score", 0.0)
                    logger.info(
                        "eval_progress %s/%s done task_id=%s score=%.6f latency_s=%.3f",
                        task_idx + 1,
                        total_tasks,
                        task_id,
                        score,
                        latency,
                    )
                    return {"task_id": task_id, "score": score, "time": latency}
            except Exception as exc:
                if any(f"HTTP {code}" in str(exc) for code in retry_statuses):
                    if attempt >= vcst.ENV_EVAL_TASK_MAX_RETRIES:
                        logger.warning(
                            "eval_progress %s/%s failed_excluded task_id=%s attempts=%s (retryable HTTP); excluding from average",
                            task_idx + 1,
                            total_tasks,
                            task_id,
                            attempt,
                        )
                        return None
                    await asyncio.sleep(vcst.ENV_EVAL_TASK_RETRY_DELAY)
                else:
                    logger.error(
                        "eval_progress %s/%s error task_id=%s: %s",
                        task_idx + 1,
                        total_tasks,
                        task_id,
                        exc,
                    )
                    logger.info(
                        "eval_progress %s/%s done task_id=%s score=0.000000 (non-retryable error)",
                        task_idx + 1,
                        total_tasks,
                        task_id,
                    )
                    return {"task_id": task_id, "score": 0.0, "time": 0.0}

    async def evaluate_with_semaphore(
        session: aiohttp.ClientSession, seed: int, task_id: int, task_idx: int
    ) -> dict | None:
        async with semaphore:
            return await evaluate_single_task(session, seed, task_id, task_idx)

    session_timeout = aiohttp.ClientTimeout(total=vcst.ENV_EVAL_SESSION_TIMEOUT)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        tasks = [
            evaluate_with_semaphore(session, seed, task_id, idx)
            for idx, (seed, task_id) in enumerate(eval_list)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        n_exc = sum(1 for r in results if isinstance(r, BaseException))
        for result in results:
            if isinstance(result, dict):
                all_results.append(result)
        if n_exc:
            logger.warning(
                "eval_progress batch: %s task(s) raised exceptions (not counted in average)",
                n_exc,
            )

    if not all_results:
        logger.warning("eval_progress batch: no successful task results; returning 0.0")
        return 0.0
    avg = sum(r["score"] for r in all_results) / len(all_results)
    logger.info(
        "eval_progress batch: finished %s/%s tasks with scores, avg_score=%.6f",
        len(all_results),
        total_tasks,
        avg,
    )
    return avg


async def _run() -> None:
    env_proc = None
    sglang_proc = None
    sglang_log_task = None
    env_log_task = None

    try:
        logger.info(
            "eval_environment: start pid=%s EVAL_LOG_LEVEL=%s",
            os.getpid(),
            os.getenv("EVAL_LOG_LEVEL", "INFO"),
        )

        models_raw = os.getenv("MODELS", "")
        model_repo = models_raw.split(",")[0].strip()
        if not model_repo:
            raise ValueError("MODELS is required and must contain a single repo")

        original_model = os.getenv("ORIGINAL_MODEL", model_repo)
        base_seed = int(os.getenv("EVAL_SEED", str(vcst.ENV_EVAL_DEFAULT_SEED)))
        temperature = float(os.getenv("ENV_EVAL_TEMPERATURE", str(vcst.ENV_EVAL_TEMPERATURE)))

        env_name = _parse_environment_name()
        env_config = vcst.ENVIRONMENTS[env_name]
        task_id_min, task_id_max = env_config["task_id_range"]
        num_seeds = env_config.get("num_seeds", vcst.ENV_EVAL_NUM_SEEDS)
        env_payload_extra = env_config.get("eval_payload_extra", {})

        seed_generator = random.Random(base_seed)
        eval_seeds = [seed_generator.randint(1, 1_000_000) for _ in range(num_seeds)]

        logger.info(
            "eval_setup config: env=%s num_seeds=%s task_id_range=(%s,%s) model_repo=%s original_model=%s "
            "eval_seed=%s temperature=%s",
            env_name,
            num_seeds,
            task_id_min,
            task_id_max,
            model_repo,
            original_model,
            base_seed,
            temperature,
        )

        t_det = time.time()
        is_lora = await asyncio.to_thread(check_for_lora, model_repo, False)
        should_merge_lora = False
        if is_lora:
            should_merge_lora = await asyncio.to_thread(check_lora_has_added_tokens, model_repo, False)
        logger.info(
            "eval_setup LoRA detection in %.2fs: is_lora=%s merge_lora_to_base=%s",
            time.time() - t_det,
            is_lora,
            should_merge_lora,
        )

        inference_model_name = model_repo
        model_path_for_sglang = model_repo
        sglang_command = os.getenv("SGLANG_START_CMD")
        if sglang_command:
            logger.info("eval_setup SGLang: using SGLANG_START_CMD from environment (override)")
        if not sglang_command:
            if is_lora and not should_merge_lora:
                logger.info(
                    "eval_setup model path: LoRA + SGLang native (base=%s lora_repo=%s)",
                    original_model,
                    model_repo,
                )
                model_path_for_sglang = await asyncio.to_thread(
                    _download_model_with_retry, original_model
                )
                lora_dir = "/lora/trained_lora"
                await asyncio.to_thread(
                    _download_lora_with_retry, model_repo, lora_dir
                )
                for model_file in glob.glob(os.path.join(lora_dir, "model-*.safetensors")):
                    try:
                        os.remove(model_file)
                        logger.info("Removed incompatible LoRA file: %s", os.path.basename(model_file))
                    except Exception as exc:
                        logger.warning("Failed to remove %s: %s", model_file, exc)
                index_file = os.path.join(lora_dir, "model.safetensors.index.json")
                if os.path.exists(index_file):
                    try:
                        os.remove(index_file)
                    except Exception as exc:
                        logger.warning("Failed to remove index file: %s", exc)
                inference_model_name = f"{original_model}:trained_lora"
                sglang_command = (
                    _build_sglang_command(model_path_for_sglang, base_seed)
                    + " --enable-lora --lora-paths trained_lora=/lora/trained_lora --lora-backend triton"
                )
            elif is_lora and should_merge_lora:
                logger.info(
                    "eval_setup model path: merge LoRA into base then SGLang (base=%s lora=%s)",
                    original_model,
                    model_repo,
                )
                base_path = await asyncio.to_thread(
                    _download_model_with_retry, original_model
                )
                lora_temp_dir = "/tmp/lora/trained_lora"
                await asyncio.to_thread(
                    _download_lora_with_retry, model_repo, lora_temp_dir
                )
                model_path_for_sglang = await asyncio.to_thread(
                    _merge_base_and_lora, base_path, lora_temp_dir
                )
                inference_model_name = model_repo
                sglang_command = _build_sglang_command(model_path_for_sglang, base_seed)
            else:
                logger.info("eval_setup model path: single HF repo (full weights) repo=%s", model_repo)
                model_path_for_sglang = await asyncio.to_thread(
                    _download_model_with_retry, model_repo
                )
                inference_model_name = model_repo
                sglang_command = _build_sglang_command(model_path_for_sglang, base_seed)

        sglang_health_timeout = int(os.getenv("SGLANG_HEALTH_TIMEOUT", "1800"))
        env_health_timeout = int(os.getenv("ENV_SERVER_HEALTH_TIMEOUT", "600"))
        logger.info(
            "eval_setup launching SGLang: model_path_for_sglang=%s inference_model_name=%s",
            model_path_for_sglang,
            inference_model_name,
        )
        logger.info("eval_setup SGLang command: %s", sglang_command)
        logger.info(
            "eval_setup health: SGLang timeout=%ss GET %s%s | env timeout=%ss GET %s%s",
            sglang_health_timeout,
            os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000"),
            os.getenv("SGLANG_HEALTH_PATH", "/v1/models"),
            env_health_timeout,
            os.getenv("ENV_SERVER_BASE_URL", "http://127.0.0.1:8001"),
            os.getenv("ENV_SERVER_HEALTH_PATH", "/health"),
        )

        sglang_proc = _start_process(sglang_command, "sglang")
        sglang_log_task = asyncio.create_task(_stream_logs(sglang_proc, "sglang"))

        sglang_base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        await _wait_for_health(
            sglang_base_url,
            os.getenv("SGLANG_HEALTH_PATH", "/v1/models"),
            sglang_health_timeout,
            service_name="SGLang",
        )

        env_command = os.getenv("ENV_SERVER_CMD")
        if not env_command and Path("/app/_affinetes/server.py").exists():
            env_command = _DEFAULT_AFFINETES_SERVER_CMD
        env_base_url = os.getenv("ENV_SERVER_BASE_URL", "http://127.0.0.1:8001")
        if env_command:
            logger.info("eval_setup starting env-server subprocess")
            env_proc = _start_process(env_command, "env-server")
            env_log_task = asyncio.create_task(_stream_logs(env_proc, "env-server"))
        else:
            logger.info(
                "eval_setup no ENV_SERVER_CMD; expecting env already up at %s",
                env_base_url,
            )

        await _wait_for_health(
            env_base_url,
            os.getenv("ENV_SERVER_HEALTH_PATH", "/health"),
            env_health_timeout,
            service_name="env-server",
        )

        logger.info(
            "eval_setup starting rollouts: inference_model_name=%s env_url=%s sglang_url=%s",
            inference_model_name,
            env_base_url,
            sglang_base_url,
        )

        avg_score = await _run_environment_evaluation(
            sglang_url=sglang_base_url,
            env_url=env_base_url,
            eval_seeds=eval_seeds,
            task_id_max=task_id_max,
            task_id_min=task_id_min,
            inference_model_name=inference_model_name,
            temperature=temperature,
            env_payload_extra=env_payload_extra,
        )

        output = {model_repo: {"is_finetune": True, "eval_loss": avg_score}}
        result_path = Path(cst.CONTAINER_EVAL_RESULTS_PATH)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(output), encoding="utf-8")
        logger.info(
            "eval_environment: wrote results to %s avg_score=%.6f",
            result_path,
            avg_score,
        )
        logger.info("Environment evaluation complete. avg_score=%.6f", avg_score)
    finally:
        _stop_process(env_proc, "env-server")
        _stop_process(sglang_proc, "sglang")
        if env_log_task:
            env_log_task.cancel()
        if sglang_log_task:
            sglang_log_task.cancel()


def main() -> int:
    _configure_logging()
    try:
        asyncio.run(_run())
        return 0
    except Exception as exc:
        logger.exception("Environment evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
