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
            logger.info("Downloading base model (attempt %s/%s): %s", attempt, max_retries, repo_id)
            start = time.time()
            path = snapshot_download(repo_id, local_files_only=False)
            elapsed = time.time() - start
            logger.info("Base model downloaded in %.1fs: %s", elapsed, path)
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
            logger.info("Downloading LoRA (attempt %s/%s): %s", attempt, max_retries, repo_id)
            start = time.time()
            snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
            elapsed = time.time() - start
            logger.info("LoRA downloaded in %.1fs", elapsed)
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

    logger.info("Merging base model and LoRA adapter...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    lora_tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    logger.info("Base model loaded in %.1fs", time.time() - t0)

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
    model = PeftModel.from_pretrained(base, lora_dir)
    logger.info("LoRA adapter loaded in %.1fs", time.time() - t1)

    t2 = time.time()
    merged = model.merge_and_unload(safe_merge=False)
    logger.info("Merge completed in %.1fs", time.time() - t2)

    os.makedirs(output_dir, exist_ok=True)
    t3 = time.time()
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    target_tokenizer.save_pretrained(output_dir)
    logger.info("Merged model saved to %s in %.1fs", output_dir, time.time() - t3)
    return output_dir


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


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


async def _wait_for_health(url: str, path: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(f"{url}{path}", timeout=aiohttp.ClientTimeout(total=8)) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"Service at {url}{path} did not become healthy within {timeout_seconds}s")


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
    retry_statuses = {404, 500, 501}
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
                logger.info("[%s/%s] Seed=%s Task=%s", task_idx + 1, len(eval_list), seed, task_id)
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
                    return {"task_id": task_id, "score": score, "time": latency}
            except Exception as exc:
                if any(f"HTTP {code}" in str(exc) for code in retry_statuses):
                    if attempt >= vcst.ENV_EVAL_TASK_MAX_RETRIES:
                        logger.warning(
                            "Task %s failed after %s attempts with retryable HTTP status; excluding from average",
                            task_id,
                            attempt,
                        )
                        return None
                    await asyncio.sleep(vcst.ENV_EVAL_TASK_RETRY_DELAY)
                else:
                    logger.error("Task %s non-retryable error: %s", task_id, exc)
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
        for result in results:
            if isinstance(result, dict):
                all_results.append(result)

    if not all_results:
        return 0.0
    return sum(r["score"] for r in all_results) / len(all_results)


async def _run() -> None:
    env_proc = None
    sglang_proc = None
    sglang_log_task = None
    env_log_task = None

    try:
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

        is_lora = await asyncio.to_thread(check_for_lora, model_repo, False)
        should_merge_lora = False
        if is_lora:
            should_merge_lora = await asyncio.to_thread(check_lora_has_added_tokens, model_repo, False)

        inference_model_name = model_repo
        model_path_for_sglang = model_repo
        sglang_command = os.getenv("SGLANG_START_CMD")
        if not sglang_command:
            if is_lora and not should_merge_lora:
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
                model_path_for_sglang = await asyncio.to_thread(
                    _download_model_with_retry, model_repo
                )
                inference_model_name = model_repo
                sglang_command = _build_sglang_command(model_path_for_sglang, base_seed)

        sglang_proc = _start_process(sglang_command, "sglang")
        sglang_log_task = asyncio.create_task(_stream_logs(sglang_proc, "sglang"))

        sglang_base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        await _wait_for_health(
            sglang_base_url,
            os.getenv("SGLANG_HEALTH_PATH", "/v1/models"),
            int(os.getenv("SGLANG_HEALTH_TIMEOUT", "1800")),
        )

        env_command = os.getenv("ENV_SERVER_CMD")
        if not env_command and Path("/app/_affinetes/server.py").exists():
            env_command = _DEFAULT_AFFINETES_SERVER_CMD
        env_base_url = os.getenv("ENV_SERVER_BASE_URL", "http://127.0.0.1:8001")
        if env_command:
            env_proc = _start_process(env_command, "env-server")
            env_log_task = asyncio.create_task(_stream_logs(env_proc, "env-server"))

        await _wait_for_health(
            env_base_url,
            os.getenv("ENV_SERVER_HEALTH_PATH", "/health"),
            int(os.getenv("ENV_SERVER_HEALTH_TIMEOUT", "600")),
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
