"""SWE Infinite evaluator for individual environment tournaments.

Runs inside a Basilica deployment. The deployment serves the candidate model
with SGLang, while the SWE Infinite environment server runs separately and
receives the public OpenAI-compatible model URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import aiohttp

import core.constants.environments as env_cst
import validator.evaluation.constants as vcst
from core.models.dataset_models import EnvironmentDatasetType
from core.models.pvp_models import PreparedModel
from core.pvp.sglang_parsers import tool_call_parser_for
from validator.evaluation.evaluation_logging import configure_eval_logging
from validator.evaluation.evaluators.environment import _start_process
from validator.evaluation.evaluators.environment import _stream_logs
from validator.evaluation.evaluators.environment import _wait_for_health
from validator.evaluation.model_checks import check_for_lora
from validator.evaluation.pvp.materialize import materialize_base_model
from validator.evaluation.pvp.server import build_sglang_command
from validator.evaluation.runtime import stop_process
from validator.evaluation.swe_infinite_config import DEFAULT_SWE_INFINITE_EVAL_CONFIG
from validator.evaluation.swe_infinite_config import DEFAULT_SWE_INFINITE_MODEL_API_KEY
from validator.evaluation.swe_infinite_config import SWE_INFINITE_AGENT_NAME
from validator.evaluation.swe_infinite_config import SWE_INFINITE_MODEL_API_KEY_ENV
from validator.evaluation.swe_infinite_config import SWE_INFINITE_MODEL_BASE_URL_ENV
from validator.evaluation.swe_infinite_config import SWE_INFINITE_SERVER_BASE_URL_ENV
from validator.evaluation.swe_infinite_config import SWE_INFINITE_VETTED_TASK_IDS
from validator.evaluation.swe_infinite_config import SweInfiniteEvalConfig
from validator.evaluation.swe_infinite_config import SweInfiniteTaskSelectionOverride
from validator.evaluation.swe_infinite_config import load_swe_infinite_eval_config
from validator.evaluation.swe_infinite_config import load_swe_infinite_task_selection_override
from validator.tasks.datasets.constants import CONTAINER_EVAL_RESULTS_PATH


logger = logging.getLogger(__name__)

DEFAULT_TASK_TIMEOUT_SECONDS = DEFAULT_SWE_INFINITE_EVAL_CONFIG.task_timeout_seconds
MINISWE_AGENT_NAME = SWE_INFINITE_AGENT_NAME
SWE_VETTED_TASK_IDS = SWE_INFINITE_VETTED_TASK_IDS


def _with_v1(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _parse_environment_name() -> env_cst.EnvironmentName:
    dataset_type_raw = os.getenv("DATASET_TYPE", "{}")
    env_name = os.getenv("ENVIRONMENT_NAME")
    if not env_name:
        try:
            dataset_type = EnvironmentDatasetType.model_validate_json(dataset_type_raw)
            environment_names = dataset_type.environment_names or []
            env_name = environment_names[0] if environment_names else None
        except Exception:
            env_name = None
    env_name = getattr(env_name, "value", env_name)
    if not env_name:
        raise ValueError("Missing environment name. Set ENVIRONMENT_NAME or DATASET_TYPE.")
    if env_name != env_cst.EnvironmentName.SWE_INFINITE.value:
        raise ValueError(
            f"eval_swe invoked with environment_name={env_name!r}; expected "
            f"{env_cst.EnvironmentName.SWE_INFINITE.value!r}"
        )
    return env_cst.EnvironmentName.SWE_INFINITE


async def _fetch_swe_completed_up_to(metadata_url: str) -> int | None:
    if not metadata_url:
        return None
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            async with session.get(metadata_url) as response:
                if response.status != 200:
                    logger.warning("SWE metadata fetch got HTTP %s from %s", response.status, metadata_url)
                    return None
                payload = await response.json()
    except Exception as exc:
        logger.warning("SWE metadata fetch failed from %s: %s", metadata_url, exc)
        return None

    try:
        return int(payload.get("tasks", {}).get("completed_up_to"))
    except (TypeError, ValueError):
        logger.warning("SWE metadata missing tasks.completed_up_to: %s", payload)
        return None


async def _resolve_task_range(
    env_config: env_cst.EnvironmentConfig,
    eval_config: SweInfiniteEvalConfig = DEFAULT_SWE_INFINITE_EVAL_CONFIG,
    task_selection_override: SweInfiniteTaskSelectionOverride | None = None,
) -> tuple[int, int]:
    task_selection_override = task_selection_override or SweInfiniteTaskSelectionOverride()
    task_id_min = (
        task_selection_override.task_id_min
        if task_selection_override.task_id_min is not None
        else env_config.task_id_min
    )
    if task_selection_override.task_id_max is not None:
        task_id_max = task_selection_override.task_id_max
    elif env_config.task_id_max > 0:
        task_id_max = env_config.task_id_max
    else:
        task_id_max = await _fetch_swe_completed_up_to(eval_config.metadata_url) or 0
        if task_id_max <= 0:
            raise ValueError("Could not resolve SWE task_id_max from metadata while EnvironmentConfig.task_id_max <= 0")

    if task_id_max < task_id_min:
        raise ValueError(f"Invalid SWE task range: min={task_id_min}, max={task_id_max}")
    return task_id_min, task_id_max


def _build_eval_list(base_seed: int, num_seeds: int, task_id_min: int, task_id_max: int) -> list[tuple[int, int]]:
    if num_seeds < 0:
        raise ValueError(f"Invalid SWE num_seeds={num_seeds}; expected a non-negative integer")
    task_range_size = task_id_max - task_id_min + 1
    if num_seeds > task_range_size:
        raise ValueError(
            f"Cannot sample {num_seeds} unique SWE task ids from range "
            f"{task_id_min}-{task_id_max} ({task_range_size} available)"
        )

    seed_generator = random.Random(base_seed)
    eval_seeds = [seed_generator.randint(1, 1_000_000) for _ in range(num_seeds)]
    task_ids = _build_random_task_ids(seed_generator, num_seeds, task_id_min, task_id_max)
    return list(zip(eval_seeds, task_ids, strict=True))


def _build_random_task_ids(
    rng: random.Random,
    num_tasks: int,
    task_id_min: int,
    task_id_max: int,
) -> list[int]:
    if num_tasks == 0:
        return []

    vetted_target_count = (num_tasks + 1) // 2
    eligible_vetted_ids = [
        task_id
        for task_id in SWE_VETTED_TASK_IDS
        if task_id_min <= task_id <= task_id_max
    ]
    vetted_count = min(vetted_target_count, len(eligible_vetted_ids))
    selected_vetted_ids = rng.sample(eligible_vetted_ids, vetted_count)
    selected_vetted_set = set(selected_vetted_ids)
    vetted_task_set = set(SWE_VETTED_TASK_IDS)

    random_count = num_tasks - vetted_count
    random_pool = [
        task_id
        for task_id in range(task_id_min, task_id_max + 1)
        if task_id not in selected_vetted_set and task_id not in vetted_task_set
    ]
    if len(random_pool) < random_count:
        random_pool = [
            task_id
            for task_id in range(task_id_min, task_id_max + 1)
            if task_id not in selected_vetted_set
        ]
    if len(random_pool) < random_count:
        raise ValueError(
            f"Cannot sample {num_tasks} unique SWE task ids from range "
            f"{task_id_min}-{task_id_max}"
        )

    task_ids = selected_vetted_ids + rng.sample(random_pool, random_count)
    rng.shuffle(task_ids)
    return task_ids


def _parse_task_ids(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return []
    task_ids = []
    for token in raw.replace(",", " ").split():
        try:
            task_id = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid SWE task id {token!r}; expected integers") from exc
        if task_id <= 0:
            raise ValueError(f"Invalid SWE task id {task_id}; expected a positive integer")
        task_ids.append(task_id)
    return task_ids


def _build_eval_list_for_task_ids(base_seed: int, task_ids: list[int]) -> list[tuple[int, int]]:
    seed_generator = random.Random(base_seed)
    return [
        (seed_generator.randint(1, 1_000_000), task_id)
        for task_id in task_ids
    ]


def _unwrap_affinetes_response(payload: dict) -> dict:
    if "status" in payload and payload.get("status") != "success":
        raise RuntimeError(f"Affinetes call failed: {payload}")
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        raise RuntimeError(f"Affinetes returned non-dict result: {result!r}")
    return result


async def _post_affinetes_evaluate(
    session: aiohttp.ClientSession,
    swe_server_url: str,
    payload: dict,
    task_timeout: int,
    eval_config: SweInfiniteEvalConfig = DEFAULT_SWE_INFINITE_EVAL_CONFIG,
) -> dict:
    call_path = eval_config.affinetes_call_path
    timeout = aiohttp.ClientTimeout(total=task_timeout + 30)
    url = f"{swe_server_url.rstrip('/')}{call_path}"
    if call_path == "/call":
        body = {"method": "evaluate", "args": [], "kwargs": payload}
    else:
        body = payload

    async with session.post(url, json=body, timeout=timeout, headers={"Connection": "close"}) as response:
        raw_text = await response.text()
        if response.status != 200:
            detail = f": {raw_text[:1000]}" if raw_text else ""
            raise RuntimeError(f"HTTP {response.status}{detail}")
        return _unwrap_affinetes_response(json.loads(raw_text))


async def _post_affinetes_evaluate_with_connect_retries(
    session: aiohttp.ClientSession,
    swe_server_url: str,
    payload: dict,
    task_timeout: int,
    eval_config: SweInfiniteEvalConfig = DEFAULT_SWE_INFINITE_EVAL_CONFIG,
) -> dict:
    max_attempts = max(1, eval_config.connect_max_attempts)
    backoff_seconds = max(0.0, eval_config.connect_retry_backoff_seconds)

    for attempt in range(1, max_attempts + 1):
        try:
            return await _post_affinetes_evaluate(session, swe_server_url, payload, task_timeout, eval_config)
        except aiohttp.ClientConnectorError as exc:
            if attempt == max_attempts:
                raise
            retry_delay = backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "eval_swe task_id=%s connection attempt %s/%s failed: %s; retrying in %.1fs",
                payload.get("task_id"),
                attempt,
                max_attempts,
                exc,
                retry_delay,
            )
            await asyncio.sleep(retry_delay)

    raise AssertionError("unreachable")


def _build_swe_payload(
    *,
    model: str,
    model_base_url: str,
    task_id: int,
    seed: int,
    temperature: float,
    task_timeout: int,
    eval_config: SweInfiniteEvalConfig = DEFAULT_SWE_INFINITE_EVAL_CONFIG,
) -> dict:
    model_api_key = eval_config.model_api_key or os.getenv(SWE_INFINITE_MODEL_API_KEY_ENV, DEFAULT_SWE_INFINITE_MODEL_API_KEY)
    payload: dict = {
        "model": model,
        "base_url": model_base_url,
        "api_key": model_api_key,
        "task_id": task_id,
        "timeout": task_timeout,
        "temperature": temperature,
        "seed": seed,
        "agent": SWE_INFINITE_AGENT_NAME,
        "max_iterations": eval_config.max_iterations,
    }
    if eval_config.collect_logprobs:
        payload["collect_logprobs"] = True
    return payload


async def _run_swe_evaluation(
    *,
    swe_server_url: str,
    model_base_url: str,
    inference_model_name: str,
    eval_list: list[tuple[int, int]],
    temperature: float,
    task_timeout: int,
    eval_config: SweInfiniteEvalConfig = DEFAULT_SWE_INFINITE_EVAL_CONFIG,
) -> float:
    all_results: list[dict] = []
    total_tasks = len(eval_list)
    concurrency = eval_config.max_concurrent_requests
    logger.info("eval_swe batch: %s tasks (concurrency=%s)", total_tasks, concurrency)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def evaluate_one(session: aiohttp.ClientSession, seed: int, task_id: int, index: int) -> dict:
        payload = _build_swe_payload(
            model=inference_model_name,
            model_base_url=model_base_url,
            task_id=task_id,
            seed=seed,
            temperature=temperature,
            task_timeout=task_timeout,
            eval_config=eval_config,
        )
        start = time.time()
        try:
            logger.info("eval_swe %s/%s start task_id=%s seed=%s", index + 1, total_tasks, task_id, seed)
            result = await _post_affinetes_evaluate_with_connect_retries(
                session,
                swe_server_url,
                payload,
                task_timeout,
                eval_config,
            )
            latency = float(result.get("time_taken", time.time() - start))
            score = float(result.get("score", 0.0))
            logger.info(
                "eval_swe %s/%s done task_id=%s score=%.6f latency_s=%.3f",
                index + 1,
                total_tasks,
                task_id,
                score,
                latency,
            )
            return {"task_id": task_id, "score": score, "time": latency}
        except Exception as exc:
            logger.warning(
                "eval_swe %s/%s failed task_id=%s: %s",
                index + 1,
                total_tasks,
                task_id,
                exc,
                exc_info=True,
            )
            return {"task_id": task_id, "score": 0.0, "time": 0.0}

    async def evaluate_with_limit(session: aiohttp.ClientSession, seed: int, task_id: int, index: int) -> dict:
        async with semaphore:
            return await evaluate_one(session, seed, task_id, index)

    session_timeout = eval_config.session_timeout_seconds
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=session_timeout)) as session:
        tasks = [
            asyncio.create_task(evaluate_with_limit(session, seed, task_id, index))
            for index, (seed, task_id) in enumerate(eval_list)
        ]
        done, pending = await asyncio.wait(tasks, timeout=session_timeout)
        for task in done:
            result = task.result()
            if isinstance(result, dict):
                all_results.append(result)
        if pending:
            logger.warning("eval_swe session timeout with %s/%s tasks complete", len(all_results), total_tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    if not all_results:
        logger.warning("eval_swe batch: no task results; returning 0.0")
        return 0.0
    avg = sum(result["score"] for result in all_results) / total_tasks
    logger.info("eval_swe batch: finished %s/%s tasks, avg_score=%.6f", len(all_results), total_tasks, avg)
    return avg


async def _prepare_sglang_command(
    *,
    model_repo: str,
    original_model: str,
    base_chain: list[str],
    base_seed: int,
) -> tuple[str, str, str]:
    detect_start = time.time()
    is_lora = await asyncio.to_thread(check_for_lora, model_repo, False)
    logger.info(
        "eval_setup LoRA detection in %.2fs: is_lora=%s",
        time.time() - detect_start,
        is_lora,
    )

    sglang_command = os.getenv("SGLANG_START_CMD")
    if sglang_command:
        logger.info("eval_setup SGLang: using SGLANG_START_CMD from environment")
        return model_repo, model_repo, sglang_command

    if is_lora:
        # Keep SWE model serving identical to PvP: SGLang loads the submitted adapter
        # natively over the base model and keeps the base tokenizer/EOS contract.
        # Merely shipping added_tokens.json is not a reason to synthesize a merged model;
        # doing so previously replaced the base tokenizer/EOS metadata only in SWE.
        model_path_for_sglang = await asyncio.to_thread(
            materialize_base_model,
            original_model,
            base_chain,
            "cand",
        )
        lora_name = "cand_trained_lora"
        prepared = PreparedModel(
            sglang_model_path=model_path_for_sglang,
            inference_name=f"{model_path_for_sglang}:{lora_name}",
            extra_sglang_args=(
                f"--enable-lora --lora-paths {lora_name}={model_repo} --lora-backend triton"
            ),
            tool_call_parser=tool_call_parser_for(model_path_for_sglang) if base_chain else None,
        )
        logger.info(
            "eval_setup model path: LoRA + SGLang native (base=%s lora=%s)",
            model_path_for_sglang,
            model_repo,
        )
    else:
        parser = tool_call_parser_for(model_repo, log_unmapped=False) or tool_call_parser_for(original_model)
        prepared = PreparedModel(
            sglang_model_path=model_repo,
            inference_name=model_repo,
            tool_call_parser=parser,
        )
        logger.info("eval_setup model path: full weights repo=%s", model_repo)

    port = int(os.getenv("SGLANG_PORT", "30000"))
    sglang_command = build_sglang_command(prepared, port=port, seed=base_seed)
    return prepared.inference_name, prepared.sglang_model_path, sglang_command


async def _run() -> None:
    sglang_proc = None
    sglang_log_task = None

    try:
        logger.info("eval_swe: start pid=%s EVAL_LOG_LEVEL=%s", os.getpid(), os.getenv("EVAL_LOG_LEVEL", "INFO"))
        models_raw = os.getenv("MODELS", "")
        model_repo = models_raw.split(",")[0].strip()
        if not model_repo:
            raise ValueError("MODELS is required and must contain a single repo")

        eval_config = load_swe_infinite_eval_config()
        task_selection_override = load_swe_infinite_task_selection_override()
        swe_server_url = os.getenv(SWE_INFINITE_SERVER_BASE_URL_ENV, "").strip()
        if not swe_server_url:
            raise ValueError(f"{SWE_INFINITE_SERVER_BASE_URL_ENV} is required for SWE Infinite evaluation")

        original_model = os.getenv("ORIGINAL_MODEL", model_repo)
        base_chain_raw = os.getenv("BASE_CHAIN", "")
        base_chain = json.loads(base_chain_raw) if base_chain_raw.strip() else []
        base_seed = int(os.getenv("EVAL_SEED", str(vcst.ENV_EVAL_DEFAULT_SEED)))
        temperature = float(os.getenv("ENV_EVAL_TEMPERATURE", str(vcst.ENV_EVAL_TEMPERATURE)))
        task_timeout = eval_config.task_timeout_seconds

        env_name = _parse_environment_name()
        env_config = env_cst.ENVIRONMENT_CONFIGS[env_name]
        explicit_task_ids = list(task_selection_override.task_ids)
        if explicit_task_ids:
            eval_list = _build_eval_list_for_task_ids(base_seed, explicit_task_ids)
            task_id_min = min(explicit_task_ids)
            task_id_max = max(explicit_task_ids)
            num_seeds = len(explicit_task_ids)
        else:
            task_id_min, task_id_max = await _resolve_task_range(env_config, eval_config, task_selection_override)
            num_seeds = (
                task_selection_override.num_seeds
                if task_selection_override.num_seeds is not None
                else env_config.num_seeds
            )
            eval_list = _build_eval_list(base_seed, num_seeds, task_id_min, task_id_max)

        logger.info(
            "eval_swe config: num_seeds=%s task_id_range=(%s,%s) explicit_task_ids=%s model_repo=%s "
            "original_model=%s eval_seed=%s temperature=%s base_chain=%s connect_max_attempts=%s "
            "connect_retry_backoff_seconds=%s",
            num_seeds,
            task_id_min,
            task_id_max,
            explicit_task_ids,
            model_repo,
            original_model,
            base_seed,
            temperature,
            base_chain,
            eval_config.connect_max_attempts,
            eval_config.connect_retry_backoff_seconds,
        )

        inference_model_name, model_path_for_sglang, sglang_command = await _prepare_sglang_command(
            model_repo=model_repo,
            original_model=original_model,
            base_chain=base_chain,
            base_seed=base_seed,
        )

        min_workspace = vcst.SGLANG_FLASHINFER_WORKSPACE_MIN_BYTES
        try:
            current_workspace = int(os.environ.get("SGLANG_FLASHINFER_WORKSPACE_SIZE", "0") or "0")
        except ValueError:
            current_workspace = 0
        if current_workspace < min_workspace:
            os.environ["SGLANG_FLASHINFER_WORKSPACE_SIZE"] = str(min_workspace)

        sglang_health_timeout = int(os.getenv("SGLANG_HEALTH_TIMEOUT", "1800"))
        sglang_base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        model_base_url = eval_config.model_base_url or os.getenv(SWE_INFINITE_MODEL_BASE_URL_ENV) or _with_v1(sglang_base_url)
        logger.info(
            "eval_setup launching SGLang: model_path=%s inference_model_name=%s public_model_base_url=%s",
            model_path_for_sglang,
            inference_model_name,
            model_base_url,
        )
        logger.info("eval_setup SGLang command: %s", sglang_command)

        sglang_proc = _start_process(sglang_command, "sglang")
        sglang_log_task = asyncio.create_task(_stream_logs(sglang_proc, "sglang"))
        await _wait_for_health(
            sglang_base_url,
            os.getenv("SGLANG_HEALTH_PATH", "/v1/models"),
            sglang_health_timeout,
            service_name="SGLang",
        )

        avg_score = await _run_swe_evaluation(
            swe_server_url=swe_server_url,
            model_base_url=model_base_url,
            inference_model_name=inference_model_name,
            eval_list=eval_list,
            temperature=temperature,
            task_timeout=task_timeout,
            eval_config=eval_config,
        )

        output = {model_repo: {"is_finetune": True, "eval_loss": avg_score}}
        result_path = Path(CONTAINER_EVAL_RESULTS_PATH)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(output), encoding="utf-8")
        logger.info("eval_swe: wrote results to %s avg_score=%.6f", result_path, avg_score)
    finally:
        stop_process(sglang_proc, "sglang")
        if sglang_log_task:
            sglang_log_task.cancel()


def main() -> int:
    configure_eval_logging()
    try:
        asyncio.run(_run())
        return 0
    except Exception as exc:
        logger.exception("SWE Infinite evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
