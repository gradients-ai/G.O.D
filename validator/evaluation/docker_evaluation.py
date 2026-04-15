import asyncio
import io
import json
import logging
import os
import re
import tarfile
import uuid
from uuid import UUID
import requests
import time
import random

from core import constants as cst
from core.models.payload_models import DockerEvaluationResults
from core.models.payload_models import DstackRunStatus
from core.models.payload_models import EvaluationResultImage
from core.models.payload_models import EvaluationResultText
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import InstructTextDatasetType
from validator.core import constants as vcst
from validator.db.database import PSQLDB
from validator.utils.logging import get_logger
from validator.utils.logging import get_environment_logger
from validator.evaluation.utils import (
    EVAL_RESULT_STATUS_PATH,
    load_eval_pair_state_for_models,
    persist_deployment_ids_for_repo,
)


logger = get_logger(__name__)
_EVAL_DB_WRITE_SEMAPHORE = asyncio.Semaphore(vcst.EVAL_DB_MAX_CONCURRENT_WRITES)


async def _db_read_with_retry(coro_factory, op_name: str):
    last_exc = None
    for attempt in range(1, vcst.EVAL_DB_RETRY_ATTEMPTS + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            delay = vcst.EVAL_DB_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            jitter = random.uniform(0.0, 0.3)
            if attempt < vcst.EVAL_DB_RETRY_ATTEMPTS:
                logger.warning(
                    f"DB read op '{op_name}' failed attempt {attempt}/{vcst.EVAL_DB_RETRY_ATTEMPTS}: {exc}; "
                    f"retrying in {delay + jitter:.2f}s"
                )
                await asyncio.sleep(delay + jitter)
            else:
                logger.error(f"DB read op '{op_name}' failed after {vcst.EVAL_DB_RETRY_ATTEMPTS} attempts: {exc}")
    raise last_exc


async def cleanup_resources(client):
    """Clean up Docker resources including containers, images, and volumes."""
    try:
        await asyncio.to_thread(client.containers.prune)
        await asyncio.to_thread(client.images.prune, filters={"dangling": True})
        await asyncio.to_thread(client.volumes.prune)
        logger.debug("Completed Docker resource cleanup")
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")


async def get_evaluation_results(container):
    archive_data = await asyncio.to_thread(container.get_archive, cst.CONTAINER_EVAL_RESULTS_PATH)
    tar_stream = archive_data[0]

    file_like_object = io.BytesIO()
    for chunk in tar_stream:
        file_like_object.write(chunk)
    file_like_object.seek(0)

    with tarfile.open(fileobj=file_like_object) as tar:
        members = tar.getnames()
        logger.debug(f"Tar archive members: {members}")
        eval_results_file = None
        for member_info in tar.getmembers():
            if member_info.name.endswith(("evaluation_results.json")):
                eval_results_file = tar.extractfile(member_info)
                break

        if eval_results_file is None:
            raise Exception("Evaluation results file not found in tar archive")

        eval_results_content = eval_results_file.read().decode("utf-8")
        return json.loads(eval_results_content)


def normalize_rewards_and_compute_loss(evaluation_results: dict) -> dict:
    """
    Normalize rewards across repos and compute final evaluation loss with KL penalty.

    Steps:
    1. For each reward type, normalize values across repos by dividing by max (after shifting if negative)
    2. Apply weights to normalized rewards (weights sum to 1)
    3. Sum weighted rewards to get final score in [0,1] range
    4. Apply KL penalty: score - (BETA_GRPO * kl_divergence)

    Special case: 2 repos with negative rewards map to [0.25, 0.75] to avoid extreme scores.

    Args:
        evaluation_results: Dict with model repos as keys and evaluation data as values

    Returns:
        Modified evaluation_results dict with updated eval_loss values
    """
    # Filter out non-repo keys (like model_params_count)
    repo_keys = [key for key in evaluation_results.keys() if key != "model_params_count"]

    if len(repo_keys) < 2:
        # Need at least 2 repos for meaningful normalization
        return evaluation_results

    reward_collections = {}
    for repo_key in repo_keys:
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):  # Skip error entries
            continue

        final_raw_rewards = repo_data.get('final_raw_rewards', {})

        for reward_name, reward_value in final_raw_rewards.items():
            if reward_name not in reward_collections:
                reward_collections[reward_name] = []
            reward_collections[reward_name].append((repo_key, reward_value))

    # Step 1: Normalize each reward type using shift + divide by max
    normalized_rewards_per_repo = {repo_key: {} for repo_key in repo_keys}

    for reward_name, repo_value_pairs in reward_collections.items():
        if len(repo_value_pairs) < 2:
            # Only one value, set to 1.0
            for repo_key, value in repo_value_pairs:
                normalized_rewards_per_repo[repo_key][reward_name] = 1.0
            continue

        values = [value for _, value in repo_value_pairs]
        min_value = min(values)

        # Check if we need to shift (have negatives)
        has_negatives = min_value < 0

        # Shift to positive if needed
        if has_negatives:
            shifted_values = [(repo, value - min_value) for repo, value in repo_value_pairs]
        else:
            shifted_values = repo_value_pairs

        # Find max of shifted values
        max_shifted = max(value for _, value in shifted_values)

        # Special case: 2 repos with negatives -> map to [0.25, 0.75]
        if len(repo_value_pairs) == 2 and has_negatives:
            sorted_pairs = sorted(shifted_values, key=lambda x: x[1])
            normalized_rewards_per_repo[sorted_pairs[0][0]][reward_name] = 0.25
            normalized_rewards_per_repo[sorted_pairs[1][0]][reward_name] = 0.75
        elif max_shifted > 0:
            # Normal case: divide by max
            for repo, shifted_value in shifted_values:
                normalized_rewards_per_repo[repo][reward_name] = shifted_value / max_shifted
        else:
            # All values are zero after shift (all were equal and negative or zero)
            for repo, _ in repo_value_pairs:
                normalized_rewards_per_repo[repo][reward_name] = 1.0

    # Step 2-3: Apply weights and sum (weights already sum to 1)
    final_scores = []

    for repo_key in repo_keys:
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):  # Skip error entries
            continue

        weights = repo_data.get('weights', {})
        normalized_rewards = normalized_rewards_per_repo.get(repo_key, {})

        # Calculate weighted sum
        weighted_sum = 0.0
        for reward_name, normalized_value in normalized_rewards.items():
            weight = weights.get(reward_name, 1.0)
            weighted_sum += normalized_value * weight

        final_scores.append(weighted_sum)

    # Step 4: Apply KL penalty and update eval_loss
    for i, repo_key in enumerate(repo_keys):
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):  # Skip error entries
            continue

        if i < len(final_scores):
            kl_divergence = repo_data.get('kl_divergence', 0.0)
            # Final score: weighted_sum - BETA_GRPO * kl_divergence
            new_eval_loss = final_scores[i] - (vcst.BETA_GRPO * kl_divergence)
            repo_data['eval_loss'] = new_eval_loss

    return evaluation_results


def process_evaluation_results(results: dict, is_image: bool = False) -> DockerEvaluationResults:
    model_params_count = results.pop("model_params_count", 0)

    processed_results = {}
    for repo, result in results.items():
        if isinstance(result, str) and not isinstance(result, dict):
            processed_results[repo] = Exception(result)
        else:
            if is_image:
                result["is_finetune"] = True
                processed_results[repo] = EvaluationResultImage.model_validate(result)
            else:
                processed_results[repo] = EvaluationResultText.model_validate(result)

    return DockerEvaluationResults(
        results=processed_results,
        base_model_params_count=model_params_count
    )


def _dstack_base() -> tuple[str, str, dict[str, str]]:
    """dstack API base URL, project name, and JSON auth headers."""
    dstack_url = os.getenv("DSTACK_URL", "").rstrip("/")
    dstack_token = os.getenv("DSTACK_TOKEN", "")
    dstack_project = os.getenv("DSTACK_PROJECT", "main")
    if not dstack_url or not dstack_token:
        raise ValueError("DSTACK_URL and DSTACK_TOKEN must be set for dstack evaluation")
    headers = {
        "Authorization": f"Bearer {dstack_token}",
        "Content-Type": "application/json",
    }
    return dstack_url, dstack_project, headers


async def _dstack_post(endpoint_fmt: str, json_body: dict, *, timeout: int = 60) -> requests.Response:
    dstack_url, dstack_project, headers = _dstack_base()
    request_url = f"{dstack_url}{endpoint_fmt.format(project=dstack_project)}"
    return await asyncio.to_thread(requests.post, request_url, headers=headers, json=json_body, timeout=timeout)


async def _dstack_runs_api(
    op: str,
    *,
    apply_plan: dict | None = None,
    run_name: str | None = None,
    timeout: int = 60,
) -> dict | str | None:
    """Single entry for dstack runs/apply, runs/get, and runs/stop."""
    if op == "apply":
        if apply_plan is None:
            raise ValueError("apply_plan required for apply")
        response = await _dstack_post(vcst.DSTACK_RUNS_APPLY_ENDPOINT, apply_plan, timeout=timeout)
        if response.status_code >= 400:
            request_url = response.url
            logger.error(
                "dstack runs/apply HTTP %s for %s: %s",
                response.status_code,
                request_url,
                response.text[:4000],
            )
        response.raise_for_status()
        payload = response.json()
        name = payload.get("run_spec", {}).get("run_name") or payload.get("run_name")
        logger.info(
            "dstack runs/apply ok run_name=%s project=%s",
            name,
            os.getenv("DSTACK_PROJECT", "main"),
        )
        return name
    if op == "get":
        if not run_name:
            raise ValueError("run_name required for get")
        response = await _dstack_post(vcst.DSTACK_RUNS_GET_ENDPOINT, {"run_name": run_name}, timeout=timeout)
        response.raise_for_status()
        return response.json()
    if op == "stop":
        if not run_name:
            raise ValueError("run_name required for stop")
        response = await _dstack_post(
            vcst.DSTACK_RUNS_STOP_ENDPOINT,
            {"runs_names": [run_name], "abort": True},
            timeout=timeout,
        )
        if response.status_code >= 400:
            logger.warning("Failed to stop dstack run %s: %s %s", run_name, response.status_code, response.text)
        return None
    raise ValueError(f"unknown dstack op: {op}")


def _dstack_service_url(dstack_url: str, project: str, run_name: str, run_details: dict | None = None) -> str:
    """Resolve public base URL for the service (no trailing slash). Prefer API ``service.url`` when present."""
    if isinstance(run_details, dict):
        paths: list[str] = []
        svc = run_details.get("service")
        if isinstance(svc, dict):
            u = svc.get("url")
            if isinstance(u, str) and u:
                paths.append(u)
        run_spec = run_details.get("run_spec")
        if isinstance(run_spec, dict):
            svc2 = run_spec.get("service")
            if isinstance(svc2, dict):
                u2 = svc2.get("url")
                if isinstance(u2, str) and u2:
                    paths.append(u2)
        for service_path in paths:
            return f"{dstack_url}{service_path.rstrip('/')}"
    return f"{dstack_url}/proxy/services/{project}/{run_name}"


def _create_dstack_service_request(
    *,
    run_name: str,
    image: str,
    command: str,
    env: dict[str, str],
    gpu_count: int,
    gpu_models: list[str],
    min_gpu_memory_gb: int,
) -> dict:
    gpu_resource: dict = {
        "name": gpu_models,
        "count": {"min": gpu_count, "max": gpu_count},
    }
    if min_gpu_memory_gb > 0:
        gpu_resource["memory"] = f"{min_gpu_memory_gb}GB.."

    return {
        "plan": {
            "run_spec": {
                "run_name": run_name,
                "configuration": {
                    "type": "service",
                    "name": run_name,
                    "image": image,
                    "commands": [command],
                    "env": env,
                    "port": vcst.EVAL_SERVICE_PORT,
                    "auth": False,
                    "gateway": False,
                    "strip_prefix": True,
                    "backends": ["runpod"],
                    "resources": {
                        "gpu": gpu_resource,
                        "disk": {"size": "100GB"},
                    },
                    "max_duration": vcst.EVAL_DSTACK_TTL_SECONDS,
                },
            }
        },
        "force": False,
    }


async def _wait_for_service_running(
    run_name: str,
    repo: str,
    eval_logger: logging.Logger,
    max_wait_seconds: int = vcst.EVAL_DSTACK_TIMEOUT,
) -> tuple[str, dict | None]:
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        try:
            run_details = await _dstack_runs_api("get", run_name=run_name)
            if not isinstance(run_details, dict):
                raise TypeError(f"expected dict from dstack get, got {type(run_details)}")
            run_status = DstackRunStatus.model_validate(run_details)
            status_str = run_status.get_status()
            eval_logger.info(f"[{repo}] dstack run {run_name} status={status_str}")
            if run_status.is_running():
                return "running", run_details
            if run_status.got_no_offers():
                return "no_offers", run_details
            if run_status.is_failed():
                return f"failed:{status_str}", run_details
        except Exception as e:
            eval_logger.warning(f"[{repo}] error waiting for running state: {e}")
        await asyncio.sleep(15)
    return "running_timeout", None


async def _poll_dstack_service_result(
    service_url: str,
    repo: str,
    eval_logger: logging.Logger,
    *,
    run_name: str,
    poll_interval_seconds: int = vcst.EVAL_DSTACK_POLL_INTERVAL_SECONDS,
    max_poll_seconds: int = vcst.EVAL_DSTACK_MAX_POLL_SECONDS,
) -> dict | str:
    dstack_url, dstack_project, poll_headers = _dstack_base()
    current_base = service_url.rstrip("/")

    def _result_poll_urls(service_base: str) -> list[str]:
        b = service_base.rstrip("/")
        bl = b.lower()
        if bl.endswith("/result"):
            candidates = [b, f"{b}/"]
        else:
            candidates = [f"{b}{EVAL_RESULT_STATUS_PATH}", f"{b}{EVAL_RESULT_STATUS_PATH}/"]
        out: list[str] = []
        seen: set[str] = set()
        for u in candidates:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    started_monotonic = time.monotonic()
    deadline = started_monotonic + max_poll_seconds
    next_poll_at = started_monotonic
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now < next_poll_at:
            await asyncio.sleep(next_poll_at - now)
        try:
            response = None
            result_url = f"{current_base}{EVAL_RESULT_STATUS_PATH}"
            for candidate_url in _result_poll_urls(current_base):
                result_url = candidate_url
                response = await asyncio.to_thread(
                    requests.get,
                    candidate_url,
                    headers=poll_headers,
                    timeout=30,
                )
                if response.status_code == 200:
                    break
                if response.status_code != 404:
                    break
            assert response is not None
            if response.status_code == 200:
                payload = response.json()
                status = payload.get("status")
                if status == "completed":
                    result = payload.get("result")
                    if isinstance(result, dict):
                        eval_logger.info(f"[{repo}] Evaluation completed and result payload received.")
                        return result
                    return f"Completed but result payload invalid: {result}"
                if status == "failed":
                    return payload.get("error", "dstack eval reported failure")
                eval_logger.info(f"[{repo}] Poll ping: status={status}.")
            elif response.status_code == 404:
                eval_logger.info(f"[{repo}] Poll got HTTP 404 from {result_url}; refreshing run + service URL")
                logger.info("[%s] dstack poll 404 for %s; refreshing service URL", repo, result_url)
                try:
                    details = await _dstack_runs_api("get", run_name=run_name)
                    if not isinstance(details, dict):
                        raise TypeError(f"expected dict from dstack get, got {type(details)}")
                    refreshed = _dstack_service_url(dstack_url, dstack_project, run_name, details)
                    if refreshed.rstrip("/") != current_base:
                        current_base = refreshed.rstrip("/")
                        eval_logger.info(f"[{repo}] using dstack service URL: {current_base}")
                        logger.info("[%s] dstack service URL now %s", repo, current_base)
                        next_poll_at = time.monotonic()
                        continue
                except Exception as refresh_exc:
                    eval_logger.warning(f"[{repo}] could not refresh dstack run for service URL: {refresh_exc}")
            else:
                eval_logger.info(f"[{repo}] Poll got HTTP {response.status_code} from {result_url}")
        except Exception as e:
            eval_logger.error(f"[{repo}] error polling dstack service result: {e}", exc_info=True)
        eval_logger.info(f"[{repo}] result not ready yet, polling again in {poll_interval_seconds}s...")
        next_poll_at += poll_interval_seconds
    return f"Timed out waiting for result after {max_poll_seconds}s"


async def _run_single_dstack_eval_repo(
    *,
    repo: str,
    model_name: str,
    task_type: str,
    image: str,
    command: str,
    env: dict[str, str],
    gpu_count: int,
    gpu_models: list[str],
    min_gpu_memory_gb: int,
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    repo_to_hotkey: dict[str, str],
    existing_deployment_name: str | None = None,
) -> dict | str:
    """Run one repo eval with retries using dstack services."""
    eval_id = str(uuid.uuid4())
    eval_logger = get_environment_logger(
        name=f"dstack-{repo.split('/')[-1]}-{eval_id[:8]}",
        repo_id=repo,
        eval_id=eval_id,
        model=model_name,
        task_type=task_type,
        enable_console_output=os.getenv("EVAL_CONSOLE_LOG", "").strip() in ("1", "true", "yes"),
    )

    async def _db_call_with_retry(coro_factory, op_name: str):
        last_exc = None
        for attempt in range(1, vcst.EVAL_DB_RETRY_ATTEMPTS + 1):
            try:
                return await coro_factory()
            except Exception as exc:
                last_exc = exc
                delay = vcst.EVAL_DB_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, 0.3)
                if attempt < vcst.EVAL_DB_RETRY_ATTEMPTS:
                    eval_logger.warning(
                        f"[{repo}] DB op '{op_name}' failed attempt {attempt}/{vcst.EVAL_DB_RETRY_ATTEMPTS}: {exc}; "
                        f"retrying in {delay + jitter:.2f}s"
                    )
                    await asyncio.sleep(delay + jitter)
                else:
                    eval_logger.error(
                        f"[{repo}] DB op '{op_name}' failed after {vcst.EVAL_DB_RETRY_ATTEMPTS} attempts: {exc}"
                    )
        raise last_exc

    if existing_deployment_name:
        try:
            eval_logger.info(f"[{repo}] resume: checking dstack run {existing_deployment_name}")
            state, run_details = await _wait_for_service_running(existing_deployment_name, repo, eval_logger)
            if state == "running" and isinstance(run_details, dict):
                dstack_url, dstack_project, _ = _dstack_base()
                service_url = _dstack_service_url(dstack_url, dstack_project, existing_deployment_name, run_details)
                eval_logger.info(f"[{repo}] resuming polling dstack service {service_url}")
                result = await _poll_dstack_service_result(
                    service_url,
                    repo,
                    eval_logger=eval_logger,
                    run_name=existing_deployment_name,
                )
                if isinstance(result, dict):
                    await _dstack_runs_api("stop", run_name=existing_deployment_name)
                    return result
                return str(result) if result else "Resume poll returned empty"
        except Exception as e:
            eval_logger.error(f"[{repo}] resume failed, redeploying: {e}", exc_info=True)

    for attempt in range(1, vcst.EVAL_DSTACK_MAX_RETRIES + 1):
        run_body = uuid.uuid4().hex[:40]
        run_name = f"e{run_body}"
        assert re.match(r"^[a-z][a-z0-9-]{1,40}$", run_name), run_name
        try:
            eval_logger.info(f"[{repo}] starting dstack evaluation attempt {attempt}/{vcst.EVAL_DSTACK_MAX_RETRIES}")
            logger.info("[%s] starting dstack evaluation attempt %s/%s", repo, attempt, vcst.EVAL_DSTACK_MAX_RETRIES)
            await asyncio.sleep(random.uniform(0.0, 0.25))
            async with _EVAL_DB_WRITE_SEMAPHORE:
                await _db_call_with_retry(
                    lambda: persist_deployment_ids_for_repo(
                        task_id,
                        psql_db,
                        repo_to_hotkey,
                        repo,
                        run_name,
                        None,
                    ),
                    "persist_deployment_ids_for_repo(pre-deploy)",
                )
            dstack_payload = _create_dstack_service_request(
                run_name=run_name,
                image=image,
                command=command,
                env=env,
                gpu_count=gpu_count,
                gpu_models=gpu_models,
                min_gpu_memory_gb=min_gpu_memory_gb,
            )
            submitted_name = await _dstack_runs_api("apply", apply_plan=dstack_payload, timeout=120)
            if isinstance(submitted_name, str) and submitted_name:
                run_name = submitted_name
            eval_logger.info(f"[{repo}] dstack service submitted: {run_name}")
            logger.info("[%s] dstack service submitted: %s", repo, run_name)

            state, run_details = await _wait_for_service_running(run_name, repo, eval_logger)
            if state == "no_offers":
                eval_logger.warning(f"[{repo}] no offers for run {run_name}; retrying in 15 minutes")
                await _dstack_runs_api("stop", run_name=run_name)
                await asyncio.sleep(vcst.EVAL_DSTACK_RETRY_DELAY_SECONDS)
                continue
            if state != "running":
                raise RuntimeError(f"Run {run_name} failed before running ({state})")

            dstack_url, dstack_project, _ = _dstack_base()
            service_url = _dstack_service_url(dstack_url, dstack_project, run_name, run_details)
            eval_logger.info(f"[{repo}] service is running at {service_url}, starting /result polling")
            result = await _poll_dstack_service_result(
                service_url,
                repo,
                eval_logger=eval_logger,
                run_name=run_name,
            )
            if isinstance(result, dict):
                await _dstack_runs_api("stop", run_name=run_name)
                return result
            if "Timed out" in str(result):
                logger.error(f"[{repo}] poll timeout, skipping retries: {result}")
                await _dstack_runs_api("stop", run_name=run_name)
                return result
            raise RuntimeError(str(result))
        except Exception as e:
            remaining = vcst.EVAL_DSTACK_MAX_RETRIES - attempt
            eval_logger.error(
                f"[{repo}] attempt {attempt}/{vcst.EVAL_DSTACK_MAX_RETRIES} failed: {e}",
                exc_info=True,
            )
            logger.error(
                "[%s] dstack evaluation attempt %s/%s failed: %s",
                repo,
                attempt,
                vcst.EVAL_DSTACK_MAX_RETRIES,
                e,
                exc_info=True,
            )
            if isinstance(e, ValueError) and "DSTACK_" in str(e):
                return f"Evaluation failed: {e}"
            if remaining > 0:
                delay_sec = vcst.EVAL_DSTACK_ERROR_RETRY_SECONDS
                eval_logger.info(
                    f"[{repo}] retrying in {delay_sec}s ({remaining} attempts remaining)"
                )
                logger.info("[%s] retrying dstack eval in %ss (%s attempts left)", repo, delay_sec, remaining)
                await asyncio.sleep(delay_sec)
            else:
                return f"Evaluation failed after {vcst.EVAL_DSTACK_MAX_RETRIES} attempts: {e}"
        finally:
            try:
                await _dstack_runs_api("stop", run_name=run_name)
            except Exception:
                pass

    return "Evaluation failed"


async def _run_dstack_eval_repos(
    *,
    repos: list[str],
    model_name: str,
    task_type: str,
    image: str,
    command: str,
    build_env_for_repo,
    gpu_count: int,
    gpu_models: list[str],
    min_gpu_memory_gb: int,
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    repo_to_hotkey: dict[str, str],
    deployment_ids_by_repo: dict[str, str] | None = None,
) -> dict[str, dict | str]:
    deployment_ids_by_repo = deployment_ids_by_repo or {}
    task_results = await asyncio.gather(
        *[
            _run_single_dstack_eval_repo(
                repo=repo,
                model_name=model_name,
                task_type=task_type,
                image=image,
                command=command,
                env=build_env_for_repo(repo),
                gpu_count=gpu_count,
                gpu_models=gpu_models,
                min_gpu_memory_gb=min_gpu_memory_gb,
                task_id=task_id,
                psql_db=psql_db,
                repo_to_hotkey=repo_to_hotkey,
                existing_deployment_name=deployment_ids_by_repo.get(repo) if isinstance(deployment_ids_by_repo.get(repo), str) else None,
            )
            for repo in repos
        ],
        return_exceptions=True,
    )
    out: dict[str, dict | str] = {}
    for repo, result in zip(repos, task_results):
        if isinstance(result, Exception):
            out[repo] = f"Evaluation failed: {result}"
        else:
            out[repo] = result
    return out


async def run_evaluation_dstack_text(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType | EnvironmentDatasetType,
    file_format: FileFormat,
    num_gpus: int,
    eval_seed: int | None = None,
    task_id: UUID | None = None,
    psql_db: PSQLDB | None = None,
    parallel_eval_slots: int = 1,
) -> DockerEvaluationResults:
    if parallel_eval_slots < 1:
        raise ValueError("parallel_eval_slots must be >= 1")
    if parallel_eval_slots > 1 and (task_id is not None or psql_db is not None):
        raise ValueError("parallel_eval_slots>1 requires task_id=None and psql_db=None (DB state is keyed by HF repo id)")

    deployment_ids_by_repo = {}
    db_deployment_ids_by_repo, repo_to_hotkey = await _db_read_with_retry(
        lambda: load_eval_pair_state_for_models(task_id, psql_db, models),
        "load_eval_pair_state_for_models",
    )
    for repo, dep_info in db_deployment_ids_by_repo.items():
        deployment_ids_by_repo.setdefault(repo, dep_info)
    task_type = type(dataset_type).__name__
    is_environment_eval = isinstance(dataset_type, EnvironmentDatasetType)
    dstack_eval_image = vcst.ENV_EVAL_IMAGE if is_environment_eval else cst.VALIDATOR_DOCKER_IMAGE
    if isinstance(dataset_type, (InstructTextDatasetType, ChatTemplateDatasetType)):
        command = "python -m validator.evaluation.eval_instruct_text"
    elif isinstance(dataset_type, DpoDatasetType):
        command = "python -m validator.evaluation.eval_dpo"
    elif isinstance(dataset_type, GrpoDatasetType):
        if parallel_eval_slots > 1:
            raise ValueError("parallel_eval_slots>1 is not supported for GRPO evaluations")
        return await run_evaluation_dstack_grpo(
            dataset, models, original_model, dataset_type, file_format, num_gpus,
            task_id=task_id,
            psql_db=psql_db,
            deployment_ids_by_repo=deployment_ids_by_repo,
        )
    elif isinstance(dataset_type, EnvironmentDatasetType):
        command = "python -m validator.evaluation.eval_environment"
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")

    if parallel_eval_slots > 1:
        if len(models) != 1:
            raise ValueError("parallel_eval_slots>1 requires exactly one HF repo in models (same checkpoint, N parallel services)")
        hf_repo = models[0]
        eval_repos = [f"{hf_repo}#eval_slot{i}" for i in range(parallel_eval_slots)]
    else:
        hf_repo = ""
        eval_repos = list(models)

    if not is_environment_eval and not dataset.startswith("http://") and not dataset.startswith("https://"):
        raise ValueError(
            "dstack text eval expects dataset to be an S3/HTTP URL. "
            "Use validator.evaluation.local_evaluation.run_evaluation_docker_text for local file paths."
        )
    dataset_type_str = dataset_type.model_dump_json()

    base_env = {
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
        "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "EVAL_SERVICE_MODE": "1",
    }
    env_eval_seed_base: int | None = None
    if is_environment_eval:
        env_name = dataset_type.environment_name
        if env_name not in vcst.ENVIRONMENTS:
            raise ValueError(f"Environment '{env_name}' not found. Supported: {list(vcst.ENVIRONMENTS.keys())}")
        env_eval_seed_base = eval_seed if eval_seed is not None else vcst.ENV_EVAL_DEFAULT_SEED
        base_env["ENVIRONMENT_NAME"] = env_name
        base_env["EVAL_SEED"] = str(env_eval_seed_base)
        base_env["ENV_EVAL_TEMPERATURE"] = str(vcst.ENV_EVAL_TEMPERATURE)
        base_env["ENV_SERVER_CMD"] = vcst.ENV_SERVER_CMD_DEFAULT
        if os.getenv("ENV_EVAL_NUM_SEEDS"):
            base_env["ENV_EVAL_NUM_SEEDS"] = os.getenv("ENV_EVAL_NUM_SEEDS", "")

    logger.debug(
        "Running dstack %s evaluation for %s (parallel_eval_slots=%s)",
        task_type,
        models if parallel_eval_slots == 1 else f"{models[0]} x{parallel_eval_slots}",
        parallel_eval_slots,
    )

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_env)
        if parallel_eval_slots > 1:
            repo_env["MODELS"] = hf_repo
            if is_environment_eval and env_eval_seed_base is not None and "#eval_slot" in repo:
                slot = int(repo.rsplit("#eval_slot", 1)[-1])
                repo_env["EVAL_SEED"] = str((env_eval_seed_base + slot) % (2**31))
        else:
            repo_env["MODELS"] = repo
        if not is_environment_eval:
            repo_env["DATASET_URL"] = dataset
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    repo_results = await _run_dstack_eval_repos(
        repos=eval_repos,
        model_name=original_model,
        task_type=task_type,
        image=dstack_eval_image,
        command=command,
        build_env_for_repo=build_env_for_repo,
        gpu_count=vcst.EVAL_REMOTE_GPU_COUNT,
        gpu_models=vcst.DSTACK_EVAL_GPU_MODELS,
        min_gpu_memory_gb=vcst.DSTACK_EVAL_MIN_GPU_MEMORY_GB,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey=repo_to_hotkey,
        deployment_ids_by_repo=deployment_ids_str,
    )

    evaluation_results: dict[str, dict | str] = {}
    model_params_count = 0
    for repo in eval_repos:
        raw_result = repo_results.get(repo)
        if not isinstance(raw_result, dict):
            evaluation_results[repo] = str(raw_result)
            continue

        if raw_result.get("model_params_count") and model_params_count == 0:
            model_params_count = raw_result["model_params_count"]

        hf_key = hf_repo if parallel_eval_slots > 1 else repo
        if hf_key in raw_result:
            evaluation_results[repo] = raw_result[hf_key]
        else:
            candidate_keys = [k for k in raw_result.keys() if k != "model_params_count"]
            if len(candidate_keys) == 1:
                evaluation_results[repo] = raw_result[candidate_keys[0]]
            else:
                evaluation_results[repo] = f"Evaluation failed: missing result key for repo {repo}"

    if model_params_count:
        evaluation_results["model_params_count"] = model_params_count

    return process_evaluation_results(evaluation_results, is_image=False)


async def run_evaluation_dstack_grpo(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: GrpoDatasetType,
    file_format: FileFormat,
    num_gpus: int,
    task_id: UUID | None = None,
    psql_db: PSQLDB | None = None,
    deployment_ids_by_repo: dict[str, str | dict[str, str]] | None = None,
) -> DockerEvaluationResults:
    """Run GRPO evaluation on dstack with separate service runs per repo."""
    deployment_ids_by_repo = deployment_ids_by_repo or {}
    db_deployment_ids_by_repo, repo_to_hotkey = await _db_read_with_retry(
        lambda: load_eval_pair_state_for_models(task_id, psql_db, models),
        "load_eval_pair_state_for_models",
    )
    for repo, dep_info in db_deployment_ids_by_repo.items():
        deployment_ids_by_repo.setdefault(repo, dep_info)
    command = "python -m validator.evaluation.eval_grpo"
    if not dataset.startswith("http://") and not dataset.startswith("https://"):
        raise ValueError(
            "dstack GRPO eval expects dataset to be an S3/HTTP URL. "
            "Use validator.evaluation.local_evaluation.run_evaluation_docker_grpo for local file paths."
        )
    dataset_type_str = dataset_type.model_dump_json()

    base_environment = {
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
        "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "EVAL_SERVICE_MODE": "1",
    }

    logger.debug(f"Starting dstack GRPO evaluation for {len(models)} repos: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_environment)
        repo_env["MODELS"] = repo
        repo_env["DATASET_URL"] = dataset
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    repo_results = await _run_dstack_eval_repos(
        repos=models,
        model_name=original_model,
        task_type="grpo",
        image=cst.VALIDATOR_DOCKER_IMAGE,
        command=command,
        build_env_for_repo=build_env_for_repo,
        gpu_count=vcst.EVAL_REMOTE_GPU_COUNT,
        gpu_models=vcst.DSTACK_EVAL_GPU_MODELS,
        min_gpu_memory_gb=vcst.DSTACK_EVAL_MIN_GPU_MEMORY_GB,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey=repo_to_hotkey,
        deployment_ids_by_repo=deployment_ids_str,
    )

    evaluation_results: dict[str, dict | str | int] = {}
    model_params_count = 0
    for repo in models:
        raw_result = repo_results.get(repo)
        if not isinstance(raw_result, dict):
            evaluation_results[repo] = str(raw_result)
            continue

        if raw_result.get("model_params_count") and model_params_count == 0:
            model_params_count = raw_result["model_params_count"]

        if repo in raw_result:
            evaluation_results[repo] = raw_result[repo]
        else:
            candidate_keys = [k for k in raw_result.keys() if k != "model_params_count"]
            if len(candidate_keys) == 1:
                evaluation_results[repo] = raw_result[candidate_keys[0]]
            else:
                evaluation_results[repo] = f"Evaluation failed: missing result key for repo {repo}"

    if model_params_count:
        evaluation_results["model_params_count"] = model_params_count

    evaluation_results = normalize_rewards_and_compute_loss(evaluation_results)
    logger.debug(f"Grpo evaluation results post normalization: {evaluation_results}")
    return process_evaluation_results(evaluation_results, is_image=False)


async def run_evaluation_dstack_image(
    test_split_url: str,
    original_model_repo: str,
    models: list[str],
    model_type: ImageModelType,
    num_gpus: int,
    task_id: UUID | None = None,
    psql_db: PSQLDB | None = None,
) -> DockerEvaluationResults:
    deployment_ids_by_repo = {}
    db_deployment_ids_by_repo, repo_to_hotkey = await _db_read_with_retry(
        lambda: load_eval_pair_state_for_models(task_id, psql_db, models),
        "load_eval_pair_state_for_models",
    )
    for repo, dep_info in db_deployment_ids_by_repo.items():
        deployment_ids_by_repo.setdefault(repo, dep_info)
    if not test_split_url.startswith("http://") and not test_split_url.startswith("https://"):
        raise ValueError("dstack image eval expects TEST_SPLIT_URL to be an S3/HTTP URL.")
    command = vcst.DIFFUSION_EVAL_CONTAINER_START

    base_env = {
        "ORIGINAL_MODEL_REPO": original_model_repo,
        "MODEL_TYPE": model_type.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
        "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
        "EVAL_SERVICE_MODE": "1",
    }

    logger.debug(f"Starting dstack image evaluation for {len(models)} repos: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_env)
        repo_env["MODELS"] = repo
        repo_env["TEST_SPLIT_URL"] = test_split_url
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    repo_results = await _run_dstack_eval_repos(
        repos=models,
        model_name=original_model_repo,
        task_type="image",
        image=cst.VALIDATOR_DOCKER_IMAGE_DIFFUSION,
        command=command,
        build_env_for_repo=build_env_for_repo,
        gpu_count=vcst.EVAL_REMOTE_GPU_COUNT,
        gpu_models=vcst.DSTACK_EVAL_GPU_MODELS,
        min_gpu_memory_gb=vcst.DSTACK_EVAL_MIN_GPU_MEMORY_GB,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey=repo_to_hotkey,
        deployment_ids_by_repo=deployment_ids_str,
    )

    evaluation_results: dict[str, dict | str] = {}
    model_params_count = 0
    for repo in models:
        raw_result = repo_results.get(repo)
        if not isinstance(raw_result, dict):
            evaluation_results[repo] = str(raw_result)
            continue

        if raw_result.get("model_params_count") and model_params_count == 0:
            model_params_count = raw_result["model_params_count"]

        if repo in raw_result:
            evaluation_results[repo] = raw_result[repo]
        else:
            candidate_keys = [k for k in raw_result.keys() if k != "model_params_count"]
            if len(candidate_keys) == 1:
                evaluation_results[repo] = raw_result[candidate_keys[0]]
            else:
                evaluation_results[repo] = f"Evaluation failed: missing result key for repo {repo}"

    if model_params_count:
        evaluation_results["model_params_count"] = model_params_count

    return process_evaluation_results(evaluation_results, is_image=True)


# Deprecated aliases (same dstack-backed remote eval; use run_evaluation_dstack_*).
run_evaluation_basilica_text = run_evaluation_dstack_text
run_evaluation_basilica_grpo = run_evaluation_dstack_grpo
run_evaluation_basilica_image = run_evaluation_dstack_image
