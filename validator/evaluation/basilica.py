import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass
from typing import Callable
from uuid import UUID

import basilica
import requests

import validator.evaluation.constants as vcst
from core.logging import get_environment_logger
from core.logging import get_logger
from core.logging import update_environment_logger_labels
from validator.db.database import PSQLDB
from validator.db.sql import tasks as tasks_sql
from validator.evaluation.basilica_deployments import EVAL_RESULT_STATUS_PATH
from validator.evaluation.basilica_deployments import deployment_is_healthy
from validator.evaluation.db_utils import persist_deployment_ids_for_repo
from validator.evaluation.evaluation_logging import _log_eval_step
from validator.evaluation.evaluation_logging import log_basilica_logs_block


logger = get_logger(__name__)
_EVAL_DB_WRITE_SEMAPHORE = asyncio.Semaphore(vcst.EVAL_DB_MAX_CONCURRENT_WRITES)


def _deployment_url(deployment) -> str | None:
    return getattr(deployment, "url", None)


class EvaluationRetryableError(RuntimeError):
    """Evaluation should be retried later without marking rows as failed."""


class EvaluationCapacityUnavailable(EvaluationRetryableError):
    pass


class DeploymentNotReadyError(EvaluationRetryableError):
    pass


@dataclass
class _BasilicaEvalContext:
    repo: str
    eval_logger: logging.Logger
    deleted_deployment_names: set[str]
    log_eval_step: Callable[..., None]


async def _db_call_with_retry(coro_factory, op_name: str, eval_logger: logging.Logger, repo: str):
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
                eval_logger.error(f"[{repo}] DB op '{op_name}' failed after {vcst.EVAL_DB_RETRY_ATTEMPTS} attempts: {exc}")
    raise last_exc


async def _poll_basilica_result(
    deployment,
    repo: str,
    eval_logger: logging.Logger,
    poll_interval_seconds: int = vcst.EVAL_BASILICA_POLL_INTERVAL_SECONDS,
    max_poll_seconds: int = vcst.EVAL_BASILICA_MAX_POLL_SECONDS,
) -> dict | str:
    """Poll Basilica /result endpoint. Handles status: completed, failed, running, in_progress."""
    started_monotonic = time.monotonic()
    deadline = started_monotonic + max_poll_seconds
    next_poll_at = started_monotonic
    deployment_name = getattr(deployment, "name", "unknown")
    consecutive_failures = 0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now < next_poll_at:
            await asyncio.sleep(next_poll_at - now)
        poll_failed = False
        try:
            await asyncio.to_thread(log_basilica_logs_block, eval_logger, repo, deployment_name, deployment)
            response = await asyncio.to_thread(
                requests.get,
                f"{deployment.url}{EVAL_RESULT_STATUS_PATH}",
                timeout=30,
            )
            if response.status_code == 200:
                payload = response.json()
                status = payload.get("status")
                if status == "completed":
                    result = payload.get("result")
                    if isinstance(result, dict):
                        eval_logger.info(f"[{repo}] Poll successful. Evaluation completed and result payload received.")
                        return result
                    return f"Completed but result payload invalid: {result}"
                if status == "failed":
                    return payload.get("error", "Basilica eval reported failure")
                eval_logger.info(f"[{repo}] Poll ping: status={status}.")
            else:
                poll_failed = True
                eval_logger.warning(f"[{repo}] poll got HTTP {response.status_code} from {deployment_name}")
        except Exception as e:
            poll_failed = True
            eval_logger.error(f"[{repo}] error polling Basilica result: {e}", exc_info=True)

        if poll_failed:
            consecutive_failures += 1
            if consecutive_failures >= vcst.EVAL_BASILICA_MAX_CONSECUTIVE_POLL_FAILURES:
                return (
                    f"Deployment {deployment_name} appears dead: "
                    f"{consecutive_failures} consecutive failed polls"
                )
            next_poll_at += vcst.EVAL_BASILICA_FAILED_POLL_RECHECK_SECONDS
            continue

        consecutive_failures = 0
        eval_logger.info(
            f"[{repo}] result not ready yet (status may be running/in_progress), "
            f"polling again in {poll_interval_seconds}s..."
        )
        next_poll_at += poll_interval_seconds
    return f"Timed out waiting for result after {max_poll_seconds}s"


async def _deployment_exists(client, deployment_name: str) -> bool:
    deployments = await asyncio.to_thread(client.list)
    return any(getattr(dep, "name", None) == deployment_name for dep in deployments)


async def _get_deployment_by_name(client, deployment_name: str):
    deployments = await asyncio.to_thread(client.list)
    by_name = {getattr(dep, "name", None): dep for dep in deployments}
    return by_name.get(deployment_name)


async def _wait_for_deployment_ready(client, deployment, deployment_name: str, timeout_seconds: int):
    deadline = time.monotonic() + timeout_seconds
    current_deployment = deployment
    while time.monotonic() < deadline:
        if current_deployment is not None and deployment_is_healthy(current_deployment):
            return current_deployment
        refreshed = await _get_deployment_by_name(client, deployment_name)
        if refreshed is not None:
            current_deployment = refreshed
        await asyncio.sleep(5)
    return None


async def _deploy_with_readiness_timeout(
    *,
    ctx: _BasilicaEvalContext,
    client,
    deployment_name: str,
    deploy_kwargs: dict,
):
    timeout_seconds = vcst.EVAL_DEPLOYMENT_READY_TIMEOUT_SECONDS
    started_monotonic = time.monotonic()
    deployment = None
    try:
        deployment = await asyncio.wait_for(asyncio.to_thread(client.deploy, **deploy_kwargs), timeout=timeout_seconds)
        resolved_deployment_name = getattr(deployment, "name", None) or deployment_name
        remaining_seconds = max(1, int(timeout_seconds - (time.monotonic() - started_monotonic)))
        ready_deployment = await _wait_for_deployment_ready(
            client,
            deployment,
            resolved_deployment_name,
            remaining_seconds,
        )
        if ready_deployment is not None:
            return ready_deployment, resolved_deployment_name
        deployment_name = resolved_deployment_name
    except asyncio.TimeoutError as exc:
        ctx.log_eval_step("deploy_ready_timeout", deployment=deployment_name, timeout_seconds=timeout_seconds)
        existing = deployment or await _get_deployment_by_name(client, deployment_name)
        if existing is not None:
            await _delete_eval_deployment(ctx, client, existing, deployment_name, "deploy_ready_timeout")
        raise DeploymentNotReadyError(f"Deployment {deployment_name} was not ready within {timeout_seconds}s") from exc
    except DeploymentNotReadyError:
        raise
    except Exception:
        raise

    existing = deployment or await _get_deployment_by_name(client, deployment_name)
    if existing is not None:
        await _delete_eval_deployment(ctx, client, existing, deployment_name, "not_ready_timeout")
    raise DeploymentNotReadyError(f"Deployment {deployment_name} was not ready within {timeout_seconds}s")


async def _delete_terminal_deployment(
    *,
    client,
    deployment,
    deployment_name: str,
    reason: str,
    repo: str,
    eval_logger: logging.Logger,
    deleted_deployment_names: set[str],
    log_eval_step: Callable[..., None],
    max_attempts: int = 3,
) -> None:
    if deployment_name in deleted_deployment_names:
        log_eval_step("delete_skipped", deployment=deployment_name, reason=reason)
        return
    try:
        log_eval_step("fetch_logs_terminal_start", deployment=deployment_name, reason=reason)
        await asyncio.to_thread(log_basilica_logs_block, eval_logger, repo, deployment_name, deployment)
        log_eval_step("fetch_logs_terminal_done", deployment=deployment_name, reason=reason)
    except Exception as e:
        eval_logger.warning(f"[{repo}] failed to fetch terminal Basilica logs for deployment {deployment_name}: {e}")
        log_eval_step("fetch_logs_terminal_failed", deployment=deployment_name, reason=reason, error=e)

    last_error = None
    for delete_attempt in range(1, max_attempts + 1):
        try:
            attempt = f"{delete_attempt}/{max_attempts}"
            log_eval_step("delete_start", deployment=deployment_name, reason=reason, attempt=attempt)
            await asyncio.to_thread(deployment.delete)
            log_eval_step("delete_verify", deployment=deployment_name, reason=reason, attempt=attempt)
            if not await _deployment_exists(client, deployment_name):
                deleted_deployment_names.add(deployment_name)
                log_eval_step("delete_done", deployment=deployment_name, reason=reason, attempt=attempt)
                return
            log_eval_step("delete_still_exists", deployment=deployment_name, reason=reason, attempt=attempt)
        except Exception as e:
            last_error = e
            eval_logger.warning(
                f"[{repo}] failed to delete terminal deployment {deployment_name} "
                f"({reason}) attempt {delete_attempt}/{max_attempts}: {e}"
            )
            log_eval_step("delete_failed_attempt", deployment=deployment_name, reason=reason, attempt=attempt, error=e)
        if delete_attempt < max_attempts:
            await asyncio.sleep(1)
    log_eval_step("delete_failed", deployment=deployment_name, reason=reason, error=last_error)


async def _delete_eval_deployment(
    ctx: _BasilicaEvalContext,
    client,
    deployment,
    deployment_name: str,
    reason: str,
) -> None:
    await _delete_terminal_deployment(
        client=client,
        deployment=deployment,
        deployment_name=deployment_name,
        reason=reason,
        repo=ctx.repo,
        eval_logger=ctx.eval_logger,
        deleted_deployment_names=ctx.deleted_deployment_names,
        log_eval_step=ctx.log_eval_step,
    )


async def _release_reserved_gpus(
    *,
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    hotkeys: list[str],
    deployment_name: str,
    ctx: _BasilicaEvalContext,
) -> None:
    if task_id is None or psql_db is None or not hotkeys:
        return
    await _db_call_with_retry(
        lambda: tasks_sql.release_evaluation_gpu_reservation(task_id, hotkeys, deployment_name, psql_db),
        "release_evaluation_gpu_reservation",
        ctx.eval_logger,
        ctx.repo,
    )


async def _get_healthy_existing_basilica_deployment(
    *,
    existing_deployment_name: str,
    ctx: _BasilicaEvalContext,
):
    try:
        ctx.log_eval_step("resume_lookup_start", deployment=existing_deployment_name)
        client = basilica.BasilicaClient()
        deployments = await asyncio.to_thread(client.list)
        by_name = {getattr(dep, "name", None): dep for dep in deployments}
        deployment = by_name.get(existing_deployment_name)
        if deployment is None:
            ctx.eval_logger.warning(f"[{ctx.repo}] resume: deployment {existing_deployment_name} not found, redeploying")
            ctx.log_eval_step("resume_lookup_missing", deployment=existing_deployment_name)
            return None

        if not deployment_is_healthy(deployment):
            ctx.eval_logger.warning(f"[{ctx.repo}] resume: deployment {existing_deployment_name} unhealthy, redeploying")
            await _delete_eval_deployment(ctx, client, deployment, existing_deployment_name, "resume_unhealthy")
            return None

        ctx.eval_logger.info(f"[{ctx.repo}] resuming polling deployment {existing_deployment_name}")
        ctx.log_eval_step("resume_lookup_healthy", deployment=existing_deployment_name)
        return client, deployment, existing_deployment_name
    except Exception as e:
        ctx.eval_logger.error(f"[{ctx.repo}] resume failed, redeploying: {e}", exc_info=True)
        ctx.log_eval_step("resume_failed_redeploying", deployment=existing_deployment_name, error=e)
        return None


async def _deploy_basilica_eval_repo(
    *,
    ctx: _BasilicaEvalContext,
    deployment_name: str,
    image: str,
    source: str,
    env: dict[str, str],
    gpu_count: int | None,
    gpu_models: list[str],
    min_gpu_memory_gb: int | None,
    storage: bool | str,
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    repo_to_hotkey: dict[str, str],
):
    client = basilica.BasilicaClient()
    await asyncio.sleep(random.uniform(0.0, 0.25))
    reserved_hotkeys = [repo_to_hotkey[ctx.repo]] if ctx.repo in repo_to_hotkey else []
    requested_gpus = max(0, gpu_count or 0)

    if task_id is not None and psql_db is not None and reserved_hotkeys and requested_gpus > 0:
        reserved = await _db_call_with_retry(
            lambda: tasks_sql.try_reserve_evaluation_gpus(
                task_id,
                reserved_hotkeys,
                deployment_name,
                requested_gpus,
                psql_db,
            ),
            "try_reserve_evaluation_gpus",
            ctx.eval_logger,
            ctx.repo,
        )
        if not reserved:
            ctx.log_eval_step("gpu_capacity_unavailable", deployment=deployment_name, gpu_count=requested_gpus)
            raise EvaluationCapacityUnavailable(
                f"Not enough evaluation GPU capacity for deployment {deployment_name} ({requested_gpus} GPUs)"
            )

    ctx.log_eval_step("deployment_id_persist_start", deployment=deployment_name)
    async with _EVAL_DB_WRITE_SEMAPHORE:
        await _db_call_with_retry(
            lambda: persist_deployment_ids_for_repo(
                task_id,
                psql_db,
                repo_to_hotkey,
                ctx.repo,
                deployment_name,
            ),
            "persist_deployment_ids_for_repo(pre-deploy)",
            ctx.eval_logger,
            ctx.repo,
        )
    ctx.log_eval_step("deployment_id_persist_complete", deployment=deployment_name)
    ctx.log_eval_step(
        "deploy_start",
        deployment=deployment_name,
        image=image,
        gpu_count=gpu_count,
        gpu_models=",".join(gpu_models),
        min_gpu_memory_gb=min_gpu_memory_gb,
    )

    deploy_kwargs = {
        "name": deployment_name,
        "source": source,
        "image": image,
        "port": 8000,
        "cpu": vcst.EVAL_BASILICA_CPU,
        "memory": vcst.EVAL_BASILICA_MEMORY,
        "storage": storage,
        "ttl_seconds": vcst.EVAL_BASILICA_TTL_SECONDS,
        "timeout": vcst.EVAL_BASILICA_TIMEOUT,
        "env": env,
    }
    if gpu_count and gpu_count > 0:
        deploy_kwargs["gpu_count"] = gpu_count
        deploy_kwargs["gpu_models"] = gpu_models
        deploy_kwargs["min_gpu_memory_gb"] = min_gpu_memory_gb

    deployment, resolved_deployment_name = await _deploy_with_readiness_timeout(
        ctx=ctx,
        client=client,
        deployment_name=deployment_name,
        deploy_kwargs=deploy_kwargs,
    )
    resolved_deployment_name = getattr(deployment, "name", None) or deployment_name
    update_environment_logger_labels(
        ctx.eval_logger,
        deployment_id=resolved_deployment_name,
        deployment_url=_deployment_url(deployment),
    )
    ctx.log_eval_step("deploy_complete", deployment=resolved_deployment_name)

    if resolved_deployment_name != deployment_name:
        if task_id is not None and psql_db is not None and reserved_hotkeys:
            await _db_call_with_retry(
                lambda: tasks_sql.release_evaluation_gpu_reservation(task_id, reserved_hotkeys, deployment_name, psql_db),
                "release_evaluation_gpu_reservation(old-name)",
                ctx.eval_logger,
                ctx.repo,
            )
            reserved = await _db_call_with_retry(
                lambda: tasks_sql.try_reserve_evaluation_gpus(
                    task_id,
                    reserved_hotkeys,
                    resolved_deployment_name,
                    requested_gpus,
                    psql_db,
                ),
                "try_reserve_evaluation_gpus(resolved-name)",
                ctx.eval_logger,
                ctx.repo,
            )
            if not reserved:
                await _delete_eval_deployment(
                    ctx,
                    client,
                    deployment,
                    resolved_deployment_name,
                    "resolved_name_capacity_unavailable",
                )
                raise EvaluationCapacityUnavailable(
                    f"Not enough evaluation GPU capacity for deployment {resolved_deployment_name} ({requested_gpus} GPUs)"
                )
        await asyncio.sleep(random.uniform(0.0, 0.25))
        ctx.log_eval_step(
            "deployment_id_repersist_start",
            deployment=resolved_deployment_name,
            previous_deployment=deployment_name,
        )
        async with _EVAL_DB_WRITE_SEMAPHORE:
            await _db_call_with_retry(
                lambda: persist_deployment_ids_for_repo(
                    task_id,
                    psql_db,
                    repo_to_hotkey,
                    ctx.repo,
                    resolved_deployment_name,
                ),
                "persist_deployment_ids_for_repo(post-deploy)",
                ctx.eval_logger,
                ctx.repo,
            )
        ctx.log_eval_step("deployment_id_repersist_complete", deployment=resolved_deployment_name)

    ctx.eval_logger.info(f"[{ctx.repo}] deployment started: {resolved_deployment_name}")
    return client, deployment, resolved_deployment_name


async def _poll_eval_deployment(
    *,
    ctx: _BasilicaEvalContext,
    client,
    deployment,
    deployment_name: str,
    success_cleanup_reason: str,
    failure_cleanup_reason: str,
    timeout_cleanup_reason: str,
    retry_on_failure: bool,
    poll_interval_seconds: int = vcst.EVAL_BASILICA_POLL_INTERVAL_SECONDS,
    max_poll_seconds: int = vcst.EVAL_BASILICA_MAX_POLL_SECONDS,
) -> dict | str:
    ctx.log_eval_step("poll_start", deployment=deployment_name)
    result = await _poll_basilica_result(
        deployment,
        ctx.repo,
        eval_logger=ctx.eval_logger,
        poll_interval_seconds=poll_interval_seconds,
        max_poll_seconds=max_poll_seconds,
    )

    if isinstance(result, dict):
        ctx.log_eval_step("poll_complete", deployment=deployment_name)
        await _delete_eval_deployment(ctx, client, deployment, deployment_name, success_cleanup_reason)
        return result

    if "Timed out" in str(result):
        logger.error(f"[{ctx.repo}] poll timeout, skipping retries: {result}")
        ctx.log_eval_step("poll_timeout", deployment=deployment_name, result=result)
        await _delete_eval_deployment(ctx, client, deployment, deployment_name, timeout_cleanup_reason)
        return result

    ctx.log_eval_step("poll_failed", deployment=deployment_name, result=result)
    await _delete_eval_deployment(ctx, client, deployment, deployment_name, failure_cleanup_reason)
    if retry_on_failure:
        raise RuntimeError(str(result))
    return str(result) if result else "Resume poll returned empty"


async def _fetch_attempt_logs(ctx: _BasilicaEvalContext, deployment, fallback_deployment_name: str) -> None:
    dep_name = getattr(deployment, "name", None) or fallback_deployment_name
    try:
        if dep_name in ctx.deleted_deployment_names:
            ctx.log_eval_step("fetch_logs_skipped_deleted", deployment=dep_name)
            return

        ctx.log_eval_step("fetch_logs_start", deployment=dep_name)
        await asyncio.to_thread(log_basilica_logs_block, ctx.eval_logger, ctx.repo, dep_name, deployment)
        ctx.log_eval_step("fetch_logs_done", deployment=dep_name)
    except Exception as e:
        ctx.eval_logger.warning(f"[{ctx.repo}] failed to fetch Basilica logs for deployment {dep_name}: {e}")
        ctx.log_eval_step("fetch_logs_failed", deployment=dep_name, error=e)


async def _run_single_basilica_eval_repo(
    *,
    repo: str,
    model_name: str,
    task_type: str,
    image: str,
    source: str,
    env: dict[str, str],
    gpu_count: int | None,
    gpu_models: list[str],
    min_gpu_memory_gb: int | None,
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    repo_to_hotkey: dict[str, str],
    storage: bool | str = False,
    hotkey: str | None = None,
    existing_deployment_name: str | None = None,
    local_logging: bool | None = False,
) -> dict | str:
    """Run one repo eval with retries. Supports resume via existing_deployment_name."""
    eval_id = str(uuid.uuid4())
    task_id_str = str(task_id) if task_id else "unknown"
    hotkey_str = hotkey or repo_to_hotkey.get(repo) or "unknown"
    if not local_logging:
        eval_logger = get_environment_logger(
            name=f"basilica-{repo.split('/')[-1]}-{eval_id[:8]}",
            repo_id=repo,
            eval_id=eval_id,
            model=model_name,
            task_type=task_type,
            task_id=task_id_str,
            hotkey=hotkey_str,
            deployment_id=existing_deployment_name,
        )
    else:
        eval_logger = get_logger(f"{__name__}.basilica.{repo.split('/')[-1]}.{eval_id[:8]}")

    def log_step(step: str, **fields) -> None:
        _log_eval_step(eval_logger, step, **fields)

    ctx = _BasilicaEvalContext(
        repo=repo,
        eval_logger=eval_logger,
        deleted_deployment_names=set(),
        log_eval_step=log_step,
    )
    reserved_hotkeys = [repo_to_hotkey[repo]] if repo in repo_to_hotkey else []

    if existing_deployment_name:
        resume_deployment = await _get_healthy_existing_basilica_deployment(
            existing_deployment_name=existing_deployment_name,
            ctx=ctx,
        )
        if resume_deployment is not None:
            client, deployment, deployment_name = resume_deployment
            update_environment_logger_labels(
                eval_logger,
                deployment_id=deployment_name,
                deployment_url=_deployment_url(deployment),
            )
            return await _poll_eval_deployment(
                ctx=ctx,
                client=client,
                deployment=deployment,
                deployment_name=deployment_name,
                success_cleanup_reason="resume_completed",
                failure_cleanup_reason="resume_failed_or_timed_out",
                timeout_cleanup_reason="resume_failed_or_timed_out",
                retry_on_failure=False,
            )
        await _release_reserved_gpus(
            task_id=task_id,
            psql_db=psql_db,
            hotkeys=reserved_hotkeys,
            deployment_name=existing_deployment_name,
            ctx=ctx,
        )

    for attempt in range(1, vcst.EVAL_BASILICA_MAX_RETRIES + 1):
        deployment = None
        deployment_name = str(uuid.uuid4())
        try:
            update_environment_logger_labels(eval_logger, deployment_id=deployment_name)
            log_step("attempt_start", attempt=f"{attempt}/{vcst.EVAL_BASILICA_MAX_RETRIES}", deployment=deployment_name)
            eval_logger.info(f"[{repo}] starting Basilica evaluation attempt {attempt}/{vcst.EVAL_BASILICA_MAX_RETRIES}")
            client, deployment, resolved_deployment_name = await _deploy_basilica_eval_repo(
                ctx=ctx,
                deployment_name=deployment_name,
                image=image,
                source=source,
                env=env,
                gpu_count=gpu_count,
                gpu_models=gpu_models,
                min_gpu_memory_gb=min_gpu_memory_gb,
                storage=storage,
                task_id=task_id,
                psql_db=psql_db,
                repo_to_hotkey=repo_to_hotkey,
            )
            update_environment_logger_labels(
                eval_logger,
                deployment_id=resolved_deployment_name,
                deployment_url=_deployment_url(deployment),
            )
            return await _poll_eval_deployment(
                ctx=ctx,
                client=client,
                deployment=deployment,
                deployment_name=resolved_deployment_name,
                success_cleanup_reason="completed",
                failure_cleanup_reason="failed",
                timeout_cleanup_reason="timed_out",
                retry_on_failure=True,
            )
        except asyncio.CancelledError:
            log_step("attempt_cancelled", attempt=f"{attempt}/{vcst.EVAL_BASILICA_MAX_RETRIES}", deployment=deployment_name)
            raise
        except EvaluationRetryableError:
            dep_name = (getattr(deployment, "name", None) or deployment_name) if deployment is not None else deployment_name
            if deployment is not None:
                await _delete_eval_deployment(ctx, client, deployment, dep_name, "attempt_retryable")
            await _release_reserved_gpus(
                task_id=task_id,
                psql_db=psql_db,
                hotkeys=reserved_hotkeys,
                deployment_name=dep_name,
                ctx=ctx,
            )
            raise
        except Exception as e:
            remaining = vcst.EVAL_BASILICA_MAX_RETRIES - attempt
            dep_name = (getattr(deployment, "name", None) or deployment_name) if deployment is not None else deployment_name
            if deployment is not None:
                await _delete_eval_deployment(ctx, client, deployment, dep_name, "attempt_exception")
            await _release_reserved_gpus(
                task_id=task_id,
                psql_db=psql_db,
                hotkeys=reserved_hotkeys,
                deployment_name=dep_name,
                ctx=ctx,
            )
            log_step(
                "attempt_failed",
                attempt=f"{attempt}/{vcst.EVAL_BASILICA_MAX_RETRIES}",
                deployment=deployment_name,
                remaining=remaining,
                error=e,
            )
            eval_logger.error(
                f"[{repo}] attempt {attempt}/{vcst.EVAL_BASILICA_MAX_RETRIES} failed: {e}",
                exc_info=True,
            )
            if remaining > 0:
                eval_logger.info(
                    f"[{repo}] retrying in {vcst.EVAL_BASILICA_RETRY_DELAY_SECONDS // 60} minutes "
                    f"({remaining} attempts remaining)"
                )
                log_step(
                    "retry_sleep_start",
                    delay_seconds=vcst.EVAL_BASILICA_RETRY_DELAY_SECONDS,
                    remaining=remaining,
                )
                await asyncio.sleep(vcst.EVAL_BASILICA_RETRY_DELAY_SECONDS)
            else:
                log_step("all_attempts_failed", deployment=deployment_name, error=e)
                return f"Evaluation failed after {vcst.EVAL_BASILICA_MAX_RETRIES} attempts: {e}"
        finally:
            if deployment is not None:
                await _fetch_attempt_logs(ctx, deployment, deployment_name)

    return "Evaluation failed"


async def run_basilica_eval_repos(
    *,
    repos: list[str],
    model_name: str,
    task_type: str,
    image: str,
    source: str,
    build_env_for_repo,
    gpu_count: int | None,
    gpu_models: list[str],
    min_gpu_memory_gb: int | None,
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    repo_to_hotkey: dict[str, str],
    storage: bool | str = False,
    deployment_ids_by_repo: dict[str, str] | None = None,
    local_logging: bool | None = False,
) -> dict[str, dict | str]:
    deployment_ids_by_repo = deployment_ids_by_repo or {}
    task_results = await asyncio.gather(
        *[
            _run_single_basilica_eval_repo(
                repo=repo,
                model_name=model_name,
                task_type=task_type,
                image=image,
                source=source,
                env=build_env_for_repo(repo),
                gpu_count=gpu_count,
                gpu_models=gpu_models,
                min_gpu_memory_gb=min_gpu_memory_gb,
                storage=storage,
                task_id=task_id,
                psql_db=psql_db,
                repo_to_hotkey=repo_to_hotkey,
                hotkey=repo_to_hotkey.get(repo),
                existing_deployment_name=(
                    deployment_ids_by_repo.get(repo) if isinstance(deployment_ids_by_repo.get(repo), str) else None
                ),
                local_logging=local_logging,
            )
            for repo in repos
        ],
        return_exceptions=True,
    )
    out: dict[str, dict | str] = {}
    for repo, result in zip(repos, task_results):
        if isinstance(result, EvaluationRetryableError):
            raise result
        elif isinstance(result, Exception):
            out[repo] = f"Evaluation failed: {result}"
        else:
            out[repo] = result
    return out
