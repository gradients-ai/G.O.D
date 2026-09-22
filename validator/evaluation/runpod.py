"""Runpod evaluation backend implemented as dstack services."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import hmac
import logging
import os
import shlex
import time
import uuid
from dataclasses import dataclass
from dataclasses import field
from typing import Awaitable
from typing import Callable
from uuid import UUID

import httpx

import validator.evaluation.constants as vcst
from core.logging import get_environment_logger
from core.logging import get_logger
from core.logging import update_environment_logger_labels
from validator.db.database import PSQLDB
from validator.db.sql import tasks as tasks_sql
from validator.evaluation.basilica import DeploymentNotReadyError
from validator.evaluation.basilica import EvaluationCapacityUnavailable
from validator.evaluation.basilica import EvaluationRetryableError
from validator.evaluation.basilica import _BasilicaEvalContext
from validator.evaluation.basilica import _db_call_with_retry
from validator.evaluation.basilica import _evaluation_cost_run_key
from validator.evaluation.basilica import _finish_evaluation_cost_run
from validator.evaluation.basilica import _release_reserved_gpus
from validator.evaluation.basilica import _start_evaluation_cost_run
from validator.evaluation.db_utils import persist_deployment_ids_for_repo
from validator.evaluation.evaluation_logging import _log_eval_step
from validator.infrastructure.dstack_client import DstackClient
from validator.infrastructure.dstack_client import parse_dstack_submitted_at


logger = get_logger(__name__)
RUNPOD_DEPLOYMENT_ID_PREFIX = "runpod:"
RUNPOD_RUN_NAME_PREFIX = "god-eval-"
_EVAL_DB_WRITE_SEMAPHORE = asyncio.Semaphore(vcst.EVAL_DB_MAX_CONCURRENT_WRITES)


def encode_runpod_deployment_id(run_name: str) -> str:
    return f"{RUNPOD_DEPLOYMENT_ID_PREFIX}{run_name}"


def decode_runpod_deployment_id(deployment_id: str | None) -> str | None:
    if not deployment_id or not deployment_id.startswith(RUNPOD_DEPLOYMENT_ID_PREFIX):
        return None
    return deployment_id.removeprefix(RUNPOD_DEPLOYMENT_ID_PREFIX)


def is_runpod_deployment_id(deployment_id: str | None) -> bool:
    return decode_runpod_deployment_id(deployment_id) is not None


def generate_runpod_run_name() -> str:
    # dstack names must start with a letter and contain only lowercase
    # alphanumerics/hyphens. This is 41 characters, below its 63-char limit.
    return f"{RUNPOD_RUN_NAME_PREFIX}{uuid.uuid4().hex}"


def _control_token(run_name: str) -> str:
    secret = os.getenv("DSTACK_TOKEN") or ""
    if not secret:
        raise RuntimeError("DSTACK_TOKEN is required for Runpod evaluations")
    return hmac.new(secret.encode(), run_name.encode(), hashlib.sha256).hexdigest()


def _is_not_found(exc: Exception) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    if exc.response.status_code == 404:
        return True
    if exc.response.status_code != 400:
        return False
    try:
        detail = exc.response.json().get("detail", [])
    except Exception:
        return False
    return any(
        item.get("code") == "resource_not_exists"
        or "not found" in str(item.get("msg", "")).lower()
        for item in detail
        if isinstance(item, dict)
    )


def _status(run: dict) -> str:
    return str(run.get("status", "")).lower()


def _got_no_offers(run: dict) -> bool:
    submission = run.get("latest_job_submission") or {}
    return str(submission.get("status_message", "")).lower() == "no offers"


def _job_submission_id(run: dict) -> str | None:
    submission = run.get("latest_job_submission") or {}
    value = submission.get("id")
    return str(value) if value else None


def _runner_control_token_from_run(run: dict, fallback: str) -> str:
    configuration = (run.get("run_spec") or {}).get("configuration") or {}
    env = configuration.get("env") or {}
    if isinstance(env, dict):
        value = env.get("EVAL_RUNNER_CONTROL_TOKEN")
        if value:
            return str(value)
    return fallback


@dataclass
class RunpodDeployment:
    run_name: str
    deployment_id: str
    url: str | None
    created_at: datetime.datetime
    control_token: str
    run: dict
    next_log_token: str | None = None
    seen_log_messages: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        return self.deployment_id

    @property
    def state(self) -> str:
        return _status(self.run)


def _build_service_plan(
    *,
    run_name: str,
    image: str,
    source: str,
    env: dict[str, str],
    gpu_count: int,
) -> dict:
    configuration: dict = {
        "type": "service",
        "name": run_name,
        "image": image,
        "env": env,
        "commands": [f"python3 -c {shlex.quote(source)}"],
        "port": 8000,
        # The runner protects /result and the SWE /v1 proxy uses its own key.
        # Disabling dstack auth lets external SWE workers reach /v1.
        "auth": False,
        "probes": [{"type": "http", "url": "/health"}],
        "resources": {
            "gpu": {
                "name": ["A100"],
                "count": {"min": gpu_count, "max": gpu_count},
            },
            "disk": {"size": os.getenv("EVAL_RUNPOD_DISK_SIZE", "200GB")},
        },
        "max_duration": vcst.EVAL_RUNPOD_MAX_DURATION_SECONDS,
    }
    regions = [part.strip() for part in os.getenv("EVAL_RUNPOD_REGIONS", "").split(",") if part.strip()]
    if regions:
        configuration["regions"] = regions
    gateway = os.getenv("EVAL_RUNPOD_GATEWAY")
    if gateway:
        configuration["gateway"] = True if gateway.lower() == "true" else gateway
    return {
        "plan": {
            "run_spec": {
                "run_name": run_name,
                "configuration": configuration,
            }
        },
        "force": False,
    }


async def _fetch_logs(
    client: DstackClient,
    deployment: RunpodDeployment,
    eval_logger: logging.Logger,
    repo: str,
) -> None:
    submission_id = _job_submission_id(deployment.run)
    if not submission_id:
        return
    try:
        page = await client.poll_logs(
            deployment.run_name,
            submission_id,
            next_token=deployment.next_log_token,
        )
        deployment.next_log_token = page.next_token
        for line in page.messages:
            if line in deployment.seen_log_messages:
                continue
            deployment.seen_log_messages.add(line)
            eval_logger.info(
                "[REMOTE_EVAL_LOG] backend=runpod repo=%s deployment=%s | %s",
                repo,
                deployment.deployment_id,
                line,
            )
    except Exception as exc:
        eval_logger.warning(
            "[REMOTE_EVAL_LOG_FETCH_FAILED] backend=runpod repo=%s deployment=%s error=%s",
            repo,
            deployment.deployment_id,
            exc,
        )


async def _cleanup_run(
    client: DstackClient,
    deployment: RunpodDeployment,
    ctx: _BasilicaEvalContext,
    reason: str,
    *,
    max_attempts: int = 3,
) -> bool:
    if deployment.deployment_id in ctx.deleted_deployment_names:
        return True
    await _fetch_logs(client, deployment, ctx.eval_logger, ctx.repo)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            ctx.log_eval_step(
                "delete_start",
                deployment=deployment.deployment_id,
                reason=reason,
                attempt=f"{attempt}/{max_attempts}",
            )
            try:
                current_run = await client.get_run(deployment.run_name)
                deployment.run = current_run
            except Exception as exc:
                if _is_not_found(exc):
                    ctx.deleted_deployment_names.add(deployment.deployment_id)
                    return True
                raise

            if _status(deployment.run) not in {"terminated", "failed", "done"}:
                try:
                    await client.stop_run(deployment.run_name, abort=True)
                except Exception as exc:
                    if not _is_not_found(exc):
                        raise

            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                try:
                    run = await client.get_run(deployment.run_name)
                except Exception as exc:
                    if _is_not_found(exc):
                        ctx.deleted_deployment_names.add(deployment.deployment_id)
                        return True
                    raise
                if _status(run) in {"terminated", "failed", "done"}:
                    break
                await asyncio.sleep(2)

            try:
                await client.delete_run(deployment.run_name)
            except Exception as exc:
                if not _is_not_found(exc):
                    raise
            try:
                await client.get_run(deployment.run_name)
            except Exception as exc:
                if _is_not_found(exc):
                    ctx.deleted_deployment_names.add(deployment.deployment_id)
                    ctx.log_eval_step(
                        "delete_done",
                        deployment=deployment.deployment_id,
                        reason=reason,
                    )
                    return True
                raise
            last_error = RuntimeError(f"dstack run {deployment.run_name} still exists after delete")
        except Exception as exc:
            last_error = exc
            ctx.eval_logger.warning(
                "[%s] Runpod cleanup failed for %s attempt %d/%d: %s",
                ctx.repo,
                deployment.deployment_id,
                attempt,
                max_attempts,
                exc,
            )
        if attempt < max_attempts:
            await asyncio.sleep(2**attempt)
    ctx.log_eval_step(
        "delete_failed",
        deployment=deployment.deployment_id,
        reason=reason,
        error=last_error,
    )
    return False


async def _create_or_recover_run(
    *,
    client: DstackClient,
    run_name: str,
    image: str,
    source: str,
    env: dict[str, str],
    gpu_count: int,
) -> RunpodDeployment:
    control_token = env["EVAL_RUNNER_CONTROL_TOKEN"]
    if "public_sglang" in source and not env.get("SWE_INFINITE_MODEL_BASE_URL"):
        gateway = os.getenv("EVAL_RUNPOD_GATEWAY")
        template = os.getenv("EVAL_RUNPOD_SERVICE_URL_TEMPLATE")
        if gateway and not template:
            raise RuntimeError(
                "EVAL_RUNPOD_SERVICE_URL_TEMPLATE is required for SWE Infinite when "
                "EVAL_RUNPOD_GATEWAY is configured"
            )
        if template:
            public_service_url = template.format(run_name=run_name).rstrip("/")
        else:
            public_service_url = (
                f"{client.config.url}/proxy/services/{client.config.project}/{run_name}"
            )
        env = {**env, "SWE_INFINITE_MODEL_BASE_URL": f"{public_service_url}/v1"}
    try:
        run = await client.apply_run(
            _build_service_plan(
                run_name=run_name,
                image=image,
                source=source,
                env=env,
                gpu_count=gpu_count,
            )
        )
    except Exception as apply_exc:
        # An apply response can time out after dstack committed the run. Reconcile
        # by name before declaring submission failed so the caller can always
        # stop/delete the service it may have created.
        for lookup_attempt in range(6):
            try:
                run = await client.get_run(run_name)
                break
            except Exception as lookup_exc:
                if not _is_not_found(lookup_exc) and lookup_attempt == 5:
                    raise apply_exc from lookup_exc
                if lookup_attempt < 5:
                    await asyncio.sleep(5)
        else:
            raise apply_exc
    created_at = parse_dstack_submitted_at(run) or datetime.datetime.now(datetime.timezone.utc)
    return RunpodDeployment(
        run_name=run_name,
        deployment_id=encode_runpod_deployment_id(run_name),
        url=client.resolve_service_url(run),
        created_at=created_at,
        control_token=_runner_control_token_from_run(run, control_token),
        run=run,
    )


async def _wait_ready(
    client: DstackClient,
    deployment: RunpodDeployment,
    ctx: _BasilicaEvalContext,
) -> RunpodDeployment:
    deadline = time.monotonic() + vcst.EVAL_DEPLOYMENT_READY_TIMEOUT_SECONDS
    async with httpx.AsyncClient(timeout=10.0) as http:
        while time.monotonic() < deadline:
            run = await client.get_run(deployment.run_name)
            deployment.run = run
            deployment.url = client.resolve_service_url(run) or deployment.url
            status = _status(run)
            if _got_no_offers(run):
                raise EvaluationCapacityUnavailable(f"No Runpod offers for {deployment.deployment_id}")
            if status in {"failed", "aborted", "terminated", "done"}:
                detail = run.get("error") or (run.get("latest_job_submission") or {}).get("status_message")
                raise RuntimeError(f"Runpod deployment {deployment.deployment_id} ended in {status}: {detail}")
            if status == "running" and deployment.url:
                try:
                    response = await http.get(f"{deployment.url.rstrip('/')}/health")
                    if response.is_success:
                        return deployment
                except Exception:
                    pass
            await _fetch_logs(client, deployment, ctx.eval_logger, ctx.repo)
            await asyncio.sleep(5)
    raise DeploymentNotReadyError(
        f"Runpod deployment {deployment.deployment_id} was not ready within "
        f"{vcst.EVAL_DEPLOYMENT_READY_TIMEOUT_SECONDS}s"
    )


async def _poll_result(
    client: DstackClient,
    deployment: RunpodDeployment,
    ctx: _BasilicaEvalContext,
) -> dict | str:
    deadline = time.monotonic() + vcst.EVAL_BASILICA_MAX_POLL_SECONDS
    failures = 0
    headers = {"Authorization": f"Bearer {deployment.control_token}"}
    async with httpx.AsyncClient(timeout=30.0) as http:
        while time.monotonic() < deadline:
            await _fetch_logs(client, deployment, ctx.eval_logger, ctx.repo)
            run = await client.get_run(deployment.run_name)
            deployment.run = run
            status = _status(run)
            if _got_no_offers(run):
                raise EvaluationCapacityUnavailable(f"No Runpod offers for {deployment.deployment_id}")
            if status in {"failed", "aborted", "terminated", "done"}:
                detail = run.get("error") or (run.get("latest_job_submission") or {}).get("status_message")
                return f"Runpod service ended in {status} before returning a result: {detail}"
            try:
                response = await http.get(f"{deployment.url.rstrip('/')}/result", headers=headers)
                response.raise_for_status()
                payload = response.json()
                failures = 0
                if payload.get("status") == "completed":
                    result = payload.get("result")
                    return result if isinstance(result, dict) else f"Completed but result payload invalid: {result}"
                if payload.get("status") == "failed":
                    return str(payload.get("error") or "Runpod eval reported failure")
                ctx.eval_logger.info("[%s] Runpod poll ping: status=%s", ctx.repo, payload.get("status"))
            except Exception as exc:
                failures += 1
                ctx.eval_logger.warning(
                    "[%s] Runpod result poll failed %d/%d: %s",
                    ctx.repo,
                    failures,
                    vcst.EVAL_BASILICA_MAX_CONSECUTIVE_POLL_FAILURES,
                    exc,
                )
                if failures >= vcst.EVAL_BASILICA_MAX_CONSECUTIVE_POLL_FAILURES:
                    return f"Deployment {deployment.deployment_id} appears dead: {failures} consecutive failed polls"
                await asyncio.sleep(vcst.EVAL_BASILICA_FAILED_POLL_RECHECK_SECONDS)
                continue
            await asyncio.sleep(vcst.EVAL_BASILICA_POLL_INTERVAL_SECONDS)
    return f"Timed out waiting for result after {vcst.EVAL_BASILICA_MAX_POLL_SECONDS}s"


async def _load_existing(
    client: DstackClient,
    deployment_id: str,
    control_token: str,
) -> RunpodDeployment | None:
    run_name = decode_runpod_deployment_id(deployment_id)
    if not run_name:
        return None
    try:
        run = await client.get_run(run_name)
    except Exception as exc:
        if _is_not_found(exc):
            return None
        raise
    return RunpodDeployment(
        run_name=run_name,
        deployment_id=deployment_id,
        url=client.resolve_service_url(run),
        created_at=parse_dstack_submitted_at(run) or datetime.datetime.now(datetime.timezone.utc),
        control_token=_runner_control_token_from_run(run, control_token),
        run=run,
    )


async def _run_single_runpod_eval_repo(
    *,
    repo: str,
    model_name: str,
    task_type: str,
    image: str,
    source: str,
    env: dict[str, str],
    gpu_count: int | None,
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    repo_to_hotkey: dict[str, str],
    hotkey: str | None,
    existing_deployment_name: str | None,
    local_logging: bool | None,
    deployment_id_persister: Callable[[str, str], Awaitable[None]] | None,
    reserve_deployment_id: bool,
) -> dict | str:
    eval_id = str(uuid.uuid4())
    if not local_logging:
        eval_logger = get_environment_logger(
            name=f"runpod-{repo.split('/')[-1]}-{eval_id[:8]}",
            repo_id=repo,
            eval_id=eval_id,
            model=model_name,
            task_type=task_type,
            task_id=str(task_id) if task_id else "unknown",
            hotkey=hotkey or repo_to_hotkey.get(repo) or "unknown",
            deployment_id=existing_deployment_name,
            eval_backend="runpod",
        )
    else:
        eval_logger = get_logger(f"{__name__}.{repo.split('/')[-1]}.{eval_id[:8]}")

    ctx = _BasilicaEvalContext(
        repo=repo,
        eval_logger=eval_logger,
        deleted_deployment_names=set(),
        log_eval_step=lambda step, **fields: _log_eval_step(eval_logger, step, backend="runpod", **fields),
    )
    client = DstackClient()
    requested_gpus = max(1, gpu_count or 1)
    reserved_hotkeys = [repo_to_hotkey[repo]] if repo in repo_to_hotkey else []
    control_token = ""
    run_env = dict(env)

    if existing_deployment_name and is_runpod_deployment_id(existing_deployment_name):
        existing_run_name = decode_runpod_deployment_id(existing_deployment_name)
        assert existing_run_name is not None
        control_token = _control_token(existing_run_name)
        resume_cost_key = (
            _evaluation_cost_run_key(task_id, existing_deployment_name)
            if task_id is not None and psql_db is not None and requested_gpus > 0
            else None
        )
        deployment = await _load_existing(client, existing_deployment_name, control_token)
        if deployment is not None:
            try:
                deployment = await _wait_ready(client, deployment, ctx)
                update_environment_logger_labels(
                    eval_logger,
                    deployment_id=deployment.deployment_id,
                    deployment_url=deployment.url,
                )
                result = await _poll_result(client, deployment, ctx)
                deleted = await _cleanup_run(
                    client,
                    deployment,
                    ctx,
                    "resume_completed" if isinstance(result, dict) else "resume_failed",
                )
                if not deleted:
                    raise EvaluationRetryableError(
                        f"Could not verify Runpod cleanup for {deployment.deployment_id}"
                    )
                if isinstance(result, dict):
                    await _finish_evaluation_cost_run(
                        run_key=resume_cost_key,
                        success=True,
                        psql_db=psql_db,
                        ctx=ctx,
                    )
                    return result
            except asyncio.CancelledError:
                deleted = await asyncio.shield(_cleanup_run(client, deployment, ctx, "resume_cancelled"))
                if deleted:
                    await asyncio.shield(
                        _release_reserved_gpus(
                            task_id=task_id,
                            psql_db=psql_db,
                            hotkeys=reserved_hotkeys,
                            deployment_name=existing_deployment_name if reserve_deployment_id else None,
                            ctx=ctx,
                        )
                    )
                    await asyncio.shield(
                        _finish_evaluation_cost_run(
                            run_key=resume_cost_key,
                            success=False,
                            psql_db=psql_db,
                            ctx=ctx,
                        )
                    )
                if not deleted:
                    raise EvaluationRetryableError(
                        f"Could not verify Runpod cleanup for cancelled deployment "
                        f"{deployment.deployment_id}"
                    )
                raise
            except Exception as exc:
                deleted = await _cleanup_run(client, deployment, ctx, "resume_exception")
                if not deleted:
                    raise EvaluationRetryableError(
                        f"Could not verify Runpod cleanup for {deployment.deployment_id}"
                    ) from exc
                if isinstance(exc, EvaluationCapacityUnavailable):
                    await _release_reserved_gpus(
                        task_id=task_id,
                        psql_db=psql_db,
                        hotkeys=reserved_hotkeys,
                        deployment_name=existing_deployment_name if reserve_deployment_id else None,
                        ctx=ctx,
                    )
                    await _finish_evaluation_cost_run(
                        run_key=resume_cost_key,
                        success=False,
                        psql_db=psql_db,
                        ctx=ctx,
                    )
                    raise
                ctx.eval_logger.warning(
                    "[%s] resumed Runpod deployment failed; redeploying: %s",
                    repo,
                    exc,
                )
        await _finish_evaluation_cost_run(
            run_key=resume_cost_key,
            success=False,
            psql_db=psql_db,
            ctx=ctx,
        )
        await _release_reserved_gpus(
            task_id=task_id,
            psql_db=psql_db,
            hotkeys=reserved_hotkeys,
            deployment_name=existing_deployment_name if reserve_deployment_id else None,
            ctx=ctx,
        )

    last_error: Exception | None = None
    for attempt in range(1, vcst.EVAL_BASILICA_MAX_RETRIES + 1):
        run_name = generate_runpod_run_name()
        deployment_id = encode_runpod_deployment_id(run_name)
        control_token = _control_token(run_name)
        run_env = {**env, "EVAL_RUNNER_CONTROL_TOKEN": control_token}
        deployment: RunpodDeployment | None = None
        cost_key = (
            _evaluation_cost_run_key(task_id, deployment_id)
            if task_id is not None and psql_db is not None and requested_gpus > 0
            else None
        )
        try:
            if task_id is not None and psql_db is not None and reserved_hotkeys:
                reserved = await _db_call_with_retry(
                    lambda: tasks_sql.try_reserve_evaluation_gpus(
                        task_id,
                        reserved_hotkeys,
                        deployment_id if reserve_deployment_id else None,
                        requested_gpus,
                        psql_db,
                    ),
                    "try_reserve_evaluation_gpus",
                    eval_logger,
                    repo,
                )
                if not reserved:
                    raise EvaluationCapacityUnavailable(
                        f"Not enough evaluation GPU capacity for {deployment_id} ({requested_gpus} GPUs)"
                    )
            if deployment_id_persister is not None:
                async with _EVAL_DB_WRITE_SEMAPHORE:
                    await _db_call_with_retry(
                        lambda: deployment_id_persister(repo, deployment_id),
                        "persist_runpod_deployment_id(pre-create)",
                        eval_logger,
                        repo,
                    )
            await _start_evaluation_cost_run(
                task_id=task_id,
                psql_db=psql_db,
                deployment_name=deployment_id,
                gpu_count=requested_gpus,
                ctx=ctx,
                backend="runpod",
                gpu_type="A100",
            )
            ctx.log_eval_step(
                "deploy_start",
                deployment=deployment_id,
                image=image,
                gpu_count=requested_gpus,
                gpu_models="A100",
            )
            deployment = await _create_or_recover_run(
                client=client,
                run_name=run_name,
                image=image,
                source=source,
                env=run_env,
                gpu_count=requested_gpus,
            )
            deployment = await _wait_ready(client, deployment, ctx)
            update_environment_logger_labels(
                eval_logger,
                deployment_id=deployment_id,
                deployment_url=deployment.url,
            )
            ctx.log_eval_step("deploy_complete", deployment=deployment_id)
            result = await _poll_result(client, deployment, ctx)
            deleted = await _cleanup_run(
                client,
                deployment,
                ctx,
                "completed" if isinstance(result, dict) else "failed",
            )
            if not deleted:
                raise EvaluationRetryableError(f"Could not verify Runpod cleanup for {deployment_id}")
            await _finish_evaluation_cost_run(
                run_key=cost_key,
                success=isinstance(result, dict),
                psql_db=psql_db,
                ctx=ctx,
            )
            if isinstance(result, dict):
                return result
            raise RuntimeError(str(result))
        except asyncio.CancelledError:
            deleted = True
            if deployment is not None:
                deleted = await asyncio.shield(_cleanup_run(client, deployment, ctx, "cancelled"))
            if deleted:
                await asyncio.shield(
                    _release_reserved_gpus(
                        task_id=task_id,
                        psql_db=psql_db,
                        hotkeys=reserved_hotkeys,
                        deployment_name=deployment_id if reserve_deployment_id else None,
                        ctx=ctx,
                    )
                )
                await asyncio.shield(
                    _finish_evaluation_cost_run(
                        run_key=cost_key,
                        success=False,
                        psql_db=psql_db,
                        ctx=ctx,
                    )
                )
            if not deleted:
                raise EvaluationRetryableError(
                    f"Could not verify Runpod cleanup for cancelled deployment {deployment_id}"
                )
            raise
        except EvaluationCapacityUnavailable as exc:
            deleted = True
            if deployment is not None:
                deleted = await _cleanup_run(client, deployment, ctx, "no_offers")
            if not deleted:
                raise EvaluationRetryableError(
                    f"Could not verify Runpod cleanup for {deployment_id}"
                ) from exc
            await _finish_evaluation_cost_run(
                run_key=cost_key,
                success=False,
                psql_db=psql_db,
                ctx=ctx,
            )
            await _release_reserved_gpus(
                task_id=task_id,
                psql_db=psql_db,
                hotkeys=reserved_hotkeys,
                deployment_name=deployment_id if reserve_deployment_id else None,
                ctx=ctx,
            )
            raise
        except Exception as exc:
            last_error = exc
            if deployment is not None:
                deleted = await _cleanup_run(client, deployment, ctx, "attempt_exception")
                if not deleted:
                    raise EvaluationRetryableError(
                        f"Could not verify Runpod cleanup for {deployment_id}"
                    ) from exc
            await _release_reserved_gpus(
                task_id=task_id,
                psql_db=psql_db,
                hotkeys=reserved_hotkeys,
                deployment_name=deployment_id if reserve_deployment_id else None,
                ctx=ctx,
            )
            await _finish_evaluation_cost_run(
                run_key=cost_key,
                success=False,
                psql_db=psql_db,
                ctx=ctx,
            )
            if isinstance(exc, EvaluationRetryableError):
                raise
            if attempt < vcst.EVAL_BASILICA_MAX_RETRIES:
                await asyncio.sleep(vcst.EVAL_BASILICA_RETRY_DELAY_SECONDS)
    return f"Evaluation failed after {vcst.EVAL_BASILICA_MAX_RETRIES} attempts: {last_error}"


async def run_runpod_eval_repos(
    *,
    repos: list[str],
    model_name: str,
    task_type: str,
    image: str,
    source: str,
    build_env_for_repo,
    gpu_count: int | None,
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    repo_to_hotkey: dict[str, str],
    deployment_ids_by_repo: dict[str, str] | None = None,
    local_logging: bool | None = False,
    persist_deployment_ids: bool = True,
    deployment_id_persister: Callable[[str, str], Awaitable[None]] | None = None,
    reserve_deployment_id: bool = False,
    propagate_retryable: bool = True,
    **_ignored,
) -> dict[str, dict | str | EvaluationRetryableError]:
    deployment_ids_by_repo = deployment_ids_by_repo or {}

    async def default_persister(repo: str, deployment_id: str) -> None:
        await persist_deployment_ids_for_repo(task_id, psql_db, repo_to_hotkey, repo, deployment_id)

    effective_persister = deployment_id_persister
    if effective_persister is None and persist_deployment_ids:
        effective_persister = default_persister
    results = await asyncio.gather(
        *[
            _run_single_runpod_eval_repo(
                repo=repo,
                model_name=model_name,
                task_type=task_type,
                image=image,
                source=source,
                env=build_env_for_repo(repo),
                gpu_count=gpu_count,
                task_id=task_id,
                psql_db=psql_db,
                repo_to_hotkey=repo_to_hotkey,
                hotkey=repo_to_hotkey.get(repo),
                existing_deployment_name=deployment_ids_by_repo.get(repo),
                local_logging=local_logging,
                deployment_id_persister=effective_persister,
                reserve_deployment_id=reserve_deployment_id,
            )
            for repo in repos
        ],
        return_exceptions=True,
    )
    output: dict[str, dict | str | EvaluationRetryableError] = {}
    for repo, result in zip(repos, results):
        if isinstance(result, EvaluationRetryableError):
            if propagate_retryable:
                raise result
            output[repo] = result
        elif isinstance(result, Exception):
            output[repo] = f"Evaluation failed: {result}"
        else:
            output[repo] = result
    return output


async def list_live_runpod_eval_deployments() -> list[RunpodDeployment]:
    """List active validator-owned Runpod eval services for reconciliation."""
    client = DstackClient()
    deployments: list[RunpodDeployment] = []
    for run in await client.list_active_runs():
        run_spec = run.get("run_spec") or {}
        run_name = str(run_spec.get("run_name") or "")
        if not run_name.startswith(RUNPOD_RUN_NAME_PREFIX):
            continue
        deployments.append(
            RunpodDeployment(
                run_name=run_name,
                deployment_id=encode_runpod_deployment_id(run_name),
                url=client.resolve_service_url(run),
                created_at=parse_dstack_submitted_at(run) or datetime.datetime.now(datetime.timezone.utc),
                control_token=_control_token(run_name),
                run=run,
            )
        )
    return deployments


async def cleanup_runpod_deployments_by_id(deployment_ids: set[str]) -> set[str]:
    """Best-effort cleanup used by the reconciler; returns verified deletions."""
    if not deployment_ids:
        return set()
    client = DstackClient()
    cleaned: set[str] = set()
    for deployment_id in deployment_ids:
        run_name = decode_runpod_deployment_id(deployment_id)
        if not run_name:
            continue
        try:
            run = await client.get_run(run_name)
        except Exception as exc:
            if _is_not_found(exc):
                cleaned.add(deployment_id)
            else:
                logger.warning("Runpod reconcile lookup failed for %s: %s", deployment_id, exc)
            continue
        eval_logger = get_logger(f"{__name__}.reconcile")
        ctx = _BasilicaEvalContext(
            repo="reconcile",
            eval_logger=eval_logger,
            deleted_deployment_names=set(),
            log_eval_step=lambda step, **fields: _log_eval_step(
                eval_logger,
                step,
                backend="runpod",
                **fields,
            ),
        )
        deployment = RunpodDeployment(
            run_name=run_name,
            deployment_id=deployment_id,
            url=client.resolve_service_url(run),
            created_at=parse_dstack_submitted_at(run) or datetime.datetime.now(datetime.timezone.utc),
            control_token=_control_token(run_name),
            run=run,
        )
        if await _cleanup_run(client, deployment, ctx, "reconcile_orphan"):
            cleaned.add(deployment_id)
    return cleaned
