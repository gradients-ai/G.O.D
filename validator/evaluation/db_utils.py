"""Evaluation helpers that depend on PostgreSQL (asyncpg). Kept separate so
modules like eval_instruct_text can import LoRA/utils without pulling DB deps."""

from uuid import UUID

from validator.db.database import PSQLDB
from validator.db.sql import tasks as tasks_sql
from validator.infrastructure.service_constants import RAYONLABS_HF_USERNAME


async def load_eval_pair_state_for_models(
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    models: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    if task_id is None or psql_db is None:
        return {}, {}

    rows = await tasks_sql.get_task_evaluation_rows(task_id, psql_db)
    model_set = set(models)
    deployment_ids_by_repo: dict[str, str] = {}
    repo_to_hotkey: dict[str, str] = {}

    for row in rows:
        expected_repo_name = row.get("expected_repo_name")
        hotkey = row.get("hotkey")
        if not expected_repo_name or not hotkey:
            continue
        repo = f"{RAYONLABS_HF_USERNAME}/{expected_repo_name}"
        if repo not in model_set:
            continue
        repo_to_hotkey[repo] = hotkey
        deployment_id = row.get("deployment_id")
        if deployment_id:
            deployment_ids_by_repo[repo] = deployment_id

    return deployment_ids_by_repo, repo_to_hotkey


async def persist_deployment_ids_for_repo(
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    repo_to_hotkey: dict[str, str],
    repo: str,
    deployment_id: str | None,
) -> None:
    if task_id is None or psql_db is None:
        return
    hotkey = repo_to_hotkey.get(repo)
    if not hotkey:
        return
    await tasks_sql.set_evaluation_deployment_id(task_id, hotkey, deployment_id, psql_db)


async def load_shared_eval_deployment_id(
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    hotkeys: list[str],
) -> str | None:
    if task_id is None or psql_db is None or not hotkeys:
        return None

    hotkey_set = set(hotkeys)
    rows = await tasks_sql.get_task_evaluation_rows(task_id, psql_db)
    deployment_ids = {
        row.get("deployment_id")
        for row in rows
        if row.get("hotkey") in hotkey_set and row.get("deployment_id")
    }
    if len(deployment_ids) == 1:
        return next(iter(deployment_ids))
    return None


async def persist_shared_eval_deployment_id(
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    hotkeys: list[str],
    deployment_id: str | None,
) -> None:
    if task_id is None or psql_db is None:
        return

    for hotkey in hotkeys:
        await tasks_sql.set_evaluation_deployment_id(task_id, hotkey, deployment_id, psql_db)
