from typing import TYPE_CHECKING
from typing import Any
from uuid import UUID

from core.logging import get_logger


if TYPE_CHECKING:
    from core.models.task_models import TaskType
    from validator.app.config import Config


logger = get_logger(__name__)


async def notify_evaluation_exception(
    config: "Config | None",
    *,
    task_id: str,
    task_type: "TaskType | Any",
    context: str,
    error: Exception | str,
    hotkeys: list[str] | None = None,
    repos: list[str] | None = None,
    deployment_ids: list[str] | None = None,
) -> None:
    if config is None or not config.discord_url:
        return

    try:
        import httpx

        details = str(error)
        if len(details) > 900:
            details = f"{details[:900]}..."
        task_type_value = getattr(task_type, "value", str(task_type))

        def _format_items(items: list[str], limit: int = 12) -> str:
            shown = items[:limit]
            suffix = f", ... (+{len(items) - limit} more)" if len(items) > limit else ""
            return f"{', '.join(shown)}{suffix}"

        lines = [
            "Evaluation exception",
            f"Task: {task_id}",
            f"Type: {task_type_value}",
            f"Context: {context}",
        ]
        if hotkeys:
            lines.append(f"Hotkeys: {_format_items(hotkeys)}")
        if repos:
            lines.append(f"Repos: {_format_items(repos)}")
        if deployment_ids:
            lines.append(f"Deployment IDs: {_format_items(deployment_ids)}")
        lines.append(f"Error: {details}")

        async with httpx.AsyncClient() as client:
            await client.post(config.discord_url, json={"content": "\n".join(lines)}, timeout=10)
    except Exception as notify_exc:
        logger.error(f"Failed to send Discord evaluation exception notification: {notify_exc}")


async def task_deployment_ids_for_hotkeys(
    task_id: UUID,
    config: "Config | None",
    hotkeys: list[str],
) -> list[str]:
    if config is None or config.psql_db is None or not hotkeys:
        return []

    hotkey_set = set(hotkeys)
    try:
        from validator.db.sql.tasks import get_task_evaluation_rows

        rows = await get_task_evaluation_rows(task_id, config.psql_db)
    except Exception as exc:
        logger.warning(f"Failed to load evaluation deployment IDs for task {task_id}: {exc}")
        return []

    deployment_ids = {
        row.get("deployment_id")
        for row in rows
        if row.get("hotkey") in hotkey_set and row.get("deployment_id")
    }
    return sorted(deployment_ids)
