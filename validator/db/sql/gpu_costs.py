from datetime import datetime
from datetime import timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

from asyncpg import Connection

from core.models.trainer_contract_models import GPUInfo
from validator.db.database import PSQLDB
from validator.tournament import cost_constants


def hourly_rate_for_gpu(gpu_type: str) -> Decimal:
    normalized = gpu_type.upper()
    if "H100" in normalized:
        return cost_constants.H100_HOURLY_USD
    if "A100" in normalized:
        return cost_constants.A100_HOURLY_USD
    return Decimal(0)


async def start_cost_run(
    *,
    run_key: str,
    task_id: str,
    category: str,
    gpu_type: str,
    gpu_count: int,
    psql_db: PSQLDB,
    tournament_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    started_at: datetime | None = None,
) -> bool:
    """Start a billable run. A duplicate key is a no-op."""
    if category not in cost_constants.COST_CATEGORIES:
        raise ValueError(f"Unsupported GPU cost category: {category}")
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    task_uuid = UUID(str(task_id))

    async with await psql_db.connection() as connection:
        if tournament_id is None:
            tournament_id = await connection.fetchval(
                """
                SELECT tournament_id FROM tournament_tasks WHERE task_id = $1
                UNION ALL
                SELECT tournament_id FROM benchmark_task_copies WHERE copy_task_id = $1
                LIMIT 1
                """,
                task_uuid,
            )
        if tournament_id is None:
            return False

        result = await connection.execute(
            """
            INSERT INTO active_gpu_cost_runs
                (run_key, task_id, tournament_id, category, gpu_type, gpu_count,
                 hourly_rate_per_gpu_usd, started_at, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, COALESCE($8, CURRENT_TIMESTAMP), $9)
            ON CONFLICT (run_key) DO NOTHING
            """,
            run_key,
            task_uuid,
            tournament_id,
            category,
            gpu_type.upper(),
            gpu_count,
            hourly_rate_for_gpu(gpu_type),
            started_at,
            metadata or {},
        )
        return result.endswith("1")


async def finish_cost_run(
    *,
    run_key: str,
    success: bool,
    psql_db: PSQLDB,
    ended_at: datetime | None = None,
) -> dict[str, Any] | None:
    """Atomically consume an active run and add it to the task aggregate."""
    ended_at = ended_at or datetime.now(timezone.utc)
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            run = await connection.fetchrow(
                "DELETE FROM active_gpu_cost_runs WHERE run_key = $1 RETURNING *",
                run_key,
            )
            if run is None:
                return None

            wall_seconds = Decimal(str(max(0.0, (ended_at - run["started_at"]).total_seconds())))
            gpu_seconds = wall_seconds * Decimal(run["gpu_count"])
            cost_usd = gpu_seconds * run["hourly_rate_per_gpu_usd"] / Decimal(3600)
            category = run["category"]
            success_column = f"{category}_success_count"
            failure_column = f"{category}_failure_count"
            count_column = success_column if success else failure_column

            row = await connection.fetchrow(
                f"""
                INSERT INTO task_gpu_costs
                    (task_id, tournament_id, {category}_wall_seconds, {category}_gpu_seconds,
                     {category}_cost_usd, {count_column})
                VALUES ($1, $2, $3, $4, $5, 1)
                ON CONFLICT (task_id) DO UPDATE SET
                    {category}_wall_seconds = task_gpu_costs.{category}_wall_seconds + EXCLUDED.{category}_wall_seconds,
                    {category}_gpu_seconds = task_gpu_costs.{category}_gpu_seconds + EXCLUDED.{category}_gpu_seconds,
                    {category}_cost_usd = task_gpu_costs.{category}_cost_usd + EXCLUDED.{category}_cost_usd,
                    {count_column} = task_gpu_costs.{count_column} + 1,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING *
                """,
                run["task_id"],
                run["tournament_id"],
                wall_seconds,
                gpu_seconds,
                cost_usd,
            )
            return {
                "task_id": run["task_id"],
                "tournament_id": run["tournament_id"],
                "category": category,
                "wall_seconds": wall_seconds,
                "gpu_seconds": gpu_seconds,
                "cost_usd": cost_usd,
                "prep_failure_count": row["prep_failure_count"],
                "metadata": run["metadata"],
            }


async def reconcile_trainer_capacity(
    connection: Connection,
    trainer_ip: str,
    gpu_infos: list[GPUInfo],
) -> None:
    """Keep one open capacity interval per currently registered trainer GPU."""
    now = datetime.now(timezone.utc)
    current = {gpu.gpu_id: gpu for gpu in gpu_infos}
    open_rows = await connection.fetch(
        """
        SELECT id, gpu_id, gpu_type, vram_gb
        FROM trainer_gpu_capacity_intervals
        WHERE trainer_ip = $1 AND ended_at IS NULL
        FOR UPDATE
        """,
        trainer_ip,
    )
    open_by_gpu = {row["gpu_id"]: row for row in open_rows}

    for gpu_id, row in open_by_gpu.items():
        gpu = current.get(gpu_id)
        if gpu is None or gpu.gpu_type != row["gpu_type"] or gpu.vram_gb != row["vram_gb"]:
            await connection.execute(
                "UPDATE trainer_gpu_capacity_intervals SET ended_at = $2 WHERE id = $1",
                row["id"],
                now,
            )

    for gpu_id, gpu in current.items():
        row = open_by_gpu.get(gpu_id)
        if row is not None and gpu.gpu_type == row["gpu_type"] and gpu.vram_gb == row["vram_gb"]:
            continue
        await connection.execute(
            """
            INSERT INTO trainer_gpu_capacity_intervals
                (trainer_ip, gpu_id, gpu_type, vram_gb, started_at)
            VALUES ($1, $2, $3, $4, $5)
            """,
            trainer_ip,
            gpu_id,
            gpu.gpu_type,
            gpu.vram_gb,
            now,
        )


async def close_trainer_capacity(connection: Connection, trainer_ip: str) -> None:
    await connection.execute(
        """
        UPDATE trainer_gpu_capacity_intervals
        SET ended_at = CURRENT_TIMESTAMP
        WHERE trainer_ip = $1 AND ended_at IS NULL
        """,
        trainer_ip,
    )


async def get_weekly_cost_rows(
    *,
    window_start: datetime,
    window_end: datetime,
    psql_db: PSQLDB,
    tournament_window_end: datetime | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return raw rows used by the API aggregation layer."""
    tournament_window_end = tournament_window_end or window_end
    async with await psql_db.connection() as connection:
        tournament_rows = await connection.fetch(
            """
            SELECT tournament_id, tournament_type, status, created_at, updated_at
            FROM tournaments
            WHERE created_at >= $1 AND created_at < $2
            ORDER BY created_at
            """,
            window_start,
            tournament_window_end,
        )
        tournament_ids = [row["tournament_id"] for row in tournament_rows]
        if tournament_ids:
            task_rows = await connection.fetch(
                """
                SELECT task_id, tournament_id
                FROM tournament_tasks
                WHERE tournament_id = ANY($1::text[])
                """,
                tournament_ids,
            )
            cost_rows = await connection.fetch(
                "SELECT * FROM task_gpu_costs WHERE tournament_id = ANY($1::text[])",
                tournament_ids,
            )
            active_rows = await connection.fetch(
                "SELECT * FROM active_gpu_cost_runs WHERE tournament_id = ANY($1::text[])",
                tournament_ids,
            )
            hotkey_rows = await connection.fetch(
                """
                SELECT tt.task_id, COUNT(DISTINCT th.hotkey) AS hotkey_count
                FROM tournament_tasks tt
                LEFT JOIN tournament_task_hotkey_trainings th ON th.task_id = tt.task_id
                WHERE tt.tournament_id = ANY($1::text[])
                GROUP BY tt.task_id
                """,
                tournament_ids,
            )
        else:
            task_rows = []
            cost_rows = []
            active_rows = []
            hotkey_rows = []

        capacity_rows = await connection.fetch(
            """
            SELECT trainer_ip, gpu_id, gpu_type, vram_gb, started_at, ended_at
            FROM trainer_gpu_capacity_intervals
            WHERE started_at < $2 AND COALESCE(ended_at, $2) > $1
            """,
            window_start,
            window_end,
        )

    return {
        "tournaments": [dict(row) for row in tournament_rows],
        "tasks": [dict(row) for row in task_rows],
        "costs": [dict(row) for row in cost_rows],
        "active": [dict(row) for row in active_rows],
        "hotkeys": [dict(row) for row in hotkey_rows],
        "capacity": [dict(row) for row in capacity_rows],
    }
