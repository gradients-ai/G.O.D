from datetime import datetime
from datetime import timezone
from decimal import Decimal
from typing import Any
from uuid import UUID
from uuid import uuid4

from asyncpg import Connection

from core.models.trainer_contract_models import GPUInfo
from validator.db.database import PSQLDB
from validator.scoring.constants import EMISSION_BURN_HOTKEY
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
    run_id: UUID | None = None,
) -> bool:
    """Start a run, closing any stale run for the same logical work first."""
    if category not in cost_constants.COST_CATEGORIES:
        raise ValueError(f"Unsupported GPU cost category: {category}")
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    task_uuid = UUID(str(task_id))
    run_id = run_id or uuid4()

    started_at = started_at or datetime.now(timezone.utc)
    async with await psql_db.connection() as connection:
        async with connection.transaction():
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

            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                run_key,
            )
            if await connection.fetchval(
                "SELECT EXISTS (SELECT 1 FROM gpu_usage_runs WHERE run_id = $1)",
                run_id,
            ):
                return True
            # A manual retry or restart may leave an old run open. Close it at
            # the new run's start so time is never double counted.
            await _finish_active_run(connection, run_key, False, started_at)
            run_id = await connection.fetchval(
                """
                INSERT INTO gpu_usage_runs
                    (run_id, source_key, task_id, tournament_id, category, gpu_type, gpu_count,
                     hourly_rate_per_gpu_usd, started_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING run_id
                """,
                run_id,
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
            return run_id is not None


async def _finish_active_run(
    connection: Connection,
    source_key: str,
    success: bool,
    ended_at: datetime,
):
    return await connection.fetchrow(
        """
        UPDATE gpu_usage_runs
        SET ended_at = GREATEST($2, started_at),
            outcome = $3,
            wall_seconds = GREATEST(EXTRACT(EPOCH FROM ($2 - started_at)), 0),
            gpu_seconds = GREATEST(EXTRACT(EPOCH FROM ($2 - started_at)), 0) * gpu_count,
            cost_usd = (
                GREATEST(EXTRACT(EPOCH FROM ($2 - started_at)), 0)
                * gpu_count * hourly_rate_per_gpu_usd / 3600
            )
        WHERE run_id = (
            SELECT run_id
            FROM gpu_usage_runs
            WHERE source_key = $1 AND ended_at IS NULL
            ORDER BY started_at DESC
            LIMIT 1
            FOR UPDATE
        )
        RETURNING *
        """,
        source_key,
        ended_at,
        "success" if success else "failure",
    )


async def finish_cost_run(
    *,
    run_key: str,
    success: bool,
    psql_db: PSQLDB,
    ended_at: datetime | None = None,
) -> dict[str, Any] | None:
    """Finish the currently active run for a logical work identity."""
    ended_at = ended_at or datetime.now(timezone.utc)
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                run_key,
            )
            run = await _finish_active_run(connection, run_key, success, ended_at)
            if run is None:
                return None
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                str(run["task_id"]),
            )
            prep_failure_count = await connection.fetchval(
                """
                SELECT COUNT(*)
                FROM gpu_usage_runs
                WHERE task_id = $1 AND category = 'prep' AND outcome = 'failure'
                """,
                run["task_id"],
            )
            should_notify_prep_failure = False
            if run["category"] == "prep" and not success and prep_failure_count >= 3:
                alert_already_sent = await connection.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM gpu_usage_runs
                        WHERE task_id = $1
                          AND category = 'prep'
                          AND metadata @> '{"prep_failure_alert_sent": true}'::jsonb
                    )
                    """,
                    run["task_id"],
                )
                if not alert_already_sent:
                    await connection.execute(
                        """
                        UPDATE gpu_usage_runs
                        SET metadata = metadata || '{"prep_failure_alert_sent": true}'::jsonb
                        WHERE run_id = $1
                        """,
                        run["run_id"],
                    )
                    should_notify_prep_failure = True
            return {
                "task_id": run["task_id"],
                "tournament_id": run["tournament_id"],
                "category": run["category"],
                "wall_seconds": run["wall_seconds"],
                "gpu_seconds": run["gpu_seconds"],
                "cost_usd": run["cost_usd"],
                "prep_failure_count": prep_failure_count,
                "should_notify_prep_failure": should_notify_prep_failure,
                "metadata": run["metadata"],
            }


async def close_stale_evaluation_runs(
    *,
    live_deployment_names: set[str],
    older_than: datetime,
    psql_db: PSQLDB,
) -> int:
    """Close abandoned eval runs after Basilica reconciliation proves them stale."""
    async with await psql_db.connection() as connection:
        result = await connection.execute(
            """
            UPDATE gpu_usage_runs
            SET ended_at = CURRENT_TIMESTAMP,
                outcome = 'failure',
                wall_seconds = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - started_at)),
                gpu_seconds = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - started_at)) * gpu_count,
                cost_usd = (
                    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - started_at))
                    * gpu_count * hourly_rate_per_gpu_usd / 3600
                )
            WHERE category = 'evaluation'
              AND ended_at IS NULL
              AND started_at < $1
              AND NOT ((metadata->>'deployment_name') = ANY($2::text[]))
            """,
            older_than,
            list(live_deployment_names),
        )
        return int(result.split()[-1])


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
                SELECT tt.task_id, tt.tournament_id,
                       t.task_type::text AS task_type,
                       t.model_id AS base_model,
                       tr.round_number, tr.round_type
                FROM tournament_tasks tt
                JOIN tasks t ON t.task_id = tt.task_id
                LEFT JOIN tournament_rounds tr ON tr.round_id = tt.round_id
                WHERE tt.tournament_id = ANY($1::text[])
                """,
                tournament_ids,
            )
            # The defending champion ("boss") is carried in for free as the base
            # contestant under EMISSION_BURN_HOTKEY, so exclude it from paying entrants.
            participant_rows = await connection.fetch(
                """
                SELECT tournament_id, COUNT(*) AS participant_count
                FROM tournament_participants
                WHERE tournament_id = ANY($1::text[])
                  AND hotkey != $2
                GROUP BY tournament_id
                """,
                tournament_ids,
                EMISSION_BURN_HOTKEY,
            )
            run_rows = await connection.fetch(
                """
                SELECT *
                FROM gpu_usage_runs
                WHERE tournament_id = ANY($1::text[])
                  AND started_at < $3
                  AND COALESCE(ended_at, $3) > $2
                """,
                tournament_ids,
                window_start,
                window_end,
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
            run_rows = []
            hotkey_rows = []
            participant_rows = []

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
        "runs": [dict(row) for row in run_rows],
        "hotkeys": [dict(row) for row in hotkey_rows],
        "participants": [dict(row) for row in participant_rows],
        "capacity": [dict(row) for row in capacity_rows],
    }
