from uuid import UUID

from asyncpg.connection import Connection

import validator.db.constants as cst
from core.constants import NETUID
from core.models.utility_models import TaskStatus
from validator.core.models import AllNodeStats
from validator.core.models import ModelMetrics
from validator.core.models import NodeStats
from validator.core.models import QualityMetrics
from validator.core.models import Submission
from validator.core.models import WorkloadMetrics
from validator.db.database import PSQLDB


async def add_submission(submission: Submission, psql_db: PSQLDB) -> Submission:
    """Add or update a submission for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.SUBMISSIONS_TABLE} (
                {cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}, {cst.REPO}
            )
            VALUES ($1, $2, $3, $4)
            ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID})
            DO UPDATE SET
                {cst.REPO} = EXCLUDED.{cst.REPO},
                updated_on = CURRENT_TIMESTAMP
            RETURNING {cst.SUBMISSION_ID}
        """
        submission_id = await connection.fetchval(
            query,
            submission.task_id,
            submission.hotkey,
            NETUID,
            submission.repo,
        )
        return await get_submission(submission_id, psql_db)


async def get_submission(submission_id: UUID, psql_db: PSQLDB) -> Submission | None:
    """Get a submission by its ID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {cst.SUBMISSIONS_TABLE} WHERE {cst.SUBMISSION_ID} = $1
        """
        row = await connection.fetchrow(query, submission_id)
        if row:
            return Submission(**dict(row))
        return None


async def set_task_node_quality_score(
    task_id: UUID,
    hotkey: str,
    quality_score: float,
    test_loss: float,
    synth_loss: float,
    psql_db: PSQLDB,
    score_reason: str | None = None,
) -> None:
    """Set quality score, losses and zero score reason for a node's task submission"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.TASK_NODES_TABLE} (
                {cst.TASK_ID},
                {cst.HOTKEY},
                {cst.NETUID},
                {cst.TASK_NODE_QUALITY_SCORE},
                {cst.TEST_LOSS},
                {cst.SYNTH_LOSS},
                {cst.SCORE_REASON}
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}) DO UPDATE
            SET
                {cst.TASK_NODE_QUALITY_SCORE} = $4,
                {cst.TEST_LOSS} = $5,
                {cst.SYNTH_LOSS} = $6,
                {cst.SCORE_REASON} = $7
        """
        await connection.execute(
            query,
            task_id,
            hotkey,
            NETUID,
            quality_score,
            test_loss,
            synth_loss,
            score_reason,
        )


async def set_task_node_losses(
    task_id: UUID,
    hotkey: str,
    test_loss: float | None,
    synth_loss: float | None,
    psql_db: PSQLDB,
    score_reason: str | None = None,
) -> None:
    """Persist raw evaluation outputs without final ranking."""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.TASK_NODES_TABLE} (
                {cst.TASK_ID},
                {cst.HOTKEY},
                {cst.NETUID},
                {cst.TEST_LOSS},
                {cst.SYNTH_LOSS},
                {cst.SCORE_REASON}
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}) DO UPDATE
            SET
                {cst.TEST_LOSS} = $4,
                {cst.SYNTH_LOSS} = $5,
                {cst.SCORE_REASON} = $6
        """
        await connection.execute(
            query,
            task_id,
            hotkey,
            NETUID,
            test_loss,
            synth_loss,
            score_reason,
        )


async def get_task_node_losses(task_id: UUID, psql_db: PSQLDB) -> list[dict]:
    """Get raw per-hotkey evaluation outputs regardless of final score state."""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT
                {cst.HOTKEY},
                {cst.TASK_NODE_QUALITY_SCORE},
                {cst.TEST_LOSS},
                {cst.SYNTH_LOSS},
                {cst.SCORE_REASON}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.NETUID} = $2
        """
        rows = await connection.fetch(query, task_id, NETUID)

        def clean_float(value):
            if value is None:
                return None
            if isinstance(value, float):
                if value in (float("inf"), float("-inf")) or value != value:
                    return None
            return value

        return [
            {
                cst.HOTKEY: row[cst.HOTKEY],
                cst.TASK_NODE_QUALITY_SCORE: clean_float(row[cst.TASK_NODE_QUALITY_SCORE]),
                cst.TEST_LOSS: clean_float(row[cst.TEST_LOSS]),
                cst.SYNTH_LOSS: clean_float(row[cst.SYNTH_LOSS]),
                cst.SCORE_REASON: row[cst.SCORE_REASON],
            }
            for row in rows
        ]


async def get_all_scores_and_losses_for_task(task_id: UUID, psql_db: PSQLDB) -> list[dict]:
    """Get all quality scores and losses for a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT
                {cst.HOTKEY},
                {cst.TASK_NODE_QUALITY_SCORE},
                {cst.TEST_LOSS},
                {cst.SYNTH_LOSS},
                {cst.SCORE_REASON}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.NETUID} = $2
            AND {cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL
        """
        rows = await connection.fetch(query, task_id, NETUID)

        def clean_float(value):
            if value is None:
                return None
            if isinstance(value, float):
                if value in (float("inf"), float("-inf")) or value != value:
                    return None
            return value

        return [
            {
                cst.HOTKEY: row[cst.HOTKEY],
                cst.TASK_NODE_QUALITY_SCORE: clean_float(row[cst.TASK_NODE_QUALITY_SCORE]),
                cst.TEST_LOSS: clean_float(row[cst.TEST_LOSS]),
                cst.SYNTH_LOSS: clean_float(row[cst.SYNTH_LOSS]),
                cst.SCORE_REASON: row[cst.SCORE_REASON],
            }
            for row in rows
        ]


async def get_all_scores_for_hotkey(hotkey: str, psql_db: PSQLDB) -> list[dict]:
    """
    Get all quality scores for a specific hotkey across all completed tasks.
    Excludes benchmark tasks from score calculations.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT
                tn.{cst.TASK_ID},
                tn.{cst.TASK_NODE_QUALITY_SCORE} as quality_score
            FROM {cst.TASK_NODES_TABLE} tn
            JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
            WHERE tn.{cst.HOTKEY} = $1
            AND tn.{cst.NETUID} = $2
            AND tn.{cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL
            AND t.{cst.STATUS} = $3
            AND t.{cst.TASK_ID} NOT IN (
                SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
                UNION
                SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
            )
        """
        rows = await connection.fetch(query, hotkey, NETUID, TaskStatus.SUCCESS.value)
        return [dict(row) for row in rows]


async def get_all_node_stats_batched(hotkeys: list[str], psql_db: PSQLDB) -> dict[str, AllNodeStats]:
    period_mapping = AllNodeStats.get_periods_sql_mapping()

    async with await psql_db.connection() as connection:
        query = f"""
        WITH periods AS (
            SELECT unnest($3::text[]) as interval
        ),
        model_counts AS (
            SELECT
                tn.{cst.HOTKEY},
                p.interval,
                t.{cst.MODEL_ID},
                COUNT(*) as model_count
            FROM periods p
            CROSS JOIN {cst.TASK_NODES_TABLE} tn
            JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
            WHERE tn.{cst.HOTKEY} = ANY($1)
            AND tn.{cst.NETUID} = $2
            AND tn.{cst.QUALITY_SCORE} IS NOT NULL
            AND t.{cst.TASK_ID} NOT IN (
                SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
                UNION
                SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
            )
            AND t.{cst.CREATED_AT} >= CASE
                WHEN p.interval = 'all' THEN '1970-01-01'::TIMESTAMP
                ELSE NOW() - p.interval::INTERVAL
            END
            GROUP BY tn.{cst.HOTKEY}, p.interval, t.{cst.MODEL_ID}
        ),
        aggregated_metrics AS (
            SELECT
                tn.{cst.HOTKEY},
                p.interval,
                -- Quality metrics
                COALESCE(AVG(tn.{cst.QUALITY_SCORE}), 0) as avg_quality_score,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > -1 THEN 1 END)::FLOAT / NULLIF(COUNT(*), 0), 0) as success_rate,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > 0 THEN 1 END)::FLOAT / NULLIF(COUNT(*), 0), 0) as quality_rate,
                COALESCE(COUNT(*), 0) as total_count,
                COALESCE(SUM(tn.{cst.QUALITY_SCORE}), 0) as total_score,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > -1 THEN 1 END), 0) as total_success,
                COALESCE(COUNT(CASE WHEN tn.{cst.QUALITY_SCORE} > 0 THEN 1 END), 0) as total_quality,

                -- Workload metrics
                COALESCE(SUM(t.{cst.HOURS_TO_COMPLETE}), 0)::INTEGER as competition_hours,
                COALESCE(SUM(
                    CASE
                        WHEN LOWER(t.{cst.MODEL_ID}) ~ '.*?([0-9]+\.?[0-9]*)[mb]' THEN
                            CASE
                                WHEN LOWER(t.{cst.MODEL_ID}) ~ '.*?([0-9]+\.?[0-9]*)b' THEN
                                    SUBSTRING(LOWER(t.{cst.MODEL_ID}) FROM '.*?([0-9]+\.?[0-9]*)b')::FLOAT
                                WHEN LOWER(t.{cst.MODEL_ID}) ~ '.*?([0-9]+\.?[0-9]*)m' THEN
                                    SUBSTRING(LOWER(t.{cst.MODEL_ID}) FROM '.*?([0-9]+\.?[0-9]*)m')::FLOAT / 1000.0
                            END
                        ELSE 1.0
                    END
                ), 0) as total_params_billions,

                -- Model metrics
                COALESCE((
                    SELECT mc.{cst.MODEL_ID}
                    FROM model_counts mc
                    WHERE mc.{cst.HOTKEY} = tn.{cst.HOTKEY}
                    AND mc.interval = p.interval
                    ORDER BY mc.model_count DESC
                    LIMIT 1
                ), 'none') as modal_model,
                COUNT(DISTINCT t.{cst.MODEL_ID}) as unique_models,
                COUNT(DISTINCT t.{cst.DS}) as unique_datasets

            FROM periods p
            CROSS JOIN {cst.TASK_NODES_TABLE} tn
            JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
            WHERE tn.{cst.HOTKEY} = ANY($1)
            AND tn.{cst.NETUID} = $2
            AND tn.{cst.QUALITY_SCORE} IS NOT NULL
            AND t.{cst.TASK_ID} NOT IN (
                SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
                UNION
                SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
            )
            AND t.{cst.CREATED_AT} >= CASE
                WHEN p.interval = 'all' THEN '1970-01-01'::TIMESTAMP
                ELSE NOW() - p.interval::INTERVAL
            END
            GROUP BY tn.{cst.HOTKEY}, p.interval
        )
        SELECT * FROM aggregated_metrics
        """

        intervals = list(period_mapping.values())
        rows = await connection.fetch(query, hotkeys, NETUID, intervals)

        results = {hotkey: {} for hotkey in hotkeys}
        for hotkey in hotkeys:
            for period_name in period_mapping.keys():
                results[hotkey][period_name] = NodeStats(
                    quality_metrics=QualityMetrics(
                        avg_quality_score=0,
                        success_rate=0,
                        quality_rate=0,
                        total_count=0,
                        total_score=0,
                        total_success=0,
                        total_quality=0,
                    ),
                    workload_metrics=WorkloadMetrics(competition_hours=0, total_params_billions=0),
                    model_metrics=ModelMetrics(modal_model="none", unique_models=0, unique_datasets=0),
                )

        for row in rows:
            hotkey = row[cst.HOTKEY]
            interval = row["interval"]

            stats = NodeStats(
                quality_metrics=QualityMetrics(
                    avg_quality_score=row["avg_quality_score"],
                    success_rate=row["success_rate"],
                    quality_rate=row["quality_rate"],
                    total_count=row["total_count"],
                    total_score=row["total_score"],
                    total_success=row["total_success"],
                    total_quality=row["total_quality"],
                ),
                workload_metrics=WorkloadMetrics(
                    competition_hours=row["competition_hours"], total_params_billions=row["total_params_billions"]
                ),
                model_metrics=ModelMetrics(
                    modal_model=row["modal_model"] or "none",
                    unique_models=row["unique_models"],
                    unique_datasets=row["unique_datasets"],
                ),
            )

            period_name = next(name for name, value in period_mapping.items() if value == interval)
            results[hotkey][period_name] = stats

        return {hotkey: AllNodeStats(**stats) for hotkey, stats in results.items()}


async def get_task_winner(task_id: UUID, psql_db: PSQLDB) -> str | None:
    """Get the winner of a task based on the best quality score (lowest loss)."""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.HOTKEY}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.NETUID} = $2
            AND {cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL AND {cst.TASK_NODE_QUALITY_SCORE} > 0
            ORDER BY {cst.TASK_NODE_QUALITY_SCORE} DESC  -- Higher score is better
            LIMIT 1
        """
        return await connection.fetchval(query, task_id, NETUID)


async def get_task_winners(task_ids: list[UUID], psql_db: PSQLDB) -> dict[str, str]:
    """Get winners for multiple tasks. Returns dict mapping task_id to winner hotkey."""
    if not task_ids:
        return {}

    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            WITH task_winners AS (
                SELECT 
                    {cst.TASK_ID}::text as task_id,
                    {cst.HOTKEY},
                    {cst.TASK_NODE_QUALITY_SCORE},
                    ROW_NUMBER() OVER (
                        PARTITION BY {cst.TASK_ID} 
                        ORDER BY {cst.TASK_NODE_QUALITY_SCORE} DESC  -- Higher score is better
                    ) as rn
                FROM {cst.TASK_NODES_TABLE}
                WHERE {cst.TASK_ID} = ANY($1)
                AND {cst.NETUID} = $2
                AND {cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL AND {cst.TASK_NODE_QUALITY_SCORE} > 0
            )
            SELECT task_id, {cst.HOTKEY}
            FROM task_winners
            WHERE rn = 1
        """

        rows = await connection.fetch(query, task_ids, NETUID)
        return {row["task_id"]: row[cst.HOTKEY] for row in rows}
