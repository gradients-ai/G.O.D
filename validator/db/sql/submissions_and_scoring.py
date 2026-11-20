import math
from uuid import UUID

from asyncpg.connection import Connection

import validator.db.constants as cst
from core.constants.network import NETUID
from core.logging import get_logger
from core.models.quant_artifact_models import QuantDecodedFile
from core.models.quant_models import QuantEvalGpu
from core.models.task_models import TaskStatus
from validator.db.database import PSQLDB
from validator.nodes.models import AllNodeStats
from validator.nodes.models import ModelMetrics
from validator.nodes.models import NodeStats
from validator.nodes.models import QualityMetrics
from validator.nodes.models import WorkloadMetrics
from validator.scoring.models import HubArtifactFile
from validator.scoring.models import QuantScorecardRow
from validator.scoring.models import Submission
from validator.scoring.quant_ranking import eval_gpu_from_column
from validator.tournament.models import QuantDecodeRow


logger = get_logger(__name__)


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
    passed_bar: bool | None = None,
    quant_failure: str | None = None,
    freeform_penalty: float | None = None,
) -> None:
    """Set quality score, losses, score reason, freeform cap status and penalty (ln P as scored) for a node's
    task submission.

    Deliberately does NOT touch the per-example loss columns. This runs later in the lifecycle than
    set_task_node_losses, and on the finalize path its results are rebuilt from the DB row without
    the vector - writing it here would NULL out what the raw-loss write persisted, leaving the boss
    round to silently fall back to the scalar margin.

    quant_failure is only ever added (COALESCE): a finalize-time scored fail is written, and a
    persisted one (e.g. from the decode job) is never erased by a result that carries none.
    """
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
                {cst.SCORE_REASON},
                {cst.PASSED_BAR},
                {cst.QUANT_FAILURE},
                {cst.FREEFORM_PENALTY}
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}) DO UPDATE
            SET
                {cst.TASK_NODE_QUALITY_SCORE} = $4,
                {cst.TEST_LOSS} = $5,
                {cst.SYNTH_LOSS} = $6,
                {cst.SCORE_REASON} = $7,
                {cst.PASSED_BAR} = $8,
                {cst.QUANT_FAILURE} = COALESCE($9, {cst.TASK_NODES_TABLE}.{cst.QUANT_FAILURE}),
                {cst.FREEFORM_PENALTY} = $10
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
            passed_bar,
            quant_failure,
            freeform_penalty,
        )


async def set_task_node_quant_failure(task_id: UUID, hotkey: str, quant_failure: str, psql_db: PSQLDB) -> None:
    """Persist a scored fail on an existing task_nodes row without touching any other column."""
    async with await psql_db.connection() as connection:
        connection: Connection
        result = await connection.execute(
            f"""
            UPDATE {cst.TASK_NODES_TABLE}
            SET {cst.QUANT_FAILURE} = $4
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2 AND {cst.NETUID} = $3
            """,
            task_id,
            hotkey,
            NETUID,
            quant_failure,
        )
    if result == "UPDATE 0":
        logger.warning(f"No task_nodes row for task {task_id} hotkey {hotkey}; quant failure not persisted")


async def get_per_example_losses(task_id: UUID, hotkey: str, psql_db: PSQLDB) -> tuple[list[float] | None, str | None]:
    """Read back a node's per-example loss vector and the fingerprint of the set it was scored on."""
    async with await psql_db.connection() as connection:
        connection: Connection
        row = await connection.fetchrow(
            f"""
            SELECT {cst.PER_EXAMPLE_LOSSES}, {cst.EVAL_SET_FINGERPRINT}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2 AND {cst.NETUID} = $3
            """,
            task_id,
            hotkey,
            NETUID,
        )
        if row is None or row[cst.PER_EXAMPLE_LOSSES] is None:
            return None, None

        return losses_from_column(row[cst.PER_EXAMPLE_LOSSES]), row[cst.EVAL_SET_FINGERPRINT]


async def update_task_node_quality_score_only(
    task_id: UUID,
    hotkey: str,
    quality_score: float,
    psql_db: PSQLDB,
    score_reason: str | None = None,
) -> None:
    """Update final score fields without changing persisted evaluator losses."""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            UPDATE {cst.TASK_NODES_TABLE}
            SET
                {cst.TASK_NODE_QUALITY_SCORE} = $3,
                {cst.SCORE_REASON} = $4
            WHERE {cst.TASK_ID} = $1
            AND {cst.HOTKEY} = $2
            AND {cst.NETUID} = $5
        """
        await connection.execute(
            query,
            task_id,
            hotkey,
            quality_score,
            score_reason,
            NETUID,
        )


async def get_task_node_eval_gpus(task_id: UUID, hotkeys: list[str], psql_db: PSQLDB) -> dict[str, QuantEvalGpu | None]:
    """The GPU each hotkey's quant row was scored on (None for a non-quant row or one with no recorded GPU)."""
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            SELECT {cst.HOTKEY}, {cst.EVAL_GPU}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = ANY($2) AND {cst.NETUID} = $3
            """,
            task_id,
            hotkeys,
            NETUID,
        )
    return {row[cst.HOTKEY]: eval_gpu_from_column(row[cst.EVAL_GPU]) for row in rows}


def json_safe_losses(losses: list[float] | None) -> list[float | None] | None:
    """Replace non-finite entries with JSON null.

    The list binds through the pooled jsonb codec, which uses json.dumps - and json.dumps emits a
    bare NaN token that PostgreSQL's json parser rejects outright. A single unscorable example
    would otherwise fail the whole INSERT, and the exception propagates far enough to mark every
    hotkey in the batch as an evaluation failure, discarding a completed GPU run. Nulls read back
    as NaN (losses_from_column) and the paired comparison already drops non-finite entries from both
    sides. The quant scorer keeps NaN in place for an example with no supervised token, so a bf16
    reference vector (quant_references) needs this as much as a submission's.
    """
    if losses is None:
        return None
    return [loss if math.isfinite(loss) else None for loss in losses]


def losses_from_column(values: list[float | None]) -> list[float]:
    """A stored per-example vector with its JSON nulls back as NaN, in example order."""
    return [float("nan") if loss is None else float(loss) for loss in values]


async def set_task_node_losses(
    task_id: UUID,
    hotkey: str,
    test_loss: float | None,
    synth_loss: float | None,
    psql_db: PSQLDB,
    score_reason: str | None = None,
    per_example_losses: list[float] | None = None,
    eval_set_fingerprint: str | None = None,
    artifact_bytes: int | None = None,
    decoder_bytes: int | None = None,
    quant_failure: str | None = None,
    eval_gpu: QuantEvalGpu | None = None,
    artifact_files: list[HubArtifactFile] | None = None,
    artifact_revision: str | None = None,
) -> None:
    """Persist raw evaluation outputs without final ranking.

    artifact_files / artifact_revision (freeform hub listing) never overwrite a stored value with NULL.

    The only write of raw evaluator losses, and so the only place the per-example vector is stored.
    Passed only for head-to-head tasks (see is_paired_comparison_task) since those are its
    only consumer. The list binds directly: a jsonb codec is registered on every pooled connection
    (validator/db/database.py), so serialising here would double-encode it.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.TASK_NODES_TABLE} (
                {cst.TASK_ID},
                {cst.HOTKEY},
                {cst.NETUID},
                {cst.TEST_LOSS},
                {cst.SYNTH_LOSS},
                {cst.SCORE_REASON},
                {cst.PER_EXAMPLE_LOSSES},
                {cst.EVAL_SET_FINGERPRINT},
                {cst.ARTIFACT_BYTES},
                {cst.DECODER_BYTES},
                {cst.QUANT_FAILURE},
                {cst.EVAL_GPU},
                {cst.ARTIFACT_FILES},
                {cst.ARTIFACT_REVISION}
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12::jsonb, $13::jsonb, $14)
            ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}) DO UPDATE
            SET
                {cst.TEST_LOSS} = $4,
                {cst.SYNTH_LOSS} = $5,
                {cst.SCORE_REASON} = $6,
                {cst.PER_EXAMPLE_LOSSES} = $7::jsonb,
                {cst.EVAL_SET_FINGERPRINT} = $8,
                {cst.ARTIFACT_BYTES} = $9,
                {cst.DECODER_BYTES} = $10,
                {cst.QUANT_FAILURE} = $11,
                {cst.EVAL_GPU} = $12::jsonb,
                {cst.ARTIFACT_FILES} = COALESCE($13::jsonb, {cst.TASK_NODES_TABLE}.{cst.ARTIFACT_FILES}),
                {cst.ARTIFACT_REVISION} = COALESCE($14, {cst.TASK_NODES_TABLE}.{cst.ARTIFACT_REVISION})
        """
        await connection.execute(
            query,
            task_id,
            hotkey,
            NETUID,
            test_loss,
            synth_loss,
            score_reason,
            json_safe_losses(per_example_losses),
            eval_set_fingerprint,
            artifact_bytes,
            decoder_bytes,
            quant_failure,
            eval_gpu.model_dump() if eval_gpu else None,
            [f.model_dump() for f in artifact_files] if artifact_files else None,
            artifact_revision,
        )


async def set_quant_decode_result(
    task_id: UUID,
    hotkey: str,
    *,
    artifact_bytes: int | None,
    decoder_bytes: int | None,
    decoder_sha256: str | None,
    quant_failure: str | None,
    psql_db: PSQLDB,
    build_warning: str | None = None,
    artifact_files: list[QuantDecodedFile] | None = None,
    decoded_files: list[QuantDecodedFile] | None = None,
) -> None:
    """Persist a freeform entry's trainer-side decode outcome on its task_nodes row (written at assignment).

    artifact_files / decoded_files are the decode's file lists (JSONB, codec-bound like per_example_losses); an empty
    or missing list is stored as NULL, the same as a row with no stored file lists.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            UPDATE {cst.TASK_NODES_TABLE}
            SET {cst.ARTIFACT_BYTES} = $4,
                {cst.DECODER_BYTES} = $5,
                {cst.DECODER_SHA256} = $6,
                {cst.QUANT_FAILURE} = $7,
                {cst.DECODER_BUILD_WARNING} = $8,
                {cst.DECODE_ARTIFACT_FILES} = $9::jsonb,
                {cst.DECODED_FILES} = $10::jsonb
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2 AND {cst.NETUID} = $3
        """
        result = await connection.execute(
            query,
            task_id,
            hotkey,
            NETUID,
            artifact_bytes,
            decoder_bytes,
            decoder_sha256,
            quant_failure,
            build_warning,
            [f.model_dump() for f in artifact_files] if artifact_files else None,
            [f.model_dump() for f in decoded_files] if decoded_files else None,
        )
    if result == "UPDATE 0":
        logger.warning(f"No task_nodes row for task {task_id} hotkey {hotkey}; decode result not persisted")


async def set_decoded_hf_repo(task_id: UUID | str, hotkey: str, repo: str, psql_db: PSQLDB) -> None:
    """Record the public HF copy of a freeform entry's decoded model on its task_nodes row."""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            UPDATE {cst.TASK_NODES_TABLE}
            SET {cst.DECODED_HF_REPO} = $4
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2 AND {cst.NETUID} = $3
        """
        await connection.execute(query, task_id, hotkey, NETUID, repo)


async def get_quant_decode_rows(task_id: UUID, psql_db: PSQLDB) -> dict[str, QuantDecodeRow]:
    """Decode outcome per hotkey for a freeform task.

    set_task_node_losses later overwrites artifact_bytes, decoder_bytes and quant_failure with the
    evaluation's values but never touches decoder_sha256, which is why QuantDecodeRow.settled keys
    off decoder_sha256 (or a persisted quant_failure) rather than the byte counts.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.HOTKEY}, {cst.ARTIFACT_BYTES}, {cst.DECODER_BYTES}, {cst.DECODER_SHA256}, {cst.QUANT_FAILURE}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2
        """
        rows = await connection.fetch(query, task_id, NETUID)
    return {
        row[cst.HOTKEY]: QuantDecodeRow(
            hotkey=row[cst.HOTKEY],
            artifact_bytes=row[cst.ARTIFACT_BYTES],
            decoder_bytes=row[cst.DECODER_BYTES],
            decoder_sha256=row[cst.DECODER_SHA256],
            quant_failure=row[cst.QUANT_FAILURE],
        )
        for row in rows
    }


async def get_quant_scorecard_rows(task_id: UUID | str, psql_db: PSQLDB) -> dict[str, QuantScorecardRow]:
    """The scorecard inputs of every task_nodes row of a freeform task, by hotkey (repo from its latest submission)."""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT tn.{cst.HOTKEY}, tn.{cst.TEST_LOSS}, tn.{cst.QUANT_FAILURE}, tn.{cst.ARTIFACT_BYTES},
                   tn.{cst.DECODER_BYTES}, tn.{cst.PASSED_BAR}, tn.{cst.FREEFORM_PENALTY}, tn.{cst.EVAL_GPU},
                   tn.{cst.ARTIFACT_FILES}, tn.{cst.ARTIFACT_REVISION}, tn.{cst.DECODE_ARTIFACT_FILES},
                   tn.{cst.DECODED_FILES}, tn.{cst.DECODED_HF_REPO}, s.{cst.REPO}
            FROM {cst.TASK_NODES_TABLE} tn
            LEFT JOIN LATERAL (
                SELECT sub.{cst.REPO}
                FROM {cst.SUBMISSIONS_TABLE} sub
                WHERE sub.{cst.TASK_ID} = tn.{cst.TASK_ID} AND sub.{cst.HOTKEY} = tn.{cst.HOTKEY}
                ORDER BY sub.{cst.CREATED_ON} DESC NULLS LAST, sub.{cst.SUBMISSION_ID} DESC
                LIMIT 1
            ) s ON TRUE
            WHERE tn.{cst.TASK_ID} = $1 AND tn.{cst.NETUID} = $2
        """
        rows = await connection.fetch(query, task_id, NETUID)
    return {row[cst.HOTKEY]: QuantScorecardRow.model_validate(dict(row)) for row in rows}


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
                {cst.SCORE_REASON},
                {cst.ARTIFACT_BYTES},
                {cst.DECODER_BYTES},
                {cst.DECODER_SHA256},
                {cst.PASSED_BAR},
                {cst.FREEFORM_PENALTY},
                {cst.QUANT_FAILURE},
                {cst.EVAL_GPU}
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
                cst.ARTIFACT_BYTES: row[cst.ARTIFACT_BYTES],
                cst.DECODER_BYTES: row[cst.DECODER_BYTES],
                cst.DECODER_SHA256: row[cst.DECODER_SHA256],
                cst.PASSED_BAR: row[cst.PASSED_BAR],
                cst.FREEFORM_PENALTY: clean_float(row[cst.FREEFORM_PENALTY]),
                cst.QUANT_FAILURE: row[cst.QUANT_FAILURE],
                cst.EVAL_GPU: row[cst.EVAL_GPU],  # QuantEvalGpu JSON, parsed by the quant scorer
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
                {cst.SCORE_REASON},
                {cst.ARTIFACT_BYTES},
                {cst.DECODER_BYTES},
                {cst.DECODER_SHA256},
                {cst.PASSED_BAR},
                {cst.FREEFORM_PENALTY},
                {cst.QUANT_FAILURE},
                {cst.EVAL_GPU}
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
                cst.ARTIFACT_BYTES: row[cst.ARTIFACT_BYTES],
                cst.DECODER_BYTES: row[cst.DECODER_BYTES],
                cst.DECODER_SHA256: row[cst.DECODER_SHA256],
                cst.PASSED_BAR: row[cst.PASSED_BAR],
                cst.FREEFORM_PENALTY: clean_float(row[cst.FREEFORM_PENALTY]),
                cst.QUANT_FAILURE: row[cst.QUANT_FAILURE],
                cst.EVAL_GPU: row[cst.EVAL_GPU],  # QuantEvalGpu JSON, parsed by the quant scorer
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


async def get_eligible_hotkeys_for_task(task_id: UUID, psql_db: PSQLDB) -> set[str]:
    """Hotkeys whose submission is eligible to win a task: a NON-NEGATIVE persisted quality score.

    The discriminator is the penalty, not the win. A submission rejected as a non-finetune carries
    SCORE_PENALTY (-1); the legitimate runner-up of a two-way contest carries 0.0, because
    calculate_miner_ranking_and_scores awards FIRST_PLACE_SCORE to rank 1 and leaves everyone else
    untouched below MIN_IDEAL_NUM_MINERS_IN_POOL. So `> 0` would drop the runner-up of every
    knockout and make the paired path unreachable, while `>= 0` separates penalised from merely
    second exactly.

    Returns nothing when no submission scored above zero: calculate_miner_ranking_and_scores
    returns early with everyone at 0.0 when there are no valid finetunes at all, and that must not
    read as two eligible competitors.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            SELECT {cst.HOTKEY}, {cst.TASK_NODE_QUALITY_SCORE}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.NETUID} = $2
            AND {cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL AND {cst.TASK_NODE_QUALITY_SCORE} >= 0
            """,
            task_id,
            NETUID,
        )
        if not any(row[cst.TASK_NODE_QUALITY_SCORE] > 0 for row in rows):
            return set()
        return {row[cst.HOTKEY] for row in rows}


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
