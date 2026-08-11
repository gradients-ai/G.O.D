"""DB access for the majority-training-failure review gate."""

import json
from datetime import datetime
from datetime import timezone
from typing import Any
from uuid import UUID

import validator.db.constants as cst
from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.tournament.models import TaskFailureReviewStatus
from validator.tournament.models import TournamentTaskFailureReview


logger = get_logger(__name__)


def _loads(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _row_to_review(row: Any) -> TournamentTaskFailureReview:
    return TournamentTaskFailureReview(
        task_id=str(row["task_id"]),
        tournament_id=row["tournament_id"],
        round_id=row["round_id"],
        status=TaskFailureReviewStatus(row["status"]),
        failed_hotkeys=_loads(row["failed_hotkeys"]) or [],
        total_trainings=row["total_trainings"],
        notes=row["notes"],
        created_at=row["created_at"],
        reviewed_at=row["reviewed_at"],
    )


async def insert_task_failure_review(review: TournamentTaskFailureReview, psql_db: PSQLDB) -> bool:
    """Open the gate for a task. Returns True only when this call created the row, so the
    caller can alert once rather than on every cycle."""
    async with await psql_db.connection() as connection:
        row = await connection.fetchrow(
            f"""
            INSERT INTO {cst.TOURNAMENT_TASK_FAILURE_REVIEWS_TABLE}
                (task_id, tournament_id, round_id, status, failed_hotkeys, total_trainings, notes)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
            ON CONFLICT (task_id) DO NOTHING
            RETURNING task_id
            """,
            UUID(review.task_id),
            review.tournament_id,
            review.round_id,
            review.status.value,
            json.dumps(review.failed_hotkeys),
            review.total_trainings,
            review.notes,
        )
    created = row is not None
    if created:
        logger.info(
            f"Opened majority-failure review for task {review.task_id} "
            f"({len(review.failed_hotkeys)}/{review.total_trainings} trainings failed)"
        )
    return created


async def get_task_failure_review(task_id: str, psql_db: PSQLDB) -> TournamentTaskFailureReview | None:
    async with await psql_db.connection() as connection:
        row = await connection.fetchrow(
            f"SELECT * FROM {cst.TOURNAMENT_TASK_FAILURE_REVIEWS_TABLE} WHERE task_id = $1", UUID(task_id)
        )
        return _row_to_review(row) if row else None


async def approve_task_failure_review(task_id: str, psql_db: PSQLDB, notes: str | None = None) -> bool:
    """Clear the gate for a task. Returns True if a pending row was approved."""
    async with await psql_db.connection() as connection:
        row = await connection.fetchrow(
            f"""
            UPDATE {cst.TOURNAMENT_TASK_FAILURE_REVIEWS_TABLE}
            SET status = $2, reviewed_at = $3, notes = COALESCE($4, notes)
            WHERE task_id = $1
            RETURNING task_id
            """,
            UUID(task_id),
            TaskFailureReviewStatus.APPROVED.value,
            datetime.now(timezone.utc),
            notes,
        )
    approved = row is not None
    if approved:
        logger.info(f"Approved majority-failure review for task {task_id}")
    return approved


async def get_pending_task_failure_reviews(psql_db: PSQLDB) -> list[TournamentTaskFailureReview]:
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            f"""
            SELECT * FROM {cst.TOURNAMENT_TASK_FAILURE_REVIEWS_TABLE}
            WHERE status = $1 ORDER BY created_at
            """,
            TaskFailureReviewStatus.PENDING_REVIEW.value,
        )
        return [_row_to_review(row) for row in rows]
