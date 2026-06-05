"""DB access for the tournament submission de-duplication review gate."""

import json
from datetime import datetime
from datetime import timezone
from typing import Any

import validator.db.constants as cst
from core.models.tournament_models import DedupClusterRecord
from core.models.tournament_models import DedupPairVerdict
from core.models.tournament_models import DedupReviewStatus
from core.models.tournament_models import PublishedRepo
from core.models.tournament_models import TournamentDedupReview
from validator.db.database import PSQLDB
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def _loads(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _row_to_review(row: Any) -> TournamentDedupReview:
    return TournamentDedupReview(
        round_id=row["round_id"],
        tournament_id=row["tournament_id"],
        tournament_type=row["tournament_type"],
        status=DedupReviewStatus(row["status"]),
        cohort=_loads(row["cohort"]) or [],
        clusters=[DedupClusterRecord(**c) for c in (_loads(row["clusters"]) or [])],
        pair_verdicts=[DedupPairVerdict(**v) for v in (_loads(row["pair_verdicts"]) or [])],
        flagged_hotkeys=_loads(row["flagged_hotkeys"]) or [],
        approved_eliminations=_loads(row["approved_eliminations"]) or [],
        published_repos=[PublishedRepo(**p) for p in (_loads(row["published_repos"]) or [])],
        report_url=row["report_url"],
        notes=row["notes"],
        created_at=row["created_at"],
        reviewed_at=row["reviewed_at"],
        resolved_at=row["resolved_at"],
    )


async def insert_dedup_review(review: TournamentDedupReview, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            INSERT INTO {cst.TOURNAMENT_DEDUP_REVIEWS_TABLE}
                (round_id, tournament_id, tournament_type, status, cohort, clusters,
                 pair_verdicts, flagged_hotkeys, approved_eliminations, report_url)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb, $9::jsonb, $10)
            ON CONFLICT (round_id) DO NOTHING
            """,
            review.round_id,
            review.tournament_id,
            review.tournament_type,
            review.status.value,
            json.dumps(review.cohort),
            json.dumps([c.model_dump() for c in review.clusters]),
            json.dumps([v.model_dump() for v in review.pair_verdicts]),
            json.dumps(review.flagged_hotkeys),
            json.dumps(review.approved_eliminations or review.flagged_hotkeys),
            review.report_url,
        )
        logger.info(f"Inserted dedup review for round {review.round_id} ({len(review.flagged_hotkeys)} flagged)")


async def get_dedup_review(round_id: str, psql_db: PSQLDB) -> TournamentDedupReview | None:
    async with await psql_db.connection() as connection:
        row = await connection.fetchrow(
            f"SELECT * FROM {cst.TOURNAMENT_DEDUP_REVIEWS_TABLE} WHERE round_id = $1", round_id
        )
        return _row_to_review(row) if row else None


async def mark_dedup_review_resolved(
    round_id: str, published_repos: list[PublishedRepo], report_url: str | None, psql_db: PSQLDB
) -> None:
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            UPDATE {cst.TOURNAMENT_DEDUP_REVIEWS_TABLE}
            SET published_repos = $2::jsonb, report_url = COALESCE($3, report_url), resolved_at = $4
            WHERE round_id = $1
            """,
            round_id,
            json.dumps([p.model_dump() for p in published_repos]),
            report_url,
            datetime.now(timezone.utc),
        )
        logger.info(f"Resolved dedup review for round {round_id} ({len(published_repos)} repos published)")


async def get_resolved_dedup_reviews(psql_db: PSQLDB, limit: int = 100, page: int = 1) -> list[TournamentDedupReview]:
    """Reviews whose eliminations have been applied — safe to expose publicly via auditing."""
    offset = (page - 1) * limit
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            f"""
            SELECT * FROM {cst.TOURNAMENT_DEDUP_REVIEWS_TABLE}
            WHERE resolved_at IS NOT NULL
            ORDER BY resolved_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
        return [_row_to_review(r) for r in rows]
