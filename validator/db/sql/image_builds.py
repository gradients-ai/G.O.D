"""Persistence for prepared tournament image builds.

Every transition is a compare-and-set on (build_id, attempts, status), so a stale worker or a late builder report
cannot overwrite a newer attempt.
"""

from datetime import datetime
from uuid import UUID

import validator.db.constants as cst
from validator.db.database import PSQLDB
from validator.tournament.image_builds.models import ImageBuild
from validator.tournament.image_builds.models import ImageBuildStatus
from validator.tournament.image_builds.models import ImageBuildVariant


_COLUMNS = f"""
    {cst.IMAGE_BUILD_ID}, {cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.IMAGE_BUILD_VARIANT}, {cst.IMAGE_BUILD_REPO},
    {cst.IMAGE_BUILD_COMMIT_REF}, {cst.IMAGE_BUILD_REPOSITORY}, {cst.STATUS}, {cst.IMAGE_BUILD_ATTEMPTS},
    {cst.IMAGE_BUILD_NEXT_ATTEMPT_AT}, {cst.IMAGE_BUILD_BUILDER}, {cst.IMAGE_BUILD_LEASE_EXPIRES_AT},
    {cst.IMAGE_BUILD_DIGEST}, {cst.IMAGE_BUILD_RESOLVED_COMMIT}, {cst.IMAGE_BUILD_FAILURE_CLASS},
    {cst.IMAGE_BUILD_ERROR_MESSAGE}, {cst.CREATED_AT}, {cst.UPDATED_AT}
"""


def _row(row) -> ImageBuild:
    return ImageBuild(
        build_id=str(row[cst.IMAGE_BUILD_ID]),
        tournament_id=row[cst.TOURNAMENT_ID],
        hotkey=row[cst.HOTKEY],
        variant=ImageBuildVariant(row[cst.IMAGE_BUILD_VARIANT]),
        repo=row[cst.IMAGE_BUILD_REPO],
        commit_ref=row[cst.IMAGE_BUILD_COMMIT_REF],
        repository=row[cst.IMAGE_BUILD_REPOSITORY],
        status=ImageBuildStatus(row[cst.STATUS]),
        attempts=row[cst.IMAGE_BUILD_ATTEMPTS],
        next_attempt_at=row[cst.IMAGE_BUILD_NEXT_ATTEMPT_AT],
        builder=row[cst.IMAGE_BUILD_BUILDER],
        lease_expires_at=row[cst.IMAGE_BUILD_LEASE_EXPIRES_AT],
        digest=row[cst.IMAGE_BUILD_DIGEST],
        resolved_commit=row[cst.IMAGE_BUILD_RESOLVED_COMMIT],
        failure_class=row[cst.IMAGE_BUILD_FAILURE_CLASS],
        error_message=row[cst.IMAGE_BUILD_ERROR_MESSAGE],
        created_at=row[cst.CREATED_AT],
        updated_at=row[cst.UPDATED_AT],
    )


async def ensure_build(
    tournament_id: str,
    hotkey: str,
    variant: ImageBuildVariant,
    repo: str,
    commit_ref: str,
    repository: str,
    psql_db: PSQLDB,
) -> ImageBuild:
    """Return the build row for this (tournament, hotkey, variant, repo, commit), creating it if missing."""
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            INSERT INTO {cst.TOURNAMENT_IMAGE_BUILDS_TABLE}
                ({cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.IMAGE_BUILD_VARIANT}, {cst.IMAGE_BUILD_REPO},
                 {cst.IMAGE_BUILD_COMMIT_REF}, {cst.IMAGE_BUILD_REPOSITORY})
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT ({cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.IMAGE_BUILD_VARIANT}, {cst.IMAGE_BUILD_REPO},
                         {cst.IMAGE_BUILD_COMMIT_REF}) DO NOTHING
            """,
            tournament_id,
            hotkey,
            variant.value,
            repo,
            commit_ref,
            repository,
        )
        row = await connection.fetchrow(
            f"""
            SELECT {_COLUMNS} FROM {cst.TOURNAMENT_IMAGE_BUILDS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.HOTKEY} = $2 AND {cst.IMAGE_BUILD_VARIANT} = $3
              AND {cst.IMAGE_BUILD_REPO} = $4 AND {cst.IMAGE_BUILD_COMMIT_REF} = $5
            """,
            tournament_id,
            hotkey,
            variant.value,
            repo,
            commit_ref,
        )
    return _row(row)


async def list_due_queued(limit: int, psql_db: PSQLDB) -> list[ImageBuild]:
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            f"""
            SELECT {_COLUMNS} FROM {cst.TOURNAMENT_IMAGE_BUILDS_TABLE}
            WHERE {cst.STATUS} = 'queued' AND {cst.IMAGE_BUILD_NEXT_ATTEMPT_AT} <= NOW()
            ORDER BY {cst.CREATED_AT}, {cst.IMAGE_BUILD_ID}
            LIMIT $1
            """,
            limit,
        )
    return [_row(row) for row in rows]


async def list_building(psql_db: PSQLDB) -> list[ImageBuild]:
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            f"SELECT {_COLUMNS} FROM {cst.TOURNAMENT_IMAGE_BUILDS_TABLE} WHERE {cst.STATUS} = 'building'"
        )
    return [_row(row) for row in rows]


async def queue_summary(psql_db: PSQLDB) -> tuple[int, int, datetime | None]:
    """(queued due now, building, oldest due-queued created_at)."""
    async with await psql_db.connection() as connection:
        row = await connection.fetchrow(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE {cst.STATUS} = 'queued' AND {cst.IMAGE_BUILD_NEXT_ATTEMPT_AT} <= NOW()) AS queued,
                COUNT(*) FILTER (WHERE {cst.STATUS} = 'building') AS building,
                MIN({cst.CREATED_AT}) FILTER (
                    WHERE {cst.STATUS} = 'queued' AND {cst.IMAGE_BUILD_NEXT_ATTEMPT_AT} <= NOW()
                ) AS oldest
            FROM {cst.TOURNAMENT_IMAGE_BUILDS_TABLE}
            WHERE {cst.STATUS} IN ('queued', 'building')
            """
        )
    return row["queued"], row["building"], row["oldest"]


async def start_attempt(build_id: str, expected_attempts: int, builder: str, lease_seconds: int, psql_db: PSQLDB) -> bool:
    """queued -> building with attempts + 1. False when another worker got there first."""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            updated = await connection.fetchval(
                f"""
                UPDATE {cst.TOURNAMENT_IMAGE_BUILDS_TABLE}
                SET {cst.STATUS} = 'building', {cst.IMAGE_BUILD_ATTEMPTS} = {cst.IMAGE_BUILD_ATTEMPTS} + 1,
                    {cst.IMAGE_BUILD_BUILDER} = $3,
                    {cst.IMAGE_BUILD_LEASE_EXPIRES_AT} = NOW() + ($4 * INTERVAL '1 second'),
                    {cst.UPDATED_AT} = NOW()
                WHERE {cst.IMAGE_BUILD_ID} = $1 AND {cst.STATUS} = 'queued' AND {cst.IMAGE_BUILD_ATTEMPTS} = $2
                RETURNING {cst.IMAGE_BUILD_ATTEMPTS}
                """,
                UUID(build_id),
                expected_attempts,
                builder,
                lease_seconds,
            )
            if updated is None:
                return False
            await connection.execute(
                f"""
                INSERT INTO {cst.TOURNAMENT_IMAGE_BUILD_ATTEMPTS_TABLE}
                    ({cst.IMAGE_BUILD_ID}, {cst.IMAGE_BUILD_ATTEMPT}, {cst.IMAGE_BUILD_BUILDER})
                VALUES ($1, $2, $3)
                ON CONFLICT ({cst.IMAGE_BUILD_ID}, {cst.IMAGE_BUILD_ATTEMPT}) DO UPDATE
                SET {cst.IMAGE_BUILD_BUILDER} = EXCLUDED.{cst.IMAGE_BUILD_BUILDER},
                    {cst.IMAGE_BUILD_STARTED_AT} = NOW(), {cst.IMAGE_BUILD_FINISHED_AT} = NULL,
                    {cst.IMAGE_BUILD_OUTCOME} = NULL, {cst.IMAGE_BUILD_FAILURE_CLASS} = NULL,
                    {cst.IMAGE_BUILD_ERROR_MESSAGE} = NULL
                """,
                UUID(build_id),
                updated,
                builder,
            )
            return True


async def undo_unsent_attempt(build_id: str, attempt: int, retry_at: datetime, psql_db: PSQLDB) -> None:
    """building -> queued without consuming the attempt: the builder never accepted the job."""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            updated = await connection.fetchval(
                f"""
                UPDATE {cst.TOURNAMENT_IMAGE_BUILDS_TABLE}
                SET {cst.STATUS} = 'queued', {cst.IMAGE_BUILD_ATTEMPTS} = {cst.IMAGE_BUILD_ATTEMPTS} - 1,
                    {cst.IMAGE_BUILD_BUILDER} = NULL, {cst.IMAGE_BUILD_LEASE_EXPIRES_AT} = NULL,
                    {cst.IMAGE_BUILD_NEXT_ATTEMPT_AT} = $3, {cst.UPDATED_AT} = NOW()
                WHERE {cst.IMAGE_BUILD_ID} = $1 AND {cst.STATUS} = 'building' AND {cst.IMAGE_BUILD_ATTEMPTS} = $2
                RETURNING 1
                """,
                UUID(build_id),
                attempt,
                retry_at,
            )
            if updated:
                await connection.execute(
                    f"""
                    DELETE FROM {cst.TOURNAMENT_IMAGE_BUILD_ATTEMPTS_TABLE}
                    WHERE {cst.IMAGE_BUILD_ID} = $1 AND {cst.IMAGE_BUILD_ATTEMPT} = $2
                    """,
                    UUID(build_id),
                    attempt,
                )


async def renew_lease(build_id: str, attempt: int, lease_seconds: int, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            UPDATE {cst.TOURNAMENT_IMAGE_BUILDS_TABLE}
            SET {cst.IMAGE_BUILD_LEASE_EXPIRES_AT} = NOW() + ($3 * INTERVAL '1 second'), {cst.UPDATED_AT} = NOW()
            WHERE {cst.IMAGE_BUILD_ID} = $1 AND {cst.STATUS} = 'building' AND {cst.IMAGE_BUILD_ATTEMPTS} = $2
            """,
            UUID(build_id),
            attempt,
            lease_seconds,
        )


async def mark_ready(
    build_id: str,
    attempt: int,
    digest: str,
    resolved_commit: str | None,
    dockerfile_path: str | None,
    size_bytes: int | None,
    psql_db: PSQLDB,
) -> bool:
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            updated = await connection.fetchval(
                f"""
                UPDATE {cst.TOURNAMENT_IMAGE_BUILDS_TABLE}
                SET {cst.STATUS} = 'ready', {cst.IMAGE_BUILD_DIGEST} = $3, {cst.IMAGE_BUILD_RESOLVED_COMMIT} = $4,
                    {cst.IMAGE_BUILD_DOCKERFILE_PATH} = $5, {cst.IMAGE_BUILD_SIZE_BYTES} = $6,
                    {cst.IMAGE_BUILD_LEASE_EXPIRES_AT} = NULL, {cst.IMAGE_BUILD_FAILURE_CLASS} = NULL,
                    {cst.IMAGE_BUILD_ERROR_MESSAGE} = NULL, {cst.IMAGE_BUILD_READY_AT} = NOW(), {cst.UPDATED_AT} = NOW()
                WHERE {cst.IMAGE_BUILD_ID} = $1 AND {cst.STATUS} = 'building' AND {cst.IMAGE_BUILD_ATTEMPTS} = $2
                RETURNING 1
                """,
                UUID(build_id),
                attempt,
                digest,
                resolved_commit,
                dockerfile_path,
                size_bytes,
            )
            if updated:
                await _finish_attempt(connection, build_id, attempt, "ready", None, None)
            return bool(updated)


async def record_failure(
    build_id: str,
    attempt: int,
    failure_class: str,
    error: str,
    retry_at: datetime | None,
    psql_db: PSQLDB,
) -> bool:
    """building -> queued (retry_at set) or failed (retry_at None)."""
    error = (error or "")[-4000:]
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            updated = await connection.fetchval(
                f"""
                UPDATE {cst.TOURNAMENT_IMAGE_BUILDS_TABLE}
                SET {cst.STATUS} = CASE WHEN $5::timestamptz IS NULL THEN 'failed' ELSE 'queued' END,
                    {cst.IMAGE_BUILD_NEXT_ATTEMPT_AT} = COALESCE($5::timestamptz, {cst.IMAGE_BUILD_NEXT_ATTEMPT_AT}),
                    {cst.IMAGE_BUILD_FAILURE_CLASS} = $3, {cst.IMAGE_BUILD_ERROR_MESSAGE} = $4,
                    {cst.IMAGE_BUILD_LEASE_EXPIRES_AT} = NULL,
                    {cst.IMAGE_BUILD_FAILED_AT} = CASE WHEN $5::timestamptz IS NULL THEN NOW() ELSE NULL END,
                    {cst.UPDATED_AT} = NOW()
                WHERE {cst.IMAGE_BUILD_ID} = $1 AND {cst.STATUS} = 'building' AND {cst.IMAGE_BUILD_ATTEMPTS} = $2
                RETURNING 1
                """,
                UUID(build_id),
                attempt,
                failure_class,
                error,
                retry_at,
            )
            if updated:
                await _finish_attempt(connection, build_id, attempt, "failed", failure_class, error)
            return bool(updated)


async def _finish_attempt(connection, build_id: str, attempt: int, outcome: str, failure_class, error) -> None:
    await connection.execute(
        f"""
        UPDATE {cst.TOURNAMENT_IMAGE_BUILD_ATTEMPTS_TABLE}
        SET {cst.IMAGE_BUILD_FINISHED_AT} = NOW(), {cst.IMAGE_BUILD_OUTCOME} = $3,
            {cst.IMAGE_BUILD_FAILURE_CLASS} = $4, {cst.IMAGE_BUILD_ERROR_MESSAGE} = $5
        WHERE {cst.IMAGE_BUILD_ID} = $1 AND {cst.IMAGE_BUILD_ATTEMPT} = $2
        """,
        UUID(build_id),
        attempt,
        outcome,
        failure_class,
        error,
    )


async def tournaments_to_prune(psql_db: PSQLDB) -> list[str]:
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            f"""
            SELECT DISTINCT b.{cst.TOURNAMENT_ID}
            FROM {cst.TOURNAMENT_IMAGE_BUILDS_TABLE} b
            JOIN {cst.TOURNAMENTS_TABLE} t ON t.{cst.TOURNAMENT_ID} = b.{cst.TOURNAMENT_ID}
            WHERE b.{cst.IMAGE_BUILD_PRUNED_AT} IS NULL
              AND t.{cst.TOURNAMENT_STATUS} IN ('completed', 'cancelled')
              AND NOT EXISTS (
                  SELECT 1 FROM {cst.TOURNAMENT_IMAGE_BUILDS_TABLE} active
                  WHERE active.{cst.TOURNAMENT_ID} = b.{cst.TOURNAMENT_ID} AND active.{cst.STATUS} = 'building'
              )
            """
        )
    return [row[cst.TOURNAMENT_ID] for row in rows]


async def mark_tournament_pruned(tournament_id: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            UPDATE {cst.TOURNAMENT_IMAGE_BUILDS_TABLE}
            SET {cst.IMAGE_BUILD_PRUNED_AT} = NOW(), {cst.UPDATED_AT} = NOW(),
                {cst.STATUS} = CASE WHEN {cst.STATUS} = 'queued' THEN 'failed' ELSE {cst.STATUS} END
            WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.IMAGE_BUILD_PRUNED_AT} IS NULL
            """,
            tournament_id,
        )
