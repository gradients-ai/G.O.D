"""Durable journal and atomic work claiming for tournament image builds."""

from datetime import datetime
from datetime import timezone
from typing import Any
from uuid import UUID

import validator.db.constants as cst
from validator.db.database import PSQLDB
from validator.tournament.models import ImageBuild
from validator.tournament.models import ImageBuildStatus
from validator.tournament.models import ImageBuildVariant


def _row_to_build(row: Any) -> ImageBuild:
    return ImageBuild(
        build_id=str(row["build_id"]),
        batch_id=str(row["batch_id"]),
        tournament_id=row["tournament_id"],
        hotkey=row["hotkey"],
        variant=ImageBuildVariant(row["variant"]),
        stable_build_id=row["stable_build_id"],
        repo=row["repo"],
        commit_sha=row["commit_sha"],
        dockerfile_path=row["dockerfile_path"],
        build_context=row["build_context"],
        platform=row["platform"],
        depot_project_id=row["depot_project_id"],
        status=ImageBuildStatus(row["status"]),
        current_attempt=row["current_attempt"],
        github_token=row.get("github_token"),
        pull_ref=row["pull_ref"],
        digest=row["digest"],
        failure_class=row["failure_class"],
        log_ref=row["log_ref"],
        error_message=row["error_message"],
        retry_after_at=row["retry_after_at"],
    )


async def create_batch_with_builds(
    tournament_id: str,
    manifests: list[dict[str, str]],
    psql_db: PSQLDB,
) -> str:
    """Create the tournament's only build batch and missing rows idempotently."""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            batch_id = await connection.fetchval(
                f"""
                INSERT INTO {cst.IMAGE_BUILD_BATCHES_TABLE} (tournament_id)
                VALUES ($1)
                ON CONFLICT (tournament_id) DO UPDATE SET updated_at = NOW()
                RETURNING batch_id
                """,
                tournament_id,
            )
            for manifest in manifests:
                await connection.execute(
                    f"""
                    INSERT INTO {cst.IMAGE_BUILDS_TABLE}
                        (batch_id, tournament_id, hotkey, variant, stable_build_id,
                         repo, commit_sha, dockerfile_path, build_context, platform)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (tournament_id, hotkey, variant) DO NOTHING
                    """,
                    batch_id,
                    tournament_id,
                    manifest["hotkey"],
                    manifest["variant"],
                    manifest["stable_build_id"],
                    manifest["repo"],
                    manifest["commit_sha"],
                    manifest["dockerfile_path"],
                    manifest["build_context"],
                    manifest.get("platform", "linux/amd64"),
                )
    return str(batch_id)


async def claim_builds(worker_id: str, limit: int, psql_db: PSQLDB) -> list[ImageBuild]:
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            rows = await connection.fetch(
                f"""
                WITH claimed AS (
                    SELECT build_id
                    FROM {cst.IMAGE_BUILDS_TABLE}
                    WHERE status IN ('queued', 'retry_wait')
                      AND (retry_after_at IS NULL OR retry_after_at <= NOW())
                    ORDER BY created_at, build_id
                    FOR UPDATE SKIP LOCKED
                    LIMIT $1
                )
                UPDATE {cst.IMAGE_BUILDS_TABLE} b
                SET status = 'building', worker_id = $2, dequeued_at = NOW(),
                    cli_started_at = NOW(), current_attempt = current_attempt + 1,
                    updated_at = NOW(), error_message = NULL
                FROM claimed
                WHERE b.build_id = claimed.build_id
                RETURNING b.*,
                    (SELECT github_token FROM {cst.TOURNAMENT_PARTICIPANTS_TABLE} p
                     WHERE p.tournament_id = b.tournament_id AND p.hotkey = b.hotkey) AS github_token
                """,
                limit,
                worker_id,
            )
    return [_row_to_build(row) for row in rows]


async def set_project_id(build_id: str, project_id: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"UPDATE {cst.IMAGE_BUILDS_TABLE} SET depot_project_id=$2, updated_at=NOW() WHERE build_id=$1",
            UUID(build_id),
            project_id,
        )


async def get_submission_project_id(tournament_id: str, hotkey: str, psql_db: PSQLDB) -> str | None:
    async with await psql_db.connection() as connection:
        return await connection.fetchval(
            f"""
            SELECT depot_project_id FROM {cst.IMAGE_BUILDS_TABLE}
            WHERE tournament_id=$1 AND hotkey=$2 AND depot_project_id IS NOT NULL
            LIMIT 1
            """,
            tournament_id,
            hotkey,
        )


async def set_submission_project_id(tournament_id: str, hotkey: str, project_id: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            UPDATE {cst.IMAGE_BUILDS_TABLE}
            SET depot_project_id=$3, updated_at=NOW()
            WHERE tournament_id=$1 AND hotkey=$2 AND depot_project_id IS NULL
            """,
            tournament_id,
            hotkey,
            project_id,
        )


async def create_attempt(build_id: str, attempt: int, depot_build_id: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            INSERT INTO {cst.IMAGE_BUILD_ATTEMPTS_TABLE}
                (build_id, attempt_number, depot_build_id, status)
            VALUES ($1, $2, $3, 'building')
            ON CONFLICT (build_id, attempt_number) DO UPDATE
            SET depot_build_id=EXCLUDED.depot_build_id, status='building', started_at=NOW()
            """,
            UUID(build_id),
            attempt,
            depot_build_id,
        )


async def mark_publishing(build_id: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            UPDATE {cst.IMAGE_BUILDS_TABLE}
            SET status='publishing', build_finished_at=NOW(), updated_at=NOW()
            WHERE build_id=$1 AND status='building'
            """,
            UUID(build_id),
        )


async def mark_ready(
    build_id: str,
    attempt: int,
    pull_ref: str,
    digest: str,
    log_ref: str | None,
    psql_db: PSQLDB,
) -> None:
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            await connection.execute(
                f"""
                UPDATE {cst.IMAGE_BUILDS_TABLE}
                SET status='ready', pull_ref=$2, digest=$3, log_ref=$4,
                    publication_confirmed_at=NOW(), ready_at=NOW(), updated_at=NOW()
                WHERE build_id=$1
                """,
                UUID(build_id),
                pull_ref,
                digest,
                log_ref,
            )
            await connection.execute(
                f"""
                UPDATE {cst.IMAGE_BUILD_ATTEMPTS_TABLE}
                SET status='ready', finished_at=NOW(),
                    duration_seconds=EXTRACT(EPOCH FROM (NOW()-started_at)), log_ref=$3
                WHERE build_id=$1 AND attempt_number=$2
                """,
                UUID(build_id),
                attempt,
                log_ref,
            )


async def fail_or_retry(
    build_id: str,
    attempt: int,
    error: str,
    failure_class: str,
    retry_at: datetime | None,
    psql_db: PSQLDB,
) -> None:
    status = ImageBuildStatus.RETRY_WAIT if retry_at else ImageBuildStatus.FAILED
    now = datetime.now(timezone.utc)
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            await connection.execute(
                f"""
                UPDATE {cst.IMAGE_BUILDS_TABLE}
                SET status=$2, failure_class=$3, error_message=$4, retry_after_at=$5,
                    failed_at=CASE WHEN $2='failed' THEN $6 ELSE NULL END, updated_at=NOW()
                WHERE build_id=$1
                """,
                UUID(build_id),
                status.value,
                failure_class,
                error[-4000:],
                retry_at,
                now,
            )
            await connection.execute(
                f"""
                UPDATE {cst.IMAGE_BUILD_ATTEMPTS_TABLE}
                SET status=$3, finished_at=NOW(),
                    duration_seconds=EXTRACT(EPOCH FROM (NOW()-started_at)),
                    error_message=$4, failure_class=$5
                WHERE build_id=$1 AND attempt_number=$2
                """,
                UUID(build_id),
                attempt,
                status.value,
                error[-4000:],
                failure_class,
            )


async def get_build(
    tournament_id: str,
    hotkey: str,
    variant: ImageBuildVariant,
    psql_db: PSQLDB,
) -> ImageBuild | None:
    async with await psql_db.connection() as connection:
        row = await connection.fetchrow(
            f"""
            SELECT b.*, p.github_token
            FROM {cst.IMAGE_BUILDS_TABLE} b
            JOIN {cst.TOURNAMENT_PARTICIPANTS_TABLE} p USING (tournament_id, hotkey)
            WHERE b.tournament_id=$1 AND b.hotkey=$2 AND b.variant=$3
            """,
            tournament_id,
            hotkey,
            variant.value,
        )
    return _row_to_build(row) if row else None


async def list_ready_builds(tournament_id: str, psql_db: PSQLDB) -> list[ImageBuild]:
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            f"SELECT * FROM {cst.IMAGE_BUILDS_TABLE} WHERE tournament_id=$1 AND status='ready'",
            tournament_id,
        )
    return [_row_to_build(row) for row in rows]


async def list_projects_due_for_cleanup(retention_days: int, psql_db: PSQLDB) -> list[tuple[str, str]]:
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            f"""
            SELECT DISTINCT b.batch_id, i.depot_project_id
            FROM {cst.IMAGE_BUILD_BATCHES_TABLE} b
            JOIN {cst.TOURNAMENTS_TABLE} t USING (tournament_id)
            JOIN {cst.IMAGE_BUILDS_TABLE} i USING (batch_id)
            WHERE b.cleaned_at IS NULL
              AND t.status IN ('completed', 'cancelled')
              AND t.updated_at < NOW() - ($1 * INTERVAL '1 day')
              AND i.depot_project_id IS NOT NULL
            """,
            retention_days,
        )
    return [(str(row["batch_id"]), row["depot_project_id"]) for row in rows]


async def mark_batch_cleaned(batch_id: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"UPDATE {cst.IMAGE_BUILD_BATCHES_TABLE} SET cleaned_at=NOW(), updated_at=NOW() WHERE batch_id=$1",
            UUID(batch_id),
        )


async def recover_stale_builds(
    timeout_seconds: int,
    max_attempts: int,
    psql_db: PSQLDB,
) -> list[tuple[str, str]]:
    """Move abandoned in-flight rows to retry/failed and return Depot builds to cancel."""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            rows = await connection.fetch(
                f"""
                SELECT b.build_id, b.current_attempt, b.depot_project_id, a.depot_build_id
                FROM {cst.IMAGE_BUILDS_TABLE} b
                LEFT JOIN {cst.IMAGE_BUILD_ATTEMPTS_TABLE} a
                  ON a.build_id=b.build_id AND a.attempt_number=b.current_attempt
                WHERE b.status IN ('building', 'publishing')
                  AND b.cli_started_at < NOW() - ($1 * INTERVAL '1 second')
                FOR UPDATE OF b
                """,
                timeout_seconds,
            )
            for row in rows:
                retry = row["current_attempt"] < max_attempts
                await connection.execute(
                    f"""
                    UPDATE {cst.IMAGE_BUILDS_TABLE}
                    SET status=$2, retry_after_at=CASE WHEN $2='retry_wait' THEN NOW() ELSE NULL END,
                        failure_class='worker_restart', error_message='Recovered stale build after worker restart',
                        failed_at=CASE WHEN $2='failed' THEN NOW() ELSE NULL END, updated_at=NOW()
                    WHERE build_id=$1
                    """,
                    row["build_id"],
                    "retry_wait" if retry else "failed",
                )
            return [
                (row["depot_project_id"], row["depot_build_id"])
                for row in rows
                if row["depot_project_id"] and row["depot_build_id"]
            ]
