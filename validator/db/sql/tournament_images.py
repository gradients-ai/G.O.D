"""Rows of tournament_images: the trainer image built once per (scope, source, variant) before training."""

import validator.db.constants as cst
from validator.db.database import PSQLDB
from validator.tournament.models import TournamentImage
from validator.tournament.models import TournamentImageStatus
from validator.tournament.models import TrainerImageVariant


_COLUMNS = (
    f"{cst.IMAGE_SCOPE}, {cst.IMAGE_SOURCE}, {cst.IMAGE_VARIANT}, {cst.IMAGE_REPO}, {cst.IMAGE_COMMIT_HASH}, "
    f"{cst.IMAGE_GITHUB_TOKEN}, {cst.IMAGE_HOTKEY}, {cst.IMAGE_STATUS}, {cst.IMAGE_BUILD_NUMBER}, "
    f"{cst.IMAGE_BUILD_ATTEMPTS}, {cst.IMAGE_REF}, {cst.IMAGE_ERROR}"
)
_KEY = f"{cst.IMAGE_SCOPE} = $1 AND {cst.IMAGE_SOURCE} = $2 AND {cst.IMAGE_VARIANT} = $3"
_UNFINISHED = (TournamentImageStatus.PENDING.value, TournamentImageStatus.BUILDING.value)


def _row_to_image(row) -> TournamentImage:
    return TournamentImage(
        scope=row[cst.IMAGE_SCOPE],
        source=row[cst.IMAGE_SOURCE],
        variant=TrainerImageVariant(row[cst.IMAGE_VARIANT]),
        repo=row[cst.IMAGE_REPO],
        commit_hash=row[cst.IMAGE_COMMIT_HASH],
        github_token=row[cst.IMAGE_GITHUB_TOKEN],
        hotkey=row[cst.IMAGE_HOTKEY],
        status=TournamentImageStatus(row[cst.IMAGE_STATUS]),
        build_number=row[cst.IMAGE_BUILD_NUMBER],
        build_attempts=row[cst.IMAGE_BUILD_ATTEMPTS],
        image_ref=row[cst.IMAGE_REF],
        error=row[cst.IMAGE_ERROR],
    )


async def add_pending_images(images: list[TournamentImage], psql_db: PSQLDB) -> None:
    """Add each image as pending unless a row with its key exists; existing rows are left as they are."""
    if not images:
        return
    async with await psql_db.connection() as connection:
        await connection.executemany(
            f"""
            INSERT INTO {cst.TOURNAMENT_IMAGES_TABLE} ({cst.IMAGE_SCOPE}, {cst.IMAGE_SOURCE}, {cst.IMAGE_VARIANT},
                {cst.IMAGE_REPO}, {cst.IMAGE_COMMIT_HASH}, {cst.IMAGE_GITHUB_TOKEN}, {cst.IMAGE_HOTKEY}, {cst.IMAGE_STATUS})
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT ({cst.IMAGE_SCOPE}, {cst.IMAGE_SOURCE}, {cst.IMAGE_VARIANT}) DO NOTHING
            """,
            [
                (i.scope, i.source, i.variant.value, i.repo, i.commit_hash, i.github_token, i.hotkey, i.status.value)
                for i in images
            ],
        )


async def get_images_with_status(status: TournamentImageStatus, psql_db: PSQLDB) -> list[TournamentImage]:
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            f"SELECT {_COLUMNS} FROM {cst.TOURNAMENT_IMAGES_TABLE} WHERE {cst.IMAGE_STATUS} = $1 ORDER BY {cst.CREATED_AT}",
            status.value,
        )
    return [_row_to_image(row) for row in rows]


async def get_image(scope: str, source: str, variant: TrainerImageVariant, psql_db: PSQLDB) -> TournamentImage | None:
    async with await psql_db.connection() as connection:
        row = await connection.fetchrow(
            f"SELECT {_COLUMNS} FROM {cst.TOURNAMENT_IMAGES_TABLE} WHERE {_KEY}", scope, source, variant.value
        )
    return _row_to_image(row) if row else None


async def count_unfinished(scope: str, source: str, psql_db: PSQLDB) -> int:
    """Images of this snapshot (any variant) still pending or building."""
    async with await psql_db.connection() as connection:
        return await connection.fetchval(
            f"""
            SELECT COUNT(*) FROM {cst.TOURNAMENT_IMAGES_TABLE}
            WHERE {cst.IMAGE_SCOPE} = $1 AND {cst.IMAGE_SOURCE} = $2 AND {cst.IMAGE_STATUS} = ANY($3)
            """,
            scope,
            source,
            list(_UNFINISHED),
        )


async def _update(image: TournamentImage, assignments: str, values: list, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"UPDATE {cst.TOURNAMENT_IMAGES_TABLE} SET {assignments}, {cst.UPDATED_AT} = NOW() WHERE {_KEY}",
            image.scope,
            image.source,
            image.variant.value,
            *values,
        )


async def mark_image_building(image: TournamentImage, build_number: int, psql_db: PSQLDB) -> None:
    """The snapshot is taken and the build started: the miner's token is no longer needed."""
    await _update(
        image,
        f"{cst.IMAGE_STATUS} = $4, {cst.IMAGE_BUILD_NUMBER} = $5, {cst.IMAGE_ERROR} = NULL, {cst.IMAGE_GITHUB_TOKEN} = NULL",
        [TournamentImageStatus.BUILDING.value, build_number],
        psql_db,
    )


async def mark_image_ready(image: TournamentImage, image_ref: str, psql_db: PSQLDB) -> None:
    await _update(
        image,
        f"{cst.IMAGE_STATUS} = $4, {cst.IMAGE_REF} = $5, {cst.IMAGE_ERROR} = NULL",
        [TournamentImageStatus.READY.value, image_ref],
        psql_db,
    )


async def mark_image_failed(image: TournamentImage, error: str, psql_db: PSQLDB) -> None:
    await _update(
        image,
        f"{cst.IMAGE_STATUS} = $4, {cst.IMAGE_ERROR} = $5, {cst.IMAGE_GITHUB_TOKEN} = NULL",
        [TournamentImageStatus.FAILED.value, error],
        psql_db,
    )


async def record_start_failure(image: TournamentImage, error: str, max_attempts: int, psql_db: PSQLDB) -> None:
    """Count a failed attempt to snapshot and start the build; the row fails once it reaches max_attempts."""
    await _update(
        image,
        f"""{cst.IMAGE_BUILD_ATTEMPTS} = {cst.IMAGE_BUILD_ATTEMPTS} + 1, {cst.IMAGE_ERROR} = $4,
            {cst.IMAGE_STATUS} = CASE WHEN {cst.IMAGE_BUILD_ATTEMPTS} + 1 >= $5 THEN $6 ELSE {cst.IMAGE_STATUS} END,
            {cst.IMAGE_GITHUB_TOKEN} = CASE WHEN {cst.IMAGE_BUILD_ATTEMPTS} + 1 >= $5
                THEN NULL ELSE {cst.IMAGE_GITHUB_TOKEN} END""",
        [error, max_attempts, TournamentImageStatus.FAILED.value],
        psql_db,
    )
