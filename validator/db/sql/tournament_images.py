"""Rows of tournament_images: the trainer image built once per tournament participant and Dockerfile variant."""

import validator.db.constants as cst
from validator.db.database import PSQLDB
from validator.tournament.models import TournamentImage
from validator.tournament.models import TournamentImageStatus
from validator.tournament.models import TrainerImageVariant


_COLUMNS = (
    f"{cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.IMAGE_VARIANT}, {cst.IMAGE_STATUS}, {cst.IMAGE_BUILD_NUMBER}, "
    f"{cst.IMAGE_BUILD_ATTEMPTS}, {cst.IMAGE_REF}, {cst.IMAGE_ERROR}"
)


def _row_to_image(row) -> TournamentImage:
    return TournamentImage(
        tournament_id=row[cst.TOURNAMENT_ID],
        hotkey=row[cst.HOTKEY],
        variant=TrainerImageVariant(row[cst.IMAGE_VARIANT]),
        status=TournamentImageStatus(row[cst.IMAGE_STATUS]),
        build_number=row[cst.IMAGE_BUILD_NUMBER],
        build_attempts=row[cst.IMAGE_BUILD_ATTEMPTS],
        image_ref=row[cst.IMAGE_REF],
        error=row[cst.IMAGE_ERROR],
    )


async def add_pending_images(tournament_id: str, wanted: list[tuple[str, TrainerImageVariant]], psql_db: PSQLDB) -> None:
    """Add a pending row for each (hotkey, variant) that has none yet; existing rows are left as they are."""
    if not wanted:
        return
    async with await psql_db.connection() as connection:
        await connection.executemany(
            f"""
            INSERT INTO {cst.TOURNAMENT_IMAGES_TABLE} ({cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.IMAGE_VARIANT}, {cst.IMAGE_STATUS})
            VALUES ($1, $2, $3, $4)
            ON CONFLICT ({cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.IMAGE_VARIANT}) DO NOTHING
            """,
            [(tournament_id, hotkey, variant.value, TournamentImageStatus.PENDING.value) for hotkey, variant in wanted],
        )


async def get_images_with_status(status: TournamentImageStatus, psql_db: PSQLDB) -> list[TournamentImage]:
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            f"SELECT {_COLUMNS} FROM {cst.TOURNAMENT_IMAGES_TABLE} WHERE {cst.IMAGE_STATUS} = $1 ORDER BY {cst.CREATED_AT}",
            status.value,
        )
    return [_row_to_image(row) for row in rows]


async def get_tournament_image(
    tournament_id: str, hotkey: str, variant: TrainerImageVariant, psql_db: PSQLDB
) -> TournamentImage | None:
    async with await psql_db.connection() as connection:
        row = await connection.fetchrow(
            f"""
            SELECT {_COLUMNS} FROM {cst.TOURNAMENT_IMAGES_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.HOTKEY} = $2 AND {cst.IMAGE_VARIANT} = $3
            """,
            tournament_id,
            hotkey,
            variant.value,
        )
    return _row_to_image(row) if row else None


async def _update(image: TournamentImage, assignments: str, values: list, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            UPDATE {cst.TOURNAMENT_IMAGES_TABLE} SET {assignments}, {cst.UPDATED_AT} = NOW()
            WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.HOTKEY} = $2 AND {cst.IMAGE_VARIANT} = $3
            """,
            image.tournament_id,
            image.hotkey,
            image.variant.value,
            *values,
        )


async def mark_image_building(image: TournamentImage, build_number: int, psql_db: PSQLDB) -> None:
    await _update(
        image,
        f"{cst.IMAGE_STATUS} = $4, {cst.IMAGE_BUILD_NUMBER} = $5, {cst.IMAGE_ERROR} = NULL",
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
    await _update(image, f"{cst.IMAGE_STATUS} = $4, {cst.IMAGE_ERROR} = $5", [TournamentImageStatus.FAILED.value, error], psql_db)


async def record_start_failure(image: TournamentImage, error: str, max_attempts: int, psql_db: PSQLDB) -> None:
    """Count a failed attempt to snapshot and start the build; the row fails once it reaches max_attempts."""
    await _update(
        image,
        f"""{cst.IMAGE_BUILD_ATTEMPTS} = {cst.IMAGE_BUILD_ATTEMPTS} + 1, {cst.IMAGE_ERROR} = $4,
            {cst.IMAGE_STATUS} = CASE WHEN {cst.IMAGE_BUILD_ATTEMPTS} + 1 >= $5 THEN $6 ELSE {cst.IMAGE_STATUS} END""",
        [error, max_attempts, TournamentImageStatus.FAILED.value],
        psql_db,
    )
