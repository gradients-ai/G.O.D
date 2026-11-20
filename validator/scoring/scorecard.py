"""The public scorecard of a freeform quant result: everything needed to verify it without the miner's decoder.

build_quant_scorecard is pure (rows and records in, model out) so the handover release can reuse it; the async
quant_scorecards_for_task does the I/O. The scorecard never carries the decoder object name, a presigned decoder
URL, the GitHub token or the private training repo: the participant fields it reads are copied one by one.
"""

import math
from uuid import UUID

import validator.infrastructure.service_constants as service_cst
from core.logging import get_logger
from core.models.quant_artifact_models import QuantDecodedFile
from core.models.task_models import TaskType
from validator.app.config import Config
from validator.db.sql import submissions_and_scoring as scoring_sql
from validator.db.sql import tournaments as tournament_sql
from validator.db.sql.quant_references import get_quant_sku_references
from validator.evaluation.quant_artifacts import SHA256_KIND_LFS
from validator.evaluation.quant_artifacts import decoded_model_prefix
from validator.infrastructure.minio_client import async_minio_client
from validator.scoring.models import HubArtifactFile
from validator.scoring.models import QuantScorecardRow
from validator.scoring.models import QuantSkuReferences
from validator.scoring.quant_ranking import reference_on
from validator.tasks.models import AnyQuantRawTask
from validator.tasks.models import QuantFreeformScorecard
from validator.tasks.models import ScorecardArtifact
from validator.tasks.models import ScorecardBytes
from validator.tasks.models import ScorecardDecoded
from validator.tasks.models import ScorecardDecoder
from validator.tasks.models import ScorecardQuality
from validator.tasks.models import ScorecardRelease
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentStatus
from validator.tournament.quant_reference import reference_hotkey_for


logger = get_logger(__name__)


def is_scored(row: QuantScorecardRow) -> bool:
    """A scorecard exists once the row has a loss or a persisted scored fail."""
    return row.test_loss is not None or row.quant_failure is not None


def decode_matches_hub(hub_files: list[HubArtifactFile] | None, decoded_inputs: list[QuantDecodedFile] | None) -> bool | None:
    """The trainer decoded the files the hub lists: same names and bytes, and the same sha256 for every LFS file
    (a git blob id is not a content sha256, so a plain file is compared on name and bytes only). None when either
    list is unknown."""
    if not hub_files or not decoded_inputs:
        return None
    decoded = {f.name: f for f in decoded_inputs}
    if len(decoded) != len(hub_files) or {f.name for f in hub_files} != set(decoded):
        return False
    for hub_file in hub_files:
        counterpart = decoded[hub_file.name]
        if counterpart.bytes != hub_file.bytes:
            return False
        if hub_file.sha256_kind == SHA256_KIND_LFS and counterpart.sha256 != hub_file.sha256:
            return False
    return True


def _usable(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _effective_bytes(total_bytes: int | None, ln_penalty: float | None) -> float | None:
    """total x exp(ln_penalty); a missing penalty is factor 1. None when the penalty is NaN or infinite or the product
    does not fit a float (math.exp raises OverflowError, float * inf gives inf)."""
    if total_bytes is None or (ln_penalty is not None and not math.isfinite(ln_penalty)):
        return None
    try:
        effective = float(total_bytes) * math.exp(ln_penalty or 0.0)
    except OverflowError:
        return None
    return effective if math.isfinite(effective) else None


def build_quant_scorecard(
    row: QuantScorecardRow,
    participant: TournamentParticipant | None,
    reference_loss: float | None,
    decoded_urls: dict[str, str] | None = None,
) -> QuantFreeformScorecard:
    """Scorecard for one scored freeform row. decoded_urls is passed only once the tournament is completed."""
    total_bytes = (
        row.artifact_bytes + row.decoder_bytes if row.artifact_bytes is not None and row.decoder_bytes is not None else None
    )
    ln_penalty = row.freeform_penalty
    effective_bytes = _effective_bytes(total_bytes, ln_penalty)
    ratio = row.test_loss / reference_loss if _usable(row.test_loss) and _usable(reference_loss) and reference_loss > 0 else None

    release = None
    if participant is not None and participant.decoder_released_at is not None:
        release = ScorecardRelease(
            released_at=participant.decoder_released_at,
            decoder_snapshot_url=participant.decoder_released_url,
            training_repo=participant.backup_repo,
            training_commit_hash=participant.training_commit_hash,
        )

    return QuantFreeformScorecard(
        bytes=ScorecardBytes(
            artifact_bytes=row.artifact_bytes,
            decoder_xz_bytes=row.decoder_bytes,
            total_bytes=total_bytes,
            ln_penalty=ln_penalty,
            effective_bytes=effective_bytes,
        ),
        quality=ScorecardQuality(
            test_loss=row.test_loss,
            reference_loss=reference_loss if _usable(reference_loss) else None,
            ratio=ratio,
            passed_bar=row.passed_bar,
            quant_failure=row.quant_failure,
        ),
        artifact=ScorecardArtifact(
            repo=row.repo,
            revision=row.artifact_revision,
            files=row.artifact_files or [],
            decode_matches_hub=decode_matches_hub(row.artifact_files, row.decode_artifact_files),
        ),
        decoder=ScorecardDecoder(
            xz_bytes=participant.decoder_xz_bytes if participant is not None else None,
            xz_sha256=participant.decoder_sha256 if participant is not None else None,
            tar_sha256=participant.decoder_tar_sha256 if participant is not None else None,
            tree_sha=participant.decoder_tree_sha if participant is not None else None,
            xz_version=participant.decoder_xz_version if participant is not None else None,
        ),
        decoded=ScorecardDecoded(files=row.decoded_files or [], urls=decoded_urls, hf_repo=row.decoded_hf_repo),
        release=release,
    )


async def _decoded_urls(task_id: UUID | str, hotkey: str, files: list[QuantDecodedFile]) -> dict[str, str] | None:
    """Fresh presigned GETs of the decoded shards; None when there are none or signing fails."""
    if not files:
        return None
    prefix = decoded_model_prefix(task_id, hotkey)
    try:
        return {f.name: await async_minio_client.get_presigned_url(service_cst.BUCKET_NAME, f"{prefix}/{f.name}") for f in files}
    except Exception as e:
        logger.warning(f"Could not sign decoded model urls for {hotkey} on task {task_id}: {type(e).__name__}")
        return None


async def quant_scorecards_for_task(task: AnyQuantRawTask, config: Config) -> dict[str, QuantFreeformScorecard]:
    """Scorecard per scored hotkey of a freeform task; empty for any other task type."""
    if task.task_type != TaskType.QUANTFREEFORMTASK:
        return {}
    rows = {
        hotkey: row
        for hotkey, row in (await scoring_sql.get_quant_scorecard_rows(task.task_id, config.psql_db)).items()
        if is_scored(row)
    }
    if not rows:
        return {}

    references: QuantSkuReferences = await get_quant_sku_references(task.task_id, config.psql_db)
    participants: dict[str, TournamentParticipant] = {}
    completed = False
    tournament_id = await tournament_sql.get_tournament_id_by_task_id(str(task.task_id), config.psql_db)
    if tournament_id is not None:
        participants = {p.hotkey: p for p in await tournament_sql.get_tournament_participants(tournament_id, config.psql_db)}
        tournament = await tournament_sql.get_tournament(tournament_id, config.psql_db)
        completed = tournament is not None and tournament.status == TournamentStatus.COMPLETED

    scorecards: dict[str, QuantFreeformScorecard] = {}
    for hotkey, row in rows.items():
        reference = reference_on(reference_hotkey_for(task, hotkey), row.eval_gpu, references)
        urls = await _decoded_urls(task.task_id, hotkey, row.decoded_files or []) if completed else None
        scorecards[hotkey] = build_quant_scorecard(row, participants.get(hotkey), reference.loss if reference else None, urls)
    return scorecards
