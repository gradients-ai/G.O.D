"""Public exposure of freeform quant results, run once a tournament has a winner.

release_freeform_decoder: the champion handover. The participant's decoder snapshot (the exact .tar.xz that was counted
and run) is copied from MinIO to a public HF dataset with a release.json that ties it to the code commit and to every
freeform task result. Policy: a decoder is public only for the tournament's champion (winner_hotkey) and only after its
code is pushed (backup_repo); the boss's own row needs the explicit allow_boss for the one-off reigning-champion release.

publish_boss_round_decoded_models: the decoded bf16 model of every boss-round freeform submission, built from the
sha256-verified shards in MinIO plus the base model's config/tokenizer, published as a public HF model repo.

Both are best-effort: a failure logs (never a token or a presigned URL), alerts Discord and is swallowed, so the
tournament's completion and the code push are never undone by them.

Neither runs inline in the tournament cycle. The tournament hooks only schedule them (schedule_freeform_release,
schedule_boss_round_publication: an asyncio task, strongly referenced until done), because the boss-round publication
moves GBs and advance_tournament never revisits a tournament once winner_hotkey is set. A restart or an HF outage would
lose a scheduled task, so reconcile_freeform_publications re-derives what is still unpublished from the database and
redoes exactly that; process_freeform_publications runs it hourly and ops/tools/quant/reconcile_freeform_publications.py
runs it (or one hand release) on demand. One lock serialises all publication work in a process; both publishers re-read
their per-item state after taking it, so a reconcile pass cannot repeat what a background task just finished.
"""

import asyncio
import hashlib
import tempfile
import weakref
from datetime import datetime
from datetime import timezone
from pathlib import Path

from huggingface_hub import HfApi
from pydantic import BaseModel

import core.constants.docker as docker_cst
import validator.infrastructure.service_constants as service_cst
from core.constants.credentials import HUGGINGFACE_TOKEN
from core.constants.credentials import RAYONLABS_HF_USERNAME
from core.logging import get_logger
from core.models.quant_artifact_models import QuantDecodedFile
from core.models.task_models import TaskType
from validator.app.config import Config
from validator.db.database import PSQLDB
from validator.db.sql import submissions_and_scoring as scoring_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournaments as tournament_sql
from validator.evaluation.quant_artifacts import decoded_model_prefix
from validator.evaluation.quant_artifacts import materialise_decoded_model
from validator.infrastructure.minio_client import async_minio_client
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import HubArtifactFile
from validator.tournament.models import FreeformReconcileReport
from validator.tournament.models import PendingDecoderRelease
from validator.tournament.notifications import notify_freeform_release_problem
from validator.tournament.quant_reference import quant_reference_model


logger = get_logger(__name__)

RELEASES_DATASET_NAME = "freeform-releases"
RELEASE_JSON_NAME = "release.json"
DECODED_REPO_SUFFIX = "-decoded"
HASH_CHUNK_BYTES = 1024 * 1024

# Strong references so a scheduled publication cannot be garbage-collected mid-run.
_background_tasks: set[asyncio.Task] = set()
# One lock per running event loop: an asyncio.Lock binds to the loop it first contends on.
_publication_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()


class FreeformReleaseDecoder(BaseModel):
    xz_bytes: int
    xz_sha256: str
    tar_sha256: str | None = None
    tree_sha: str | None = None
    xz_version: str | None = None


class FreeformReleaseTask(BaseModel):
    task_id: str
    repo: str | None = None
    revision: str | None = None
    artifact_files: list[HubArtifactFile] = []
    decoded_files: list[QuantDecodedFile] = []


class FreeformRelease(BaseModel):
    """release.json next to a released decoder snapshot."""

    tournament_id: str
    hotkey: str
    training_commit_hash: str | None = None
    training_repo: str | None = None
    decoder: FreeformReleaseDecoder
    tasks: list[FreeformReleaseTask] = []


def _hf_api() -> HfApi:
    return HfApi(token=HUGGINGFACE_TOKEN)


def _releases_repo() -> str:
    return f"{RAYONLABS_HF_USERNAME}/{RELEASES_DATASET_NAME}"


def _release_path(tournament_id: str, hotkey: str, name: str) -> str:
    return f"{tournament_id}/{hotkey}/{name}"


def _resolve_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _release_tasks(tournament_id: str, hotkey: str, psql_db: PSQLDB) -> list[FreeformReleaseTask]:
    """This hotkey's result on every freeform task of the tournament."""
    tasks: list[FreeformReleaseTask] = []
    for task_id in await tournament_sql.get_tournament_task_ids_of_type(tournament_id, TaskType.QUANTFREEFORMTASK, psql_db):
        row = (await scoring_sql.get_quant_scorecard_rows(task_id, psql_db)).get(hotkey)
        if row is None:
            continue
        tasks.append(
            FreeformReleaseTask(
                task_id=task_id,
                repo=row.repo,
                revision=row.artifact_revision,
                artifact_files=row.artifact_files or [],
                decoded_files=row.decoded_files or [],
            )
        )
    return tasks


async def release_freeform_decoder(
    tournament_id: str, hotkey: str, config: Config, psql_db: PSQLDB, allow_boss: bool = False
) -> str | None:
    """Publish the participant's decoder snapshot; the public URL, or None when skipped, refused or failed. Never raises.

    Refused unless the hotkey is the tournament's winner with its code pushed. The boss (burn) hotkey is skipped unless
    allow_boss, which is only for the one-off release of the reigning champion's pinned decoder.
    """
    is_boss = hotkey == EMISSION_BURN_HOTKEY
    if is_boss and not allow_boss:
        return None
    try:
        if not (is_boss and allow_boss):
            tournament = await tournament_sql.get_tournament(tournament_id, psql_db)
            if tournament is None or not tournament.winner_hotkey or tournament.winner_hotkey != hotkey:
                logger.warning(f"{hotkey} is not the winner of {tournament_id}; its decoder stays private")
                return None
        participant = await tournament_sql.get_tournament_participant(tournament_id, hotkey, psql_db)
        if participant is None or not participant.has_decoder_snapshot:
            logger.info(f"No decoder snapshot for {hotkey} in {tournament_id}; nothing to release")
            return None
        if participant.decoder_released_at is not None and participant.decoder_released_url:
            logger.info(f"Decoder of {hotkey} in {tournament_id} already released")
            return participant.decoder_released_url
        if not participant.backup_repo:
            logger.warning(f"{hotkey} in {tournament_id} has no backup_repo yet; the code push must happen before the release")
            return None

        with tempfile.TemporaryDirectory() as work:
            snapshot_path = Path(work) / "decoder.tar.xz"
            await async_minio_client.download_file(service_cst.BUCKET_NAME, participant.decoder_object_name, str(snapshot_path))
            actual_sha = await asyncio.to_thread(_sha256_of, snapshot_path)
            actual_bytes = snapshot_path.stat().st_size
            if actual_sha != participant.decoder_sha256 or actual_bytes != participant.decoder_xz_bytes:
                message = (
                    f"decoder snapshot of {hotkey} in {tournament_id} does not match its registration "
                    f"(sha256 {actual_sha} vs {participant.decoder_sha256}, "
                    f"bytes {actual_bytes} vs {participant.decoder_xz_bytes}); "
                    "refusing to publish it"
                )
                logger.error(message)
                await notify_freeform_release_problem("decoder release refused", message, config.discord_url)
                return None

            release = FreeformRelease(
                tournament_id=tournament_id,
                hotkey=hotkey,
                training_commit_hash=participant.training_commit_hash,
                training_repo=participant.backup_repo,
                decoder=FreeformReleaseDecoder(
                    xz_bytes=participant.decoder_xz_bytes,
                    xz_sha256=participant.decoder_sha256,
                    tar_sha256=participant.decoder_tar_sha256,
                    tree_sha=participant.decoder_tree_sha,
                    xz_version=participant.decoder_xz_version,
                ),
                tasks=await _release_tasks(tournament_id, hotkey, psql_db),
            )
            repo = _releases_repo()
            snapshot_name = _release_path(tournament_id, hotkey, f"decoder-{participant.decoder_sha256}.tar.xz")
            release_name = _release_path(tournament_id, hotkey, RELEASE_JSON_NAME)

            def _publish() -> None:
                api = _hf_api()
                api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
                api.upload_file(path_or_fileobj=str(snapshot_path), path_in_repo=snapshot_name, repo_id=repo, repo_type="dataset")
                api.upload_file(
                    path_or_fileobj=release.model_dump_json(indent=2).encode(),
                    path_in_repo=release_name,
                    repo_id=repo,
                    repo_type="dataset",
                )

            await asyncio.to_thread(_publish)

        url = _resolve_url(repo, snapshot_name)
        await tournament_sql.set_decoder_released(tournament_id, hotkey, url, datetime.now(timezone.utc), psql_db)
        logger.info(f"Released decoder snapshot of {hotkey} in {tournament_id}")
        return url
    except Exception as e:
        logger.error(f"Decoder release failed for {hotkey} in {tournament_id}: {type(e).__name__}")
        await notify_freeform_release_problem(
            "decoder release failed", f"{hotkey} in {tournament_id}: {type(e).__name__}", config.discord_url
        )
        return None


async def _publish_decoded_model(
    task_id: str, hotkey: str, original_model: str, repo_name: str, files: list[QuantDecodedFile], psql_db: PSQLDB
) -> str:
    prefix = decoded_model_prefix(task_id, hotkey)
    names = [docker_cst.DECODED_MODEL_INDEX_NAME] + [f.name for f in files]
    urls = {name: await async_minio_client.get_presigned_url(service_cst.BUCKET_NAME, f"{prefix}/{name}") for name in names}
    repo = f"{RAYONLABS_HF_USERNAME}/{repo_name}{DECODED_REPO_SUFFIX}"
    with tempfile.TemporaryDirectory() as work:
        dest = Path(work) / "model"
        model_dir = await asyncio.to_thread(materialise_decoded_model, original_model, urls, dest, HUGGINGFACE_TOKEN)

        def _upload() -> None:
            api = _hf_api()
            api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
            api.upload_folder(folder_path=str(model_dir), repo_id=repo, repo_type="model")

        await asyncio.to_thread(_upload)
    await scoring_sql.set_decoded_hf_repo(task_id, hotkey, repo, psql_db)
    return repo


async def publish_boss_round_decoded_models(tournament_id: str, config: Config, psql_db: PSQLDB) -> list[str]:
    """Public HF repo of every boss-round freeform submission not yet published; the repos created. Never raises."""
    published: list[str] = []
    try:
        task_ids = await tournament_sql.get_tournament_task_ids_of_type(
            tournament_id, TaskType.QUANTFREEFORMTASK, psql_db, final_round_only=True
        )
        for task_id in task_ids:
            task = await task_sql.get_task_by_id(task_id, psql_db)
            rows = await scoring_sql.get_quant_scorecard_rows(task_id, psql_db)
            for hotkey, row in rows.items():
                if not row.decoded_files or row.decoded_hf_repo:
                    continue
                try:
                    original_model = await quant_reference_model(task, hotkey, psql_db)
                    repo_name = await task_sql.get_expected_repo_name(task_id, hotkey, psql_db)
                    if not original_model or not repo_name:
                        logger.warning(f"Task {task_id} {hotkey}: no base model or repo name; decoded model not published")
                        continue
                    repo = await _publish_decoded_model(task_id, hotkey, original_model, repo_name, row.decoded_files, psql_db)
                    published.append(repo)
                except Exception as e:
                    logger.error(f"Decoded model publication failed for {hotkey} on task {task_id}: {type(e).__name__}")
                    await notify_freeform_release_problem(
                        "decoded model publication failed", f"{hotkey} on task {task_id}: {type(e).__name__}", config.discord_url
                    )
    except Exception as e:
        logger.error(f"Boss-round decoded publication failed for {tournament_id}: {type(e).__name__}")
        await notify_freeform_release_problem(
            "boss-round decoded publication failed", f"{tournament_id}: {type(e).__name__}", config.discord_url
        )
    return published


def _publication_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _publication_locks.get(loop)
    if lock is None:
        lock = _publication_locks[loop] = asyncio.Lock()
    return lock


async def _locked_release(tournament_id: str, hotkey: str, config: Config, psql_db: PSQLDB) -> str | None:
    async with _publication_lock():
        return await release_freeform_decoder(tournament_id, hotkey, config, psql_db)


async def _locked_publish_boss_round(tournament_id: str, config: Config, psql_db: PSQLDB) -> list[str]:
    async with _publication_lock():
        return await publish_boss_round_decoded_models(tournament_id, config, psql_db)


def _on_task_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.error(f"Background freeform publication raised: {type(task.exception()).__name__}")


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_on_task_done)


def schedule_freeform_release(tournament_id: str, hotkey: str, config: Config, psql_db: PSQLDB) -> None:
    """Release the decoder in the background and return at once. The reconcile picks up whatever this loses."""
    _spawn(_locked_release(tournament_id, hotkey, config, psql_db))


def schedule_boss_round_publication(tournament_id: str, config: Config, psql_db: PSQLDB) -> None:
    """Publish the boss round's decoded models in the background and return at once."""
    _spawn(_locked_publish_boss_round(tournament_id, config, psql_db))


async def wait_for_background_publications() -> None:
    """Block until every publication scheduled on this event loop has finished (tests and the CLI)."""
    loop = asyncio.get_running_loop()
    while True:
        running = [t for t in _background_tasks if not t.done() and t.get_loop() is loop]
        if not running:
            return
        await asyncio.gather(*running, return_exceptions=True)


async def reconcile_freeform_publications(config: Config, psql_db: PSQLDB, apply: bool = True) -> FreeformReconcileReport:
    """Find the winners' decoders and boss-round decoded models still unpublished and, with apply, publish them.

    Safe to run any time and repeatedly: both publishers are idempotent and run under the publication lock.
    """
    pending = await tournament_sql.get_pending_decoder_releases(EMISSION_BURN_HOTKEY, psql_db)
    blocked = await tournament_sql.get_blocked_decoder_releases(EMISSION_BURN_HOTKEY, psql_db)
    decoded = await tournament_sql.get_tournaments_with_pending_decoded_publication(psql_db)
    report = FreeformReconcileReport(
        applied=apply, pending_releases=pending, blocked_releases=blocked, pending_decoded_tournaments=decoded
    )
    for item in blocked:
        logger.warning(f"Champion {item.hotkey} of {item.tournament_id} has no backup_repo; its decoder cannot be released yet")
    if not apply:
        return report

    for item in pending:
        url = await _locked_release(item.tournament_id, item.hotkey, config, psql_db)
        (report.released if url else report.failed_releases).append(PendingDecoderRelease(**item.model_dump()))
    for tournament_id in decoded:
        report.published_models.extend(await _locked_publish_boss_round(tournament_id, config, psql_db))
    logger.info(
        f"Freeform reconcile: {len(report.released)} decoders released, {len(report.failed_releases)} failed, "
        f"{len(report.published_models)} decoded models published, {len(blocked)} blocked"
    )
    return report
