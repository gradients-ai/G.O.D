"""
Trainer images, built on the image build box before training. Trainers never build: every training job runs one of these.

An image is identified by (scope, source, variant): scope is the tournament id ('untracked' for a training row outside a
tournament), source the code (repo and commit) and variant the Dockerfile family. When a tournament is active, its
participants' images are queued at once, so they are built from the registered commits at the start. Any other training
row (a benchmark, a hotkey added later) queues its own image the first time the scheduler sees it. The same code is built
once per scope, whoever it is for.

For each source the validator copies the commit into a private snapshot repo (branch t/<scope>/<source>), then asks
Woodpecker to build each variant. The build runs on the image build box (ops/image_builder/) and pushes to its Zot
registry; the validator records the digest and the scheduler hands it to the trainer as TrainerProxyRequest.prebuilt_image.
All images share one registry repository, so the layers they have in common are uploaded and stored once.

Row states (table tournament_images): pending -> building -> ready | failed. A pending row whose snapshot or build start
fails is retried each cycle up to MAX_START_ATTEMPTS; the build recipe retries the build itself. A failed image fails the
training rows that need it. The snapshot branch is deleted once every variant of it is ready or failed.

Every IMAGE_BUILD_* setting is required. Without them no image is built and tournament training rows wait.
"""

import asyncio
import hashlib
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import Enum

import httpx

from core.constants.docker import IMAGE_DOCKERFILE_PATHS
from core.constants.docker import IMAGE_TOOLKIT_DOCKERFILE_PATHS
from core.constants.docker import TEXT_DOCKERFILE_PATHS
from core.git import build_authenticated_git_url
from core.git import sanitize_git_text
from core.logging import get_logger
from core.models.image_models import ImageModelType
from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType
from validator.app.config import Config
from validator.db.sql import tournament_images as images_sql
from validator.db.sql import tournaments as tournament_sql
from validator.tournament.models import TournamentImage
from validator.tournament.models import TournamentImageStatus
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TrainerImageVariant


logger = get_logger(__name__)

CYCLE_INTERVAL_SECONDS = 30
MAX_START_ATTEMPTS = 3
# Sources snapshotted and handed to Woodpecker at once. Woodpecker queues the builds and runs them as the agent allows.
START_CONCURRENCY = 8
GIT_TIMEOUT_SECONDS = 600
HTTP_TIMEOUT_SECONDS = 30
# A build still not finished after this long is failed (the recipe's own limit is 60 minutes per run).
BUILD_DEADLINE_SECONDS = 4 * 3600
UNTRACKED_SCOPE = "untracked"
SNAPSHOT_BRANCH_PREFIX = "t/"
REGISTRY_REPOSITORY = "god/trainers"

DOCKERFILE_PATHS: dict[TrainerImageVariant, tuple[str, ...]] = {
    TrainerImageVariant.TEXT: TEXT_DOCKERFILE_PATHS,
    TrainerImageVariant.IMAGE: IMAGE_DOCKERFILE_PATHS,
    TrainerImageVariant.IMAGE_TOOLKIT: IMAGE_TOOLKIT_DOCKERFILE_PATHS,
}
# Compression jobs run the text Dockerfile, like text jobs.
TOURNAMENT_VARIANTS: dict[TournamentType, tuple[TrainerImageVariant, ...]] = {
    TournamentType.TEXT: (TrainerImageVariant.TEXT,),
    TournamentType.QUANT: (TrainerImageVariant.TEXT,),
    TournamentType.IMAGE: (TrainerImageVariant.IMAGE, TrainerImageVariant.IMAGE_TOOLKIT),
}
WOODPECKER_DONE = "success"
WOODPECKER_FAILED = frozenset({"failure", "error", "killed", "declined", "skipped"})
MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


@dataclass(frozen=True)
class ImageBuildSettings:
    woodpecker_url: str  # https://host, without /api
    woodpecker_token: str
    woodpecker_repo_id: int
    snapshot_repo: str  # https://github.com/<owner>/<repo>.git, private
    snapshot_token: str  # read and write on snapshot_repo
    registry: str  # host[:port] of the Zot registry
    registry_username: str  # read-only registry login, to look up digests
    registry_password: str


def load_image_build_settings() -> ImageBuildSettings | None:
    """The IMAGE_BUILD_* settings, or None when any is missing."""
    values = {
        "woodpecker_url": (os.getenv("IMAGE_BUILD_WOODPECKER_URL") or "").rstrip("/"),
        "woodpecker_token": os.getenv("IMAGE_BUILD_WOODPECKER_TOKEN") or "",
        "woodpecker_repo_id": os.getenv("IMAGE_BUILD_WOODPECKER_REPO_ID") or "",
        "snapshot_repo": os.getenv("IMAGE_BUILD_SNAPSHOT_REPO") or "",
        "snapshot_token": os.getenv("IMAGE_BUILD_SNAPSHOT_TOKEN") or "",
        "registry": (os.getenv("IMAGE_BUILD_REGISTRY") or "").removeprefix("https://").rstrip("/"),
        "registry_username": os.getenv("IMAGE_BUILD_REGISTRY_USERNAME") or "",
        "registry_password": os.getenv("IMAGE_BUILD_REGISTRY_PASSWORD") or "",
    }
    if not all(values.values()):
        return None
    return ImageBuildSettings(**{**values, "woodpecker_repo_id": int(values["woodpecker_repo_id"])})


def source_key(repo: str, commit: str | None) -> str:
    return hashlib.sha256(f"{repo}\n{commit or ''}".encode()).hexdigest()[:16]


def snapshot_branch(scope: str, source: str) -> str:
    return f"{SNAPSHOT_BRANCH_PREFIX}{scope}/{source}"


def image_tag(image: TournamentImage) -> str:
    return f"{image.scope}-{image.source}-{image.variant.value}"


def variant_for_task(task_type: TaskType, model_type: ImageModelType | None = None) -> TrainerImageVariant | None:
    """The Dockerfile family a training job runs: FLUX image jobs the image one, other image jobs the toolkit one."""
    if task_type == TaskType.IMAGETASK:
        return TrainerImageVariant.IMAGE if model_type == ImageModelType.FLUX else TrainerImageVariant.IMAGE_TOOLKIT
    if task_type == TaskType.ENVIRONMENTTASK:
        return None
    return TrainerImageVariant.TEXT


def wanted_image(
    scope: str, repo: str, commit: str | None, github_token: str | None, hotkey: str | None, variant: TrainerImageVariant
) -> TournamentImage:
    return TournamentImage(
        scope=scope,
        source=source_key(repo, commit),
        variant=variant,
        repo=repo,
        commit_hash=commit,
        github_token=github_token,
        hotkey=hotkey,
    )


# ---------------------------------------------------------------------------------------------------------------- snapshot


class SnapshotError(RuntimeError):
    pass


def _git(args: list[str], cwd: str, *secrets: str | None) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    proc = subprocess.run(
        ["git", "-c", "credential.helper=", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS
    )
    if proc.returncode != 0:
        raise SnapshotError(sanitize_git_text(f"git {args[0]} failed: {proc.stderr.strip()[-500:]}", *secrets))
    return proc.stdout


def snapshot_submission(
    repo: str, commit: str | None, github_token: str | None, branch: str, settings: ImageBuildSettings
) -> dict[TrainerImageVariant, str | None]:
    """Copy the tree at `commit` (the default branch when None) to `branch` of the snapshot repo, as a single commit
    without history. Returns the first Dockerfile found for each variant (None: the repository has none)."""
    with tempfile.TemporaryDirectory(prefix="god-snapshot-") as work:
        secrets = (github_token, settings.snapshot_token)
        _git(["init", "-q"], work)
        _git(["fetch", "-q", "--depth", "1", build_authenticated_git_url(repo, github_token), commit or "HEAD"], work, *secrets)
        tree = _git(["rev-parse", "FETCH_HEAD^{tree}"], work).strip()
        files = set(_git(["ls-tree", "-r", "--name-only", "FETCH_HEAD"], work).splitlines())
        identity = ["-c", "user.name=god-validator", "-c", "user.email=validator@localhost"]
        snapshot = _git([*identity, "commit-tree", tree, "-m", f"snapshot of {repo} at {commit or 'HEAD'}"], work).strip()
        target = build_authenticated_git_url(settings.snapshot_repo, settings.snapshot_token)
        _git(["push", "-q", "--force", target, f"{snapshot}:refs/heads/{branch}"], work, *secrets)
    return {variant: next((p for p in paths if p in files), None) for variant, paths in DOCKERFILE_PATHS.items()}


def delete_snapshot_branch(branch: str, settings: ImageBuildSettings) -> None:
    with tempfile.TemporaryDirectory(prefix="god-snapshot-") as work:
        _git(["init", "-q"], work)  # git push needs a repository, even to only delete a remote branch
        target = build_authenticated_git_url(settings.snapshot_repo, settings.snapshot_token)
        _git(["push", "-q", target, f":refs/heads/{branch}"], work, settings.snapshot_token)


# --------------------------------------------------------------------------------------------------------- build service


class BuildServiceError(RuntimeError):
    pass


async def start_build(settings: ImageBuildSettings, branch: str, dockerfile: str, tag: str) -> int:
    """Start the Woodpecker build of one image; returns its pipeline number."""
    variables = {
        "SNAPSHOT": branch,
        "DOCKERFILE": dockerfile,
        "IMAGE": f"{REGISTRY_REPOSITORY}:{tag}",
        "REGISTRY": settings.registry,
        "SNAPSHOT_REPO": settings.snapshot_repo,
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{settings.woodpecker_url}/api/repos/{settings.woodpecker_repo_id}/pipelines",
            headers={"Authorization": f"Bearer {settings.woodpecker_token}"},
            json={"branch": "main", "variables": variables},
        )
    if response.status_code >= 300:
        raise BuildServiceError(f"Woodpecker refused the build ({response.status_code}): {response.text[:300]}")
    return int(response.json()["number"])


async def build_state(settings: ImageBuildSettings, build_number: int) -> tuple[str, int | None]:
    """(status, created unix time) of a Woodpecker pipeline; status 'missing' when Woodpecker no longer knows it."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(
            f"{settings.woodpecker_url}/api/repos/{settings.woodpecker_repo_id}/pipelines/{build_number}",
            headers={"Authorization": f"Bearer {settings.woodpecker_token}"},
        )
    if response.status_code == 404:
        return "missing", None
    if response.status_code >= 300:
        raise BuildServiceError(f"Woodpecker status lookup failed ({response.status_code})")
    body = response.json()
    return str(body.get("status", "")), body.get("created")


async def image_digest(settings: ImageBuildSettings, tag: str) -> str:
    """The digest the registry serves for `tag` (what the build pushed)."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.head(
            f"https://{settings.registry}/v2/{REGISTRY_REPOSITORY}/manifests/{tag}",
            headers={"Accept": MANIFEST_ACCEPT},
            auth=(settings.registry_username, settings.registry_password),
        )
    digest = response.headers.get("Docker-Content-Digest")
    if response.status_code != 200 or not digest:
        raise BuildServiceError(f"Registry has no manifest for {tag} ({response.status_code})")
    return digest


def image_ref(settings: ImageBuildSettings, digest: str) -> str:
    return f"{settings.registry}/{REGISTRY_REPOSITORY}@{digest}"


# ----------------------------------------------------------------------------------------------------------------- cycle


async def add_tournament_images(tournament_id: str, tournament_type: TournamentType, config: Config) -> None:
    """Queue every participant's images when the tournament is active (existing rows are kept)."""
    variants = TOURNAMENT_VARIANTS.get(tournament_type, ())
    if not variants:
        return
    wanted = []
    for participant in await tournament_sql.get_tournament_participants(tournament_id, config.psql_db):
        repo = await tournament_sql.get_tournament_training_repo_and_commit(participant.hotkey, tournament_id, config.psql_db)
        if not repo.training_repo:
            continue
        for variant in variants:
            wanted.append(
                wanted_image(
                    tournament_id, repo.training_repo, repo.training_commit_hash, repo.github_token, participant.hotkey, variant
                )
            )
    await images_sql.add_pending_images(wanted, config.psql_db)


async def _start_source(rows: list[TournamentImage], settings: ImageBuildSettings, config: Config) -> None:
    """Snapshot one source and start the build of each of its pending variants."""
    first = rows[0]
    branch = snapshot_branch(first.scope, first.source)
    try:
        dockerfiles = await asyncio.to_thread(
            snapshot_submission, first.repo, first.commit_hash, first.github_token, branch, settings
        )
    except Exception as e:
        error = f"snapshot failed: {e}"
        logger.warning(f"Image build for {first.repo} ({branch}): {error}")
        for row in rows:
            await images_sql.record_start_failure(row, error, MAX_START_ATTEMPTS, config.psql_db)
        return

    for row in rows:
        dockerfile = dockerfiles[row.variant]
        if dockerfile is None:
            paths = ", ".join(DOCKERFILE_PATHS[row.variant])
            await images_sql.mark_image_failed(row, f"no Dockerfile at any of {paths}", config.psql_db)
            continue
        try:
            number = await start_build(settings, branch, dockerfile, image_tag(row))
        except Exception as e:
            logger.warning(f"Image build for {branch} ({row.variant.value}) did not start: {e}")
            await images_sql.record_start_failure(row, f"build did not start: {e}", MAX_START_ATTEMPTS, config.psql_db)
            continue
        await images_sql.mark_image_building(row, number, config.psql_db)
        logger.info(f"Image build #{number} started for {first.repo} ({branch}, {row.variant.value}, {dockerfile})")
    await _drop_snapshot_if_done(first, settings, config)


async def start_pending_builds(settings: ImageBuildSettings, config: Config) -> None:
    pending = await images_sql.get_images_with_status(TournamentImageStatus.PENDING, config.psql_db)
    by_source: dict[tuple[str, str], list[TournamentImage]] = {}
    for row in pending:
        by_source.setdefault((row.scope, row.source), []).append(row)
    semaphore = asyncio.Semaphore(START_CONCURRENCY)

    async def start(rows: list[TournamentImage]) -> None:
        async with semaphore:
            await _start_source(rows, settings, config)

    await asyncio.gather(*(start(rows) for rows in by_source.values()))


async def _drop_snapshot_if_done(image: TournamentImage, settings: ImageBuildSettings, config: Config) -> None:
    """Delete the snapshot branch once no variant of it is pending or building (a rebuild takes a new snapshot)."""
    if await images_sql.count_unfinished(image.scope, image.source, config.psql_db):
        return
    try:
        await asyncio.to_thread(delete_snapshot_branch, snapshot_branch(image.scope, image.source), settings)
    except Exception as e:
        logger.warning(f"Could not delete snapshot branch {snapshot_branch(image.scope, image.source)}: {e}")


async def refresh_building_images(settings: ImageBuildSettings, config: Config) -> None:
    for row in await images_sql.get_images_with_status(TournamentImageStatus.BUILDING, config.psql_db):
        label = f"{image_tag(row)} build #{row.build_number}"
        try:
            status, created = await build_state(settings, row.build_number)
            if status == WOODPECKER_DONE:
                digest = await image_digest(settings, image_tag(row))
                await images_sql.mark_image_ready(row, image_ref(settings, digest), config.psql_db)
                logger.info(f"Image ready: {label}: {digest}")
            elif status in WOODPECKER_FAILED or status == "missing":
                await images_sql.mark_image_failed(row, f"build #{row.build_number} ended: {status}", config.psql_db)
                logger.warning(f"Image build failed: {label}: {status}")
            elif created and time.time() - created > BUILD_DEADLINE_SECONDS:
                error = f"build #{row.build_number} still {status} after the deadline"
                await images_sql.mark_image_failed(row, error, config.psql_db)
                logger.warning(f"Image build timed out: {label}")
            else:
                continue
            await _drop_snapshot_if_done(row, settings, config)
        except Exception as e:
            logger.warning(f"Could not refresh {label}: {e}")


async def run_image_builds_cycle(settings: ImageBuildSettings, config: Config) -> None:
    for tournament in await tournament_sql.get_tournaments_with_status(TournamentStatus.ACTIVE, config.psql_db):
        await add_tournament_images(tournament.tournament_id, tournament.tournament_type, config)
    await start_pending_builds(settings, config)
    await refresh_building_images(settings, config)


async def process_tournament_image_builds(config: Config) -> None:
    while True:
        settings = load_image_build_settings()
        if settings is None:
            logger.error("Image builds are not configured (IMAGE_BUILD_* settings): no images, so no training starts")
        else:
            try:
                await run_image_builds_cycle(settings, config)
            except Exception as e:
                logger.error(f"Error in the image build cycle: {e}", exc_info=True)
        await asyncio.sleep(CYCLE_INTERVAL_SECONDS)


# ------------------------------------------------------------------------------------------------------- scheduler gate


class ImageGate(str, Enum):
    READY = "ready"
    WAIT = "wait"
    FAILED = "failed"


async def image_for_training(
    tournament_id: str | None,
    hotkey: str,
    training_repo: str,
    training_commit_hash: str | None,
    github_token: str | None,
    task_type: TaskType,
    model_type: ImageModelType | None,
    config: Config,
) -> tuple[ImageGate, str | None]:
    """The image a training row runs: (READY, image_ref), (WAIT, None) while it is queued or building (queueing it if
    nobody has), or (FAILED, error)."""
    variant = variant_for_task(task_type, model_type)
    if variant is None:
        return ImageGate.FAILED, f"no trainer image for task type {task_type}"
    if load_image_build_settings() is None:
        return ImageGate.WAIT, None
    wanted = wanted_image(
        tournament_id or UNTRACKED_SCOPE, training_repo, training_commit_hash, github_token, hotkey, variant
    )
    image = await images_sql.get_image(wanted.scope, wanted.source, variant, config.psql_db)
    if image is None:
        await images_sql.add_pending_images([wanted], config.psql_db)
        return ImageGate.WAIT, None
    if image.status == TournamentImageStatus.READY:
        return ImageGate.READY, image.image_ref
    if image.status == TournamentImageStatus.FAILED:
        return ImageGate.FAILED, image.error
    return ImageGate.WAIT, None
