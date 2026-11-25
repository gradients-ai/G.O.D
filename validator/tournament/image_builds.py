"""
Trainer images built once per tournament participant, when the tournament is active, instead of on every training job.

For each participant the validator copies the registered commit into a private snapshot repo (branch
t/<tournament_id>/<hotkey>), then asks Woodpecker to build each Dockerfile variant the tournament needs. The build runs on
the image build box (ops/image_builder/) and pushes to its Zot registry; the validator records the digest and the
scheduler hands it to the trainer as TrainerProxyRequest.prebuilt_image. All images share one registry repository, so the
layers participants have in common (base images, dependencies) are uploaded and stored once.

Row states (table tournament_images): pending -> building -> ready | failed. A pending row whose snapshot or build start
fails is retried each cycle up to MAX_START_ATTEMPTS. The build recipe retries the build itself. A failed image fails the
participant's training jobs that need it, the same as a failed build on the trainer.

Disabled (the old per-job builds on the trainer) unless every IMAGE_BUILD_* setting below is set.
"""

import asyncio
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
# Participants snapshotted and handed to Woodpecker at once. Woodpecker queues the builds and runs them as the agent allows.
START_CONCURRENCY = 8
GIT_TIMEOUT_SECONDS = 600
HTTP_TIMEOUT_SECONDS = 30
# A build still not finished after this long is failed (the recipe's own limit is 60 minutes per attempt run).
BUILD_DEADLINE_SECONDS = 4 * 3600
SNAPSHOT_PRUNE_INTERVAL_SECONDS = 24 * 3600
SNAPSHOT_BRANCH_PREFIX = "t/"
REGISTRY_REPOSITORY = "god/trainers"

DOCKERFILE_PATHS: dict[TrainerImageVariant, tuple[str, ...]] = {
    TrainerImageVariant.TEXT: TEXT_DOCKERFILE_PATHS,
    TrainerImageVariant.IMAGE: IMAGE_DOCKERFILE_PATHS,
    TrainerImageVariant.IMAGE_TOOLKIT: IMAGE_TOOLKIT_DOCKERFILE_PATHS,
}
# Compression jobs build the text Dockerfile, like text jobs.
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
    """The IMAGE_BUILD_* settings, or None (feature off) unless all are set."""
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


def snapshot_branch(tournament_id: str, hotkey: str) -> str:
    return f"{SNAPSHOT_BRANCH_PREFIX}{tournament_id}/{hotkey}"


def image_tag(tournament_id: str, hotkey: str, variant: TrainerImageVariant) -> str:
    return f"{tournament_id}-{hotkey}-{variant.value}"


def variant_for_task(task_type: TaskType, model_type: ImageModelType | None = None) -> TrainerImageVariant | None:
    """The image a training job runs, mirroring the trainer's Dockerfile choice (trainer.runtime.get_dockerfile_path)."""
    if task_type == TaskType.IMAGETASK:
        return TrainerImageVariant.IMAGE if model_type == ImageModelType.FLUX else TrainerImageVariant.IMAGE_TOOLKIT
    if task_type == TaskType.ENVIRONMENTTASK:
        return None
    return TrainerImageVariant.TEXT


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
    """Copy the submission's tree at `commit` (the default branch when None) to `branch` of the snapshot repo, as a single
    commit without history. Returns the first Dockerfile found for each variant (None: the submission has none)."""
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


def list_snapshot_branches(settings: ImageBuildSettings) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="god-snapshot-") as work:
        target = build_authenticated_git_url(settings.snapshot_repo, settings.snapshot_token)
        out = _git(["ls-remote", "--heads", target, f"refs/heads/{SNAPSHOT_BRANCH_PREFIX}*"], work, settings.snapshot_token)
    return [line.split("refs/heads/", 1)[1] for line in out.splitlines() if "refs/heads/" in line]


def delete_snapshot_branches(branches: list[str], settings: ImageBuildSettings) -> None:
    with tempfile.TemporaryDirectory(prefix="god-snapshot-") as work:
        _git(["init", "-q"], work)  # git push needs a repository, even to only delete remote branches
        target = build_authenticated_git_url(settings.snapshot_repo, settings.snapshot_token)
        for start in range(0, len(branches), 100):
            refs = [f":refs/heads/{branch}" for branch in branches[start : start + 100]]
            _git(["push", "-q", target, *refs], work, settings.snapshot_token)


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
    """A pending row for every participant and variant of the tournament that has none yet."""
    variants = TOURNAMENT_VARIANTS.get(tournament_type, ())
    if not variants:
        return
    participants = await tournament_sql.get_tournament_participants(tournament_id, config.psql_db)
    wanted = [(p.hotkey, variant) for p in participants for variant in variants]
    await images_sql.add_pending_images(tournament_id, wanted, config.psql_db)


async def _start_participant(rows: list[TournamentImage], settings: ImageBuildSettings, config: Config) -> None:
    """Snapshot one participant's submission and start the build of each of its pending variants."""
    first = rows[0]
    tag_context = f"{first.tournament_id}/{first.hotkey}"
    try:
        repo = await tournament_sql.get_tournament_training_repo_and_commit(first.hotkey, first.tournament_id, config.psql_db)
        if not repo.training_repo:
            for row in rows:
                await images_sql.mark_image_failed(row, "no training repository registered", config.psql_db)
            return
        branch = snapshot_branch(first.tournament_id, first.hotkey)
        dockerfiles = await asyncio.to_thread(
            snapshot_submission, repo.training_repo, repo.training_commit_hash, repo.github_token, branch, settings
        )
    except Exception as e:
        error = f"snapshot failed: {e}"
        logger.warning(f"Image build for {tag_context}: {error}")
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
            number = await start_build(settings, branch, dockerfile, image_tag(row.tournament_id, row.hotkey, row.variant))
        except Exception as e:
            logger.warning(f"Image build for {tag_context} ({row.variant.value}) did not start: {e}")
            await images_sql.record_start_failure(row, f"build did not start: {e}", MAX_START_ATTEMPTS, config.psql_db)
            continue
        await images_sql.mark_image_building(row, number, config.psql_db)
        logger.info(f"Image build #{number} started for {tag_context} ({row.variant.value}, {dockerfile})")


async def start_pending_builds(settings: ImageBuildSettings, config: Config) -> None:
    pending = await images_sql.get_images_with_status(TournamentImageStatus.PENDING, config.psql_db)
    by_participant: dict[tuple[str, str], list[TournamentImage]] = {}
    for row in pending:
        by_participant.setdefault((row.tournament_id, row.hotkey), []).append(row)
    semaphore = asyncio.Semaphore(START_CONCURRENCY)

    async def start(rows: list[TournamentImage]) -> None:
        async with semaphore:
            await _start_participant(rows, settings, config)

    await asyncio.gather(*(start(rows) for rows in by_participant.values()))


async def refresh_building_images(settings: ImageBuildSettings, config: Config) -> None:
    for row in await images_sql.get_images_with_status(TournamentImageStatus.BUILDING, config.psql_db):
        label = f"{row.tournament_id}/{row.hotkey} ({row.variant.value}) build #{row.build_number}"
        try:
            status, created = await build_state(settings, row.build_number)
            if status == WOODPECKER_DONE:
                digest = await image_digest(settings, image_tag(row.tournament_id, row.hotkey, row.variant))
                await images_sql.mark_image_ready(row, image_ref(settings, digest), config.psql_db)
                logger.info(f"Image ready for {label}: {digest}")
            elif status in WOODPECKER_FAILED or status == "missing":
                await images_sql.mark_image_failed(row, f"build #{row.build_number} ended: {status}", config.psql_db)
                logger.warning(f"Image build failed for {label}: {status}")
            elif created and time.time() - created > BUILD_DEADLINE_SECONDS:
                await images_sql.mark_image_failed(
                    row, f"build #{row.build_number} still {status} after the deadline", config.psql_db
                )
                logger.warning(f"Image build timed out for {label}")
        except Exception as e:
            logger.warning(f"Could not refresh {label}: {e}")


async def prune_snapshot_branches(settings: ImageBuildSettings, config: Config) -> None:
    """Delete the snapshot branches of tournaments that are over (completed, cancelled or gone)."""
    branches = await asyncio.to_thread(list_snapshot_branches, settings)
    by_tournament: dict[str, list[str]] = {}
    for branch in branches:
        by_tournament.setdefault(branch.removeprefix(SNAPSHOT_BRANCH_PREFIX).split("/", 1)[0], []).append(branch)
    finished = []
    for tournament_id, tournament_branches in by_tournament.items():
        tournament = await tournament_sql.get_tournament(tournament_id, config.psql_db)
        if tournament is None or tournament.status not in (TournamentStatus.PENDING, TournamentStatus.ACTIVE):
            finished.extend(tournament_branches)
    if finished:
        await asyncio.to_thread(delete_snapshot_branches, finished, settings)
        logger.info(f"Deleted {len(finished)} snapshot branches of finished tournaments")


async def run_image_builds_cycle(settings: ImageBuildSettings, config: Config) -> None:
    for tournament in await tournament_sql.get_tournaments_with_status(TournamentStatus.ACTIVE, config.psql_db):
        await add_tournament_images(tournament.tournament_id, tournament.tournament_type, config)
    await start_pending_builds(settings, config)
    await refresh_building_images(settings, config)


async def process_tournament_image_builds(config: Config) -> None:
    settings = load_image_build_settings()
    if settings is None:
        logger.info("Tournament image builds are off (IMAGE_BUILD_* not set); trainers build images per job")
        return
    last_prune = 0.0
    while True:
        try:
            await run_image_builds_cycle(settings, config)
            if time.time() - last_prune > SNAPSHOT_PRUNE_INTERVAL_SECONDS:
                await prune_snapshot_branches(settings, config)
                last_prune = time.time()
        except Exception as e:
            logger.error(f"Error in the tournament image build cycle: {e}", exc_info=True)
        await asyncio.sleep(CYCLE_INTERVAL_SECONDS)


# ------------------------------------------------------------------------------------------------------- scheduler gate


class ImageGate(str, Enum):
    NOT_USED = "not_used"  # build on the trainer as before (feature off, not a tournament participant, no variant)
    READY = "ready"
    WAIT = "wait"
    FAILED = "failed"


async def image_for_training(
    tournament_id: str | None,
    hotkey: str,
    task_type: TaskType,
    model_type: ImageModelType | None,
    config: Config,
) -> tuple[ImageGate, str | None]:
    """What the scheduler does with a training row: send the prebuilt image, wait for it, fail the row, or build as before.

    Returns (gate, image_ref or error)."""
    if tournament_id is None or load_image_build_settings() is None:
        return ImageGate.NOT_USED, None
    variant = variant_for_task(task_type, model_type)
    if variant is None:
        return ImageGate.NOT_USED, None
    image = await images_sql.get_tournament_image(tournament_id, hotkey, variant, config.psql_db)
    if image is None:
        tournament = await tournament_sql.get_tournament(tournament_id, config.psql_db)
        if tournament is None or variant not in TOURNAMENT_VARIANTS.get(tournament.tournament_type, ()):
            return ImageGate.NOT_USED, None
        repo = await tournament_sql.get_tournament_training_repo_and_commit(hotkey, tournament_id, config.psql_db)
        # Not a participant (no registered repo): nothing is built for it.
        return (ImageGate.WAIT, None) if repo.training_repo else (ImageGate.NOT_USED, None)
    if image.status == TournamentImageStatus.READY:
        return ImageGate.READY, image.image_ref
    if image.status == TournamentImageStatus.FAILED:
        return ImageGate.FAILED, image.error
    return ImageGate.WAIT, None
