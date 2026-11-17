import httpx

from core.models.tournament_models import TournamentType
from validator.db.sql import image_builds as image_build_sql
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_participants
from validator.tournament.models import ImageBuildVariant


DOCKERFILES = {
    ImageBuildVariant.TEXT: (
        "ops/docker/standalone-text-trainer.dockerfile",
        "dockerfiles/standalone-text-trainer.dockerfile",
    ),
    ImageBuildVariant.IMAGE_FLUX: (
        "ops/docker/standalone-image-trainer.dockerfile",
        "dockerfiles/standalone-image-trainer.dockerfile",
    ),
    ImageBuildVariant.IMAGE_TOOLKIT: (
        "ops/docker/standalone-image-toolkit-trainer.dockerfile",
        "dockerfiles/standalone-image-toolkit-trainer.dockerfile",
    ),
}


def _variants(tournament_type: TournamentType) -> tuple[ImageBuildVariant, ...]:
    if tournament_type == TournamentType.IMAGE:
        return (ImageBuildVariant.IMAGE_FLUX, ImageBuildVariant.IMAGE_TOOLKIT)
    return (ImageBuildVariant.TEXT,)


async def _resolve_dockerfile(repo: str, sha: str, token: str | None, variant: ImageBuildVariant) -> str:
    owner_repo = repo.removeprefix("https://github.com/").removesuffix(".git")
    headers = {"Accept": "application/vnd.github.raw+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=20) as client:
        for path in DOCKERFILES[variant]:
            response = await client.get(
                f"https://api.github.com/repos/{owner_repo}/contents/{path}", params={"ref": sha}, headers=headers
            )
            if response.status_code == 200:
                return path
    # Persist a deterministic row so a missing Dockerfile fails independently in Depot.
    return DOCKERFILES[variant][0]


async def create_tournament_build_batch(tournament_id: str, config) -> str:
    tournament = await get_tournament(tournament_id, config.psql_db)
    if tournament is None:
        raise ValueError(f"Tournament {tournament_id} does not exist")
    manifests = []
    for participant in await get_tournament_participants(tournament_id, config.psql_db):
        if not participant.training_repo or not participant.training_commit_hash:
            continue
        for variant in _variants(tournament.tournament_type):
            dockerfile = await _resolve_dockerfile(
                participant.training_repo,
                participant.training_commit_hash,
                participant.github_token,
                variant,
            )
            manifests.append(
                {
                    "hotkey": participant.hotkey,
                    "variant": variant.value,
                    "stable_build_id": f"{tournament_id}:{participant.hotkey}:{variant.value}",
                    "repo": participant.training_repo,
                    "commit_sha": participant.training_commit_hash,
                    "dockerfile_path": dockerfile,
                    "build_context": f"{participant.training_repo.removesuffix('.git')}.git#{participant.training_commit_hash}",
                    "platform": "linux/amd64",
                }
            )
    return await image_build_sql.create_batch_with_builds(tournament_id, manifests, config.psql_db)
