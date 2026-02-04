import tempfile
from typing import Optional

import aiohttp
from git import Repo
from git.exc import GitCommandError

from validator.core.config import Config
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def create_github_repository(name: str, description: str, token: str, username: str) -> dict:
    """Create a new repository on GitHub."""
    url = f"https://api.github.com/orgs/{username}/repos"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GOD-Tournament-Winner-Reuploader",
    }
    data = {"name": name, "description": description, "auto_init": False}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status == 201:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to create repository: {response.status} - {error_text}")


async def repository_exists(name: str, token: str, username: str) -> bool:
    """Check if a repository exists."""
    url = f"https://api.github.com/repos/{username}/{name}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GOD-Tournament-Winner-Reuploader",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            return response.status == 200


async def update_repository_description(name: str, description: str, token: str, username: str) -> bool:
    url = f"https://api.github.com/repos/{username}/{name}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GOD-Tournament-Winner-Reuploader",
    }
    data = {"description": description}

    async with aiohttp.ClientSession() as session:
        async with session.patch(url, json=data, headers=headers) as response:
            if response.status == 200:
                logger.info(f"Successfully updated description for repository {name}")
                return True
            else:
                error_text = await response.text()
                logger.error(f"Failed to update repository description: {response.status} - {error_text}")
                return False


def clone_and_push_repository(repo_url: str, new_repo_url: str, github_token: str, commit_hash: Optional[str] = None) -> None:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Repo.clone_from(repo_url, temp_dir)

            if commit_hash:
                repo.git.fetch("--all")
                repo.git.checkout(commit_hash)

            for remote in repo.remotes:
                if remote.name == "origin":
                    repo.delete_remote("origin")

            if new_repo_url.startswith("https://github.com/"):
                org_repo = new_repo_url.replace("https://github.com/", "").replace(".git", "")
                new_repo_url_with_token = f"https://{github_token}@github.com/{org_repo}.git"
                logger.info(f"Using token-based URL for push: https://***@github.com/{org_repo}.git")
            else:
                new_repo_url_with_token = new_repo_url.replace("https://", f"https://{github_token}@")

            repo.create_remote("origin", new_repo_url_with_token)

            try:
                repo.git.branch("-D", "main")
            except GitCommandError:
                pass
            except Exception as e:
                logger.error(f"Error deleting main branch: {e}. Will continue anyway.")

            repo.git.checkout("--orphan", "main")

            repo.git.add(".")

            commit_message = f"Tournament winner repository - Commit: {commit_hash[:8] if commit_hash else 'latest'}"
            repo.git.commit("-m", commit_message)

            repo.git.config("--local", "user.name", "GOD Tournament Bot")
            repo.git.config("--local", "user.email", "tournament@god.ai")

            repo.git.push("origin", "main", "--force")

            logger.info(f"Successfully pushed to {new_repo_url} with only the specified commit")

    except GitCommandError as e:
        raise RuntimeError(f"Git operation failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")


async def upload_tournament_participant_repository(
    tournament_id: str,
    tournament_type: str,
    participant_hotkey: str,
    training_repo: str,
    commit_hash: str,
    config: Config,
    position: Optional[int] = None,
) -> Optional[str]:
    """Upload a tournament participant's repository to GitHub."""
    github_token = config.github_token
    github_username = config.github_username

    if not github_token:
        logger.warning("GitHub token not available, skipping repository upload")
        return None

    if not github_username:
        logger.warning("GitHub username not available, skipping repository upload")
        return None

    try:
        repo_name = f"god-{tournament_type}-{tournament_id}-position-{position}".replace("_", "-")
        description = f"G.O.D {tournament_type.title()} Tournament {position}{'st' if position == 1 else 'nd' if position == 2 else 'rd' if position == 3 else 'th'} Place - {tournament_id} - Participant: {participant_hotkey}"

        logger.info(f"Processing tournament participant repository: {training_repo}")
        logger.info(f"Generated name: {repo_name}")

        if await repository_exists(repo_name, github_token, github_username):
            logger.info(f"Repository {repo_name} already exists, will force push to it...")
            new_repo_url = f"https://github.com/{github_username}/{repo_name}.git"
            await update_repository_description(repo_name, description, github_token, github_username)
        else:
            logger.info(f"Creating repository: {repo_name}")
            new_repo = await create_github_repository(repo_name, description, github_token, github_username)
            new_repo_url = new_repo["clone_url"]

        logger.info(f"Cloning and pushing {training_repo}...")
        clone_and_push_repository(training_repo, new_repo_url, github_token, commit_hash)

        logger.info(f"Successfully re-uploaded {training_repo} to {new_repo_url}")

        return new_repo_url

    except Exception as e:
        logger.error(f"Error uploading tournament participant repository {training_repo}: {e}")
        return None
