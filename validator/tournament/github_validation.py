import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import httpx

from core.git import build_authenticated_git_url
from core.git import sanitize_git_text
from core.logging import get_logger
from validator.tournament import constants as t_cst
from validator.tournament.models import GitHubOwnerRepo
from validator.tournament.models import RespondingNode


logger = get_logger(__name__)


async def validate_repo_obfuscation(
    repo_url: str, commit_hash: str | None = None, github_token: str | None = None
) -> bool:
    """
    Validate that a repository is not obfuscated using the obfuscation detection.

    Args:
        repo_url: The repository URL to validate
        commit_hash: Optional commit hash to validate instead of the default branch

    Returns:
        bool: True if repo is not obfuscated, False if obfuscated
    """
    try:
        clone_url = build_authenticated_git_url(repo_url, github_token)
        cmd = [t_cst.OBFUSCATION_DETECTION_PATH, "--repo", clone_url]
        if commit_hash:
            cmd += ["--commit", commit_hash]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        logger.info(f"Obfuscation detection output: {proc.stdout}")

        if proc.returncode == 0:
            logger.info(f"Repo {repo_url} is not obfuscated (exit code 0)")
            return True
        else:
            logger.warning(f"Repo {repo_url} is obfuscated (exit code {proc.returncode})")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Obfuscation detection timed out for repo {repo_url}")
        return False
    except Exception as e:
        logger.error(f"Obfuscation detection failed for repo {repo_url}: {str(e)}")
        return False

async def validate_repo_license(repo_url: str, github_token: str | None = None) -> bool:
    """
    Validate that a repository has verbatim LICENSE and NOTICE files matching the current repository.

    Args:
        repo_url: The repository URL to validate

    Returns:
        bool: True if repo has valid LICENSE and NOTICE files, False otherwise
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Cloning repository {repo_url} for license validation")
            clone_url = build_authenticated_git_url(repo_url, github_token)

            clone_proc = subprocess.run(
                ["git", "clone", clone_url, temp_dir],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if clone_proc.returncode != 0:
                sanitized_stderr = sanitize_git_text(clone_proc.stderr, github_token)
                logger.error(f"Failed to clone repository {repo_url}: {sanitized_stderr}")
                return False

            temp_path = Path(temp_dir)
            current_file_path = Path(__file__).resolve()
            repo_root = current_file_path.parent.parent.parent

            expected_license_path = repo_root / "LICENSE.md"
            if not expected_license_path.exists():
                expected_license_path = repo_root / "LICENSE"
                if not expected_license_path.exists():
                    logger.warning(
                        f"Expected LICENSE file not found in validator repository at "
                        f"{repo_root / 'LICENSE.md'} or {repo_root / 'LICENSE'}. "
                        f"Skipping license validation for {repo_url}"
                    )
                    return True

            expected_notice_path = None
            for notice_filename in ["NOTICE", "NOTICE.txt", "notice.txt", "Notice.txt", "notice", "Notice"]:
                potential_path = repo_root / notice_filename
                if potential_path.exists():
                    expected_notice_path = potential_path
                    break

            if not expected_notice_path:
                logger.warning(
                    f"Expected NOTICE file not found in validator repository at {repo_root} "
                    f"(checked NOTICE, NOTICE.txt, notice.txt, Notice.txt, notice, Notice). "
                    f"Skipping license validation for {repo_url}"
                )
                return True

            license_file_path = None
            for license_filename in ["LICENSE.md", "LICENSE", "license.md", "license", "License.md", "License"]:
                potential_path = temp_path / license_filename
                if potential_path.exists():
                    license_file_path = potential_path
                    break

            if not license_file_path:
                logger.warning(
                    f"License file not found in repository {repo_url} "
                    f"(checked LICENSE.md, LICENSE, license.md, license, License.md, License)"
                )
                return False

            license_content = license_file_path.read_text(encoding="utf-8")
            expected_license = expected_license_path.read_text(encoding="utf-8")

            expected_license_normalized = "\n".join(line.rstrip() for line in expected_license.splitlines())
            actual_license_normalized = "\n".join(line.rstrip() for line in license_content.splitlines())

            if expected_license_normalized != actual_license_normalized:
                logger.warning(f"LICENSE file content does not match verbatim for repository {repo_url}")
                return False

            notice_file_path = None
            for notice_filename in ["NOTICE", "NOTICE.txt", "notice.txt", "Notice.txt", "notice", "Notice"]:
                potential_path = temp_path / notice_filename
                if potential_path.exists():
                    notice_file_path = potential_path
                    break

            if not notice_file_path:
                logger.warning(
                    f"NOTICE file not found in repository {repo_url} "
                    f"(checked NOTICE, NOTICE.txt, notice.txt, Notice.txt, notice, Notice)"
                )
                return False

            notice_content = notice_file_path.read_text(encoding="utf-8")
            expected_notice = expected_notice_path.read_text(encoding="utf-8")

            expected_notice_normalized = "\n".join(line.rstrip() for line in expected_notice.splitlines())
            actual_notice_normalized = "\n".join(line.rstrip() for line in notice_content.splitlines())

            if expected_notice_normalized != actual_notice_normalized:
                logger.warning(f"NOTICE file content does not match verbatim for repository {repo_url}")
                return False

            logger.info(f"Repository {repo_url} passed license validation")
            return True

    except subprocess.TimeoutExpired:
        logger.error(f"Repository validation timed out for repo {repo_url}")
        return False
    except Exception as e:
        logger.error(f"Repository validation failed for repo {repo_url}: {str(e)}")
        return False

def parse_github_owner_repo(repo_url: str) -> GitHubOwnerRepo | None:
    path = urlparse(repo_url).path.strip("/")
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] and parts[1]:
        owner, repo_name = parts[0], parts[1].removesuffix(".git")
        return GitHubOwnerRepo(owner=owner, repo=repo_name)
    return None

async def validate_github_tokens(nodes: list[RespondingNode]) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        for node in nodes:
            token = node.training_repo_response.github_token
            if not token:
                continue

            parsed = parse_github_owner_repo(node.training_repo_response.github_repo)
            if not parsed:
                node.training_repo_response.github_token = None
                continue

            try:
                resp = await client.get(
                    f"https://api.github.com/repos/{parsed.owner}/{parsed.repo}",
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                )
                if resp.status_code != 200:
                    logger.warning(
                        f"Token for {node.node.hotkey} does not grant access to "
                        f"{parsed.owner}/{parsed.repo} (HTTP {resp.status_code}) — ignoring token"
                    )
                    node.training_repo_response.github_token = None
            except Exception as e:
                logger.warning(f"Token validation failed for {node.node.hotkey}: {e} — ignoring token")
                node.training_repo_response.github_token = None

def deduplicate_by_github_account(nodes: list[RespondingNode]) -> list[RespondingNode]:
    by_account: defaultdict[str, list[RespondingNode]] = defaultdict(list)
    no_account: list[RespondingNode] = []

    for node in nodes:
        parsed = parse_github_owner_repo(node.training_repo_response.github_repo)
        if parsed:
            by_account[parsed.owner.lower()].append(node)
        else:
            no_account.append(node)

    kept: list[RespondingNode] = list(no_account)
    for account, group in by_account.items():
        if len(group) == 1:
            kept.append(group[0])
            continue

        with_token = [n for n in group if n.training_repo_response.github_token]
        without_token = [n for n in group if not n.training_repo_response.github_token]

        if with_token:
            winner = with_token[0]
            rejected = with_token[1:] + without_token
        else:
            winner = without_token[0]
            rejected = without_token[1:]

        kept.append(winner)
        for r in rejected:
            logger.warning(
                f"Rejecting {r.node.hotkey} — duplicate GitHub account '{account}' "
                f"(kept {winner.node.hotkey})"
            )

    return kept


def deduplicate_by_ip_address(nodes: list[RespondingNode]) -> list[RespondingNode]:
    by_ip: defaultdict[str, list[RespondingNode]] = defaultdict(list)

    for node in nodes:
        by_ip[node.node.ip].append(node)

    kept: list[RespondingNode] = []
    for ip, group in by_ip.items():
        if len(group) == 1:
            kept.append(group[0])
            continue

        with_token = [n for n in group if n.training_repo_response.github_token]
        without_token = [n for n in group if not n.training_repo_response.github_token]

        if with_token:
            winner = with_token[0]
            rejected = with_token[1:] + without_token
        else:
            winner = without_token[0]
            rejected = without_token[1:]

        kept.append(winner)
        for r in rejected:
            logger.warning(
                f"Rejecting {r.node.hotkey} - duplicate IP address '{ip}' "
                f"(kept {winner.node.hotkey})"
            )

    return kept
