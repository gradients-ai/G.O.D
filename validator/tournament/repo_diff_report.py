import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import validator.tournament.constants as t_cst
from core.constants.credentials import BUCKET_NAME
from core.git import build_authenticated_git_url
from core.git import sanitize_git_text
from core.logging import get_logger
from validator.tasks.details import upload_file_to_minio


logger = get_logger(__name__)

CONFIG_PATH = Path(__file__).with_name("repo_diff_report_config.json")


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open() as handle:
        return json.load(handle)


_CONFIG = _load_config()
EXCLUDED_PARTS = set(_CONFIG["excluded_parts"])
EXCLUDED_SUFFIXES = set(_CONFIG["excluded_suffixes"])
EXCLUDED_NAMES = set(_CONFIG["excluded_names"])


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _git_head(repo: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()


def _clone_repo(repo_url: str, dest: Path, commit_hash: str | None = None, github_token: str | None = None) -> str:
    logger.info(f"Cloning repository for diff report: {repo_url}")
    clone_url = build_authenticated_git_url(repo_url, github_token)
    try:
        clone_cmd = ["git", "clone", clone_url, str(dest)]
        if not commit_hash:
            clone_cmd[2:2] = ["--depth", "1"]
        _run(clone_cmd)
        if commit_hash:
            _run(["git", "checkout", commit_hash], cwd=dest)
    except Exception as exc:
        sanitized_error = sanitize_git_text(str(exc), github_token)
        raise RuntimeError(f"Failed to clone repository for diff report: {sanitized_error}") from exc
    head = _git_head(dest)
    logger.info(f"Cloned {repo_url} at {head}")
    return head


def _safe_rel(path: Path) -> bool:
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and not any(part in EXCLUDED_PARTS for part in path.parts)
        and path.name not in EXCLUDED_NAMES
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def _is_text(path: Path) -> bool:
    try:
        return b"\x00" not in path.read_bytes()[:4096]
    except OSError:
        return False


def _collect_files(root: Path) -> set[str]:
    files = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if _safe_rel(rel) and _is_text(path):
            files.add(str(rel))
    return files


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _changed_text_files(current_repo: Path, previous_repo: Path) -> list[tuple[str, str]]:
    current_files = _collect_files(current_repo)
    previous_files = _collect_files(previous_repo)
    rows: list[tuple[str, str]] = []

    for rel in sorted(current_files | previous_files):
        in_current = rel in current_files
        in_previous = rel in previous_files
        if in_current and not in_previous:
            rows.append(("A", rel))
        elif in_previous and not in_current:
            rows.append(("D", rel))
        elif _digest(current_repo / rel) != _digest(previous_repo / rel):
            rows.append(("M", rel))

    return rows


def _write_diff_focus(
    path: Path,
    challenger_repo_url: str,
    previous_repo_url: str,
    challenger_repo_path: Path,
    previous_repo_path: Path,
    challenger_head: str,
    previous_head: str,
    rows: list[tuple[str, str]],
) -> None:
    included = rows[:t_cst.CLAUDE_REPO_DIFF_MAX_FOCUS_FILES]
    omitted = max(0, len(rows) - len(included))
    changed_files = "\n".join(f"- {status} `{rel}`" for status, rel in included)
    if changed_files:
        changed_files += "\n"
    else:
        changed_files = "No changed text files detected.\n"
    omitted_files_note = f"\nOmitted {omitted} additional changed files after the first {len(included)}.\n" if omitted else ""
    path.write_text(
        _CONFIG["diff_focus_template"].format(
            challenger_repo_url=challenger_repo_url,
            challenger_repo_path=challenger_repo_path,
            challenger_head=challenger_head,
            previous_repo_url=previous_repo_url,
            previous_repo_path=previous_repo_path,
            previous_head=previous_head,
            changed_files=changed_files,
            omitted_files_note=omitted_files_note,
        )
    )


def _write_name_status_diff(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("\n".join(f"{status}\t{rel}" for status, rel in rows) + ("\n" if rows else ""))


def _build_prompt(challenger_repo_url: str, previous_repo_url: str, result_summary: str, diff_focus: str) -> str:
    return _CONFIG["report_prompt_template"].format(
        challenger_repo_url=challenger_repo_url,
        previous_repo_url=previous_repo_url,
        result_summary=result_summary,
        diff_focus=diff_focus,
    )


def _clean_report(text: str) -> str:
    stripped = text.strip()
    heading_index = stripped.find("#")
    if heading_index > 0:
        stripped = stripped[heading_index:].lstrip()
    return stripped + "\n"


def _import_claude_sdk():
    try:
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk import ResultMessage
        from claude_agent_sdk import query
    except ImportError as exc:
        raise RuntimeError("claude-agent-sdk is required for repo diff reports") from exc
    return ClaudeAgentOptions, ResultMessage, query


async def _run_claude_report(prompt: str, cwd: Path) -> str:
    ClaudeAgentOptions, ResultMessage, query = _import_claude_sdk()
    options = ClaudeAgentOptions(
        cwd=cwd,
        model=t_cst.CLAUDE_REPO_DIFF_MODEL,
        max_turns=t_cst.CLAUDE_REPO_DIFF_MAX_TURNS,
        max_budget_usd=t_cst.CLAUDE_REPO_DIFF_MAX_BUDGET_USD,
        permission_mode="dontAsk",
        allowed_tools=["Read", "Glob", "Grep"],
        disallowed_tools=["Write", "Edit", "Bash"],
        setting_sources=[],
        system_prompt=_CONFIG["system_prompt"],
    )

    final_result = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            final_result = message.result or ""

    if not final_result:
        raise RuntimeError("Claude Agent SDK returned an empty repo diff report")
    return _clean_report(final_result)


async def generate_and_upload_repo_diff_report(
    tournament_id: str,
    tournament_type: str,
    challenger_repo_url: str,
    previous_boss_repo_url: str,
    result_summary: str,
    challenger_commit_hash: str | None = None,
    challenger_github_token: str | None = None,
    previous_boss_commit_hash: str | None = None,
    previous_boss_github_token: str | None = None,
) -> str | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY is not set; skipping repo diff report")
        return None
    if not BUCKET_NAME:
        logger.warning("S3_BUCKET_NAME is not set; skipping repo diff report upload")
        return None

    temp_root = Path(tempfile.mkdtemp(prefix=f"repo-diff-{tournament_id}-"))
    try:
        current_repo = temp_root / "challenger"
        previous_repo = temp_root / "previous_boss"
        current_head = await asyncio.to_thread(
            _clone_repo, challenger_repo_url, current_repo, challenger_commit_hash, challenger_github_token
        )
        previous_head = await asyncio.to_thread(
            _clone_repo, previous_boss_repo_url, previous_repo, previous_boss_commit_hash, previous_boss_github_token
        )

        rows = await asyncio.to_thread(_changed_text_files, current_repo, previous_repo)
        _write_name_status_diff(temp_root / "changed_files.diff", rows)
        diff_focus_path = temp_root / "diff_focus.md"
        _write_diff_focus(
            diff_focus_path,
            challenger_repo_url,
            previous_boss_repo_url,
            current_repo,
            previous_repo,
            current_head,
            previous_head,
            rows,
        )

        prompt = _build_prompt(challenger_repo_url, previous_boss_repo_url, result_summary, diff_focus_path.read_text())
        report = await _run_claude_report(prompt, temp_root)

        report_path = temp_root / "repo_diff_report.md"
        report_path.write_text(report)
        object_name = f"tournament-diff-reports/{tournament_type}/{tournament_id}-{int(time.time())}.md"
        report_url = await upload_file_to_minio(str(report_path), BUCKET_NAME, object_name)
        if report_url:
            logger.info(f"Uploaded repo diff report for tournament {tournament_id}: {report_url}")
        else:
            logger.warning(f"Failed to upload repo diff report for tournament {tournament_id}")
        return report_url
    except Exception as exc:
        logger.error(f"Failed to generate repo diff report for tournament {tournament_id}: {exc}", exc_info=True)
        return None
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
