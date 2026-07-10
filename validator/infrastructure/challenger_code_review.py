"""Forensic code review for the boss-round challenger."""

import asyncio
import json
import os
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import validator.tournament.constants as t_cst
from core.logging import get_logger
from validator.infrastructure.repo_dedup import _sanitize_reason
from validator.infrastructure.repo_dedup import _snapshot_god_source
from validator.tournament.models import IntegrityVerdict
from validator.tournament.models import TournamentParticipant
from validator.tournament.repo_diff_report import _clone_repo


CONFIG_PATH = Path(__file__).with_name("challenger_code_review_config.json")
logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open() as handle:
        return json.load(handle)


def _parse_verdict(text: str) -> IntegrityVerdict:
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Claude returned no JSON verdict")
    value = json.loads(cleaned[start : end + 1])
    verdict = str(value.get("verdict", "")).lower()
    if verdict not in {"clean", "flagged"}:
        raise ValueError(f"Unknown integrity verdict: {verdict!r}")
    evidence = value.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    return IntegrityVerdict(
        flagged=verdict == "flagged",
        confidence=float(value.get("confidence", 0.0)),
        reason=str(value.get("reason", "")),
        evidence=[str(item) for item in evidence],
    )


def _import_claude_sdk():
    try:
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk import ResultMessage
        from claude_agent_sdk import query
    except ImportError as exc:
        raise RuntimeError("claude-agent-sdk is required for challenger code reviews") from exc
    return ClaudeAgentOptions, ResultMessage, query


async def _run_claude(cwd: Path, tournament_type: str, requested_datasets: list[str]) -> IntegrityVerdict:
    ClaudeAgentOptions, ResultMessage, query = _import_claude_sdk()
    config = _load_config()
    prompt = config["user_prompt_template"].format(
        tournament_type=tournament_type,
        requested_datasets=", ".join(requested_datasets) if requested_datasets else "none",
    )
    options = ClaudeAgentOptions(
        cwd=str(cwd),
        model=t_cst.TOURN_DEDUP_CLAUDE_MODEL,
        max_turns=t_cst.TOURN_DEDUP_CLAUDE_MAX_TURNS,
        max_budget_usd=t_cst.TOURN_DEDUP_CLAUDE_MAX_BUDGET_USD,
        permission_mode="dontAsk",
        allowed_tools=["Read", "Glob", "Grep"],
        disallowed_tools=["Write", "Edit", "Bash"],
        setting_sources=[],
        system_prompt=config["system_prompt"],
    )
    result_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result_text = message.result or ""
    return _parse_verdict(result_text)


async def review_challenger_code(
    participant: TournamentParticipant, tournament_type: str
) -> IntegrityVerdict:
    if not participant.training_repo:
        raise RuntimeError("challenger has no training repository")

    temp_root = Path(tempfile.mkdtemp(prefix="challenger-code-review-"))
    token = participant.github_token
    try:
        logger.info(f"Starting challenger code review for {participant.hotkey}")
        repo = temp_root / "challenger"
        await asyncio.to_thread(
            _clone_repo,
            participant.training_repo,
            repo,
            participant.training_commit_hash,
            token,
        )
        await asyncio.to_thread(shutil.rmtree, repo / ".git", ignore_errors=True)

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        if not await asyncio.to_thread(_snapshot_god_source, temp_root / "_god_source"):
            raise RuntimeError("could not snapshot validator source for the code review")
        verdict = await _run_claude(
            temp_root,
            tournament_type,
            participant.requested_datasets or [],
        )
        verdict.reason = _sanitize_reason(verdict.reason, {token} if token else set())
        verdict.evidence = [_sanitize_reason(item, {token} if token else set()) for item in verdict.evidence]
        logger.info(
            f"Completed challenger code review for {participant.hotkey}: "
            f"{'FLAGGED' if verdict.flagged else 'CLEAN'} (confidence={verdict.confidence:.2f})"
        )
        return verdict
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
