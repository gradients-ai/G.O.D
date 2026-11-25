"""Forensic code review for a round-2 tournament entrant."""

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
from validator.infrastructure.challenger_dependency_review import materialize_dependencies
from validator.infrastructure.claude_sandbox import agent_env
from validator.infrastructure.claude_sandbox import confine_to_cwd_hooks
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
    """Extract the integrity JSON object from Claude's reply.

    Claude often prepends prose that itself contains braces (e.g. `{task_id}`),
    so we must not take the first `{` — locate the object that contains
    `"verdict"` (preferring the last match) and parse from there.
    """
    cleaned = re.sub(r"```(?:json)?", "", text)
    matches = list(re.finditer(r'\{\s*"verdict"\s*:', cleaned))
    if not matches:
        raise ValueError("Claude returned no JSON verdict")
    start = matches[-1].start()
    end = cleaned.rfind("}")
    if end <= start:
        raise ValueError("Claude returned no JSON verdict")
    value = json.loads(cleaned[start : end + 1])
    verdict = str(value.get("verdict", "")).lower()
    if verdict not in {"clean", "flagged", "needs_review"}:
        raise ValueError(f"Unknown integrity verdict: {verdict!r}")
    evidence = value.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    unresolved = value.get("unresolved_dependencies") or []
    if not isinstance(unresolved, list):
        unresolved = [str(unresolved)]
    return IntegrityVerdict(
        flagged=verdict == "flagged",
        needs_review=verdict == "needs_review",
        confidence=float(value.get("confidence", 0.0)),
        reason=str(value.get("reason", "")),
        evidence=[str(item) for item in evidence],
        unresolved_dependencies=[str(item) for item in unresolved],
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
        env=agent_env(),
        hooks=confine_to_cwd_hooks(cwd),
        system_prompt=config["system_prompt"],
    )
    result_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result_text = message.result or ""
    return _parse_verdict(result_text)


# Infra/access failures while fetching a package. These must not pause R2 by themselves —
# Claude already reviewed with whatever could be materialized, and a transient PyPI miss
# (or a garbage token like `https`) is not evidence of cheating.
_ACCESS_UNRESOLVED_MARKERS = (
    "pypi lookup failed",
    "download/extract failed",
    "artifact missing url",
    "clone not yet materialized",
)


def _is_access_unresolved(item: str) -> bool:
    lowered = item.lower()
    return any(marker in lowered for marker in _ACCESS_UNRESOLVED_MARKERS)


# Claude sometimes still returns needs_review because ai-toolkit / musubi / ComfyUI were
# not unpacked, even when first-party wiring is clean. That is not exploit evidence.
_KNOWN_TRAINER_FRAMEWORK = re.compile(
    r"\b(musubi(?:-tuner)?|ai-toolkit|comfyui|comfy[_-]?kitchen|diffusers|transformers)\b",
    re.IGNORECASE,
)
_INSPECT_GAP = re.compile(
    r"(not materialized|un-?inspectable|cannot be inspected|could not be inspected|"
    r"could not be certified|PACKAGE_FILE_INVENTORY)",
    re.IGNORECASE,
)
_EXPLOIT_SIGNAL = re.compile(
    r"(inject|bundled.{0,40}(data|dataset|row|sample)|fingerprint.{0,40}(lookup|select|match)|"
    r"score.?manip|external.?dataset|undeclared|pretrained.?adapter|eval.?leak|"
    r"benchmark.?answer|holdout.?leak|test.?set)",
    re.IGNORECASE | re.DOTALL,
)


def _demote_materialization_noise(verdict: IntegrityVerdict) -> IntegrityVerdict:
    """Drop needs_review that is only about missing common-trainer unpacks."""
    if not verdict.needs_review or verdict.flagged:
        return verdict
    text = " ".join([verdict.reason, *verdict.evidence, *verdict.unresolved_dependencies])
    if _EXPLOIT_SIGNAL.search(text):
        return verdict
    if not (_KNOWN_TRAINER_FRAMEWORK.search(text) and _INSPECT_GAP.search(text)):
        return verdict
    logger.warning(
        "Demoting needs_review that only cites unmaterialized common trainer deps "
        f"(reason={verdict.reason[:200]!r})"
    )
    verdict.needs_review = False
    note = "Demoted needs_review: missing common-trainer materialization is not exploit evidence"
    if not any("demoted needs_review" in item.lower() for item in verdict.evidence):
        verdict.evidence = [*verdict.evidence, note]
    return verdict


def _merge_materialization_into_verdict(verdict: IntegrityVerdict, unresolved: list[str]) -> IntegrityVerdict:
    """Attach unresolved deps; only policy/security misses force needs_review.

    Access failures (PyPI down, missing artifact, download errors) stay visible in
    the verdict but do not override a clean Claude result or pause round 2.
    Unpinned git deps and sha256 mismatches still fail closed.
    """
    if not unresolved:
        return verdict

    access = [item for item in unresolved if _is_access_unresolved(item)]
    policy = [item for item in unresolved if not _is_access_unresolved(item)]
    verdict.unresolved_dependencies = list(
        dict.fromkeys([*verdict.unresolved_dependencies, *unresolved])
    )

    if access and not any("access failure" in e.lower() for e in verdict.evidence):
        note = "Dependency access failures (non-blocking): " + "; ".join(access[:12])
        verdict.evidence = [*verdict.evidence, f"_manifests/unresolved.json - {note}"]
        logger.warning(f"Ignoring non-blocking dependency access failures: {access[:12]}")

    if not policy:
        return verdict
    if verdict.flagged:
        return verdict

    verdict.needs_review = True
    verdict.flagged = False
    extra = "Unresolved dependencies require manual review: " + "; ".join(policy[:12])
    if verdict.reason:
        verdict.reason = f"{verdict.reason.rstrip()} | {extra}"
    else:
        verdict.reason = extra
    if not any("require manual review" in e.lower() for e in verdict.evidence):
        verdict.evidence = [*verdict.evidence, f"_manifests/unresolved.json - {extra}"]
    return verdict


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

        materialization = await asyncio.to_thread(materialize_dependencies, repo, temp_root)
        logger.info(
            f"Materialized dependencies for {participant.hotkey}: "
            f"extracted={sum(1 for p in materialization.packages if p.status == 'extracted')} "
            f"unresolved={len(materialization.unresolved)} findings={len(materialization.findings)}"
        )

        verdict = await _run_claude(
            temp_root,
            tournament_type,
            participant.requested_datasets or [],
        )
        verdict = _merge_materialization_into_verdict(verdict, materialization.unresolved)
        verdict = _demote_materialization_noise(verdict)
        verdict.reason = _sanitize_reason(verdict.reason, {token} if token else set())
        verdict.evidence = [_sanitize_reason(item, {token} if token else set()) for item in verdict.evidence]
        status = (
            "FLAGGED"
            if verdict.flagged
            else ("NEEDS_REVIEW" if verdict.needs_review else "CLEAN")
        )
        logger.info(
            f"Completed challenger code review for {participant.hotkey}: "
            f"{status} (confidence={verdict.confidence:.2f})"
        )
        return verdict
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
