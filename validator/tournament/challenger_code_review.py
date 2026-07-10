"""Claude code-review gate for the boss-round challenger."""

import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from core.constants.credentials import BUCKET_NAME
from core.logging import get_logger
from validator.app.config import Config
from validator.db.database import PSQLDB
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_groups
from validator.db.sql.tournaments import get_tournament_pairs
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.db.sql.tournaments import update_tournament_code_review
from validator.infrastructure.challenger_code_review import review_challenger_code
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.tasks import calculate_miner_ranking_and_scores
from validator.tasks.details import upload_file_to_minio
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentRoundData
from validator.tournament.notifications import notify_challenger_code_review
from validator.tournament.task_results import get_task_results_for_ranking


logger = get_logger(__name__)


@dataclass(frozen=True)
class ChallengerCodeReviewDecision:
    halt: bool
    disqualified: bool = False
    replacement_hotkey: str | None = None


async def _upload_report(
    tournament: TournamentData,
    challenger: TournamentParticipant,
    reason: str,
    evidence: list[str],
) -> str | None:
    directory = Path(tempfile.mkdtemp(prefix="challenger-code-review-"))
    try:
        report = directory / "report.md"
        report.write_text(
            "# Boss-round challenger code review\n\n"
            f"- Tournament: `{tournament.tournament_id}`\n"
            f"- Challenger: `{challenger.hotkey}`\n\n"
            f"## Finding\n\n{reason}\n\n"
            "## Evidence\n\n"
            + "\n".join(f"- {item}" for item in evidence)
            + "\n"
        )
        object_name = (
            f"tournament-code-reviews/{tournament.tournament_type.value}/"
            f"{tournament.tournament_id}-{int(time.time())}.md"
        )
        return await upload_file_to_minio(str(report), BUCKET_NAME, object_name)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


async def _best_pre_boss_non_finalist(
    tournament: TournamentData,
    final_round: TournamentRoundData,
    challenger_hotkey: str,
    psql_db: PSQLDB,
) -> str | None:
    rounds = await get_tournament_rounds(tournament.tournament_id, psql_db)
    pre_boss = next((item for item in rounds if item.round_number == final_round.round_number - 1), None)
    if not pre_boss:
        return None

    competitors: set[str] = set()
    for pair in await get_tournament_pairs(pre_boss.round_id, psql_db):
        competitors.update((pair.hotkey1, pair.hotkey2))
    for group in await get_tournament_groups(pre_boss.round_id, psql_db):
        members = await get_tournament_group_members(group.group_id, psql_db)
        competitors.update(member.hotkey for member in members)
    competitors -= {EMISSION_BURN_HOTKEY, challenger_hotkey}

    ranks = {hotkey: [] for hotkey in competitors}
    for task in await get_tournament_tasks(pre_boss.round_id, psql_db):
        results = await get_task_results_for_ranking(task.task_id, psql_db)
        for rank, result in enumerate(calculate_miner_ranking_and_scores(results), start=1):
            if result.hotkey in ranks:
                ranks[result.hotkey].append(rank)

    ordered = sorted(
        competitors,
        key=lambda hotkey: (
            not ranks[hotkey],
            sum(ranks[hotkey]) / len(ranks[hotkey]) if ranks[hotkey] else float("inf"),
            hotkey,
        ),
    )
    return ordered[0] if ordered else None


async def evaluate_challenger_code_review(
    tournament: TournamentData,
    final_round: TournamentRoundData,
    challenger: TournamentParticipant,
    config: Config,
    psql_db: PSQLDB,
) -> ChallengerCodeReviewDecision:
    """Run once on the boss-round challenger, immediately before final repository upload."""
    status = tournament.code_review
    logger.info(
        f"Challenger code-review gate: tournament={tournament.tournament_id}, "
        f"challenger={challenger.hotkey}, status={status or 'not_reviewed'}"
    )
    if status in ("clean", "rejected"):
        logger.info(f"Challenger code review resolved as {status}; allowing finalization")
        return ChallengerCodeReviewDecision(halt=False)
    if status == "accepted":
        replacement = await _best_pre_boss_non_finalist(
            tournament, final_round, challenger.hotkey, psql_db
        )
        logger.warning(
            f"Accepted challenger code-review flag for {challenger.hotkey}; "
            f"disqualifying and promoting {replacement} to second place"
        )
        return ChallengerCodeReviewDecision(halt=False, disqualified=True, replacement_hotkey=replacement)
    if status in ("pending", "error"):
        logger.warning(f"Challenger code review is {status}; blocking all tournament finalization")
        return ChallengerCodeReviewDecision(halt=True)

    try:
        logger.info("No challenger code review exists; running Claude before any finalization or repository upload")
        verdict = await review_challenger_code(challenger, tournament.tournament_type.value)
    except Exception as exc:  # noqa: BLE001 - errors must hold completion
        await update_tournament_code_review(tournament.tournament_id, "error", psql_db)
        logger.error(f"Challenger code review failed; blocking tournament finalization: {exc}")
        await notify_challenger_code_review(
            tournament.tournament_id,
            tournament.tournament_type.value,
            challenger.hotkey,
            f"Code review failed: {exc}",
            None,
            config.discord_url,
        )
        return ChallengerCodeReviewDecision(halt=True)

    if not verdict.flagged:
        await update_tournament_code_review(tournament.tournament_id, "clean", psql_db)
        logger.info("Persisted challenger code review as clean; finalization may now continue")
        return ChallengerCodeReviewDecision(halt=False)

    try:
        report_url = await _upload_report(tournament, challenger, verdict.reason, verdict.evidence)
    except Exception:  # noqa: BLE001 - the Discord finding is still useful without an uploaded report
        report_url = None
    await update_tournament_code_review(tournament.tournament_id, "pending", psql_db)
    logger.warning(
        f"Persisted challenger code review as pending; blocking finalization until agree/skip "
        f"(report={report_url or 'upload failed'})"
    )
    await notify_challenger_code_review(
        tournament.tournament_id,
        tournament.tournament_type.value,
        challenger.hotkey,
        verdict.reason,
        report_url,
        config.discord_url,
    )
    return ChallengerCodeReviewDecision(halt=True)
