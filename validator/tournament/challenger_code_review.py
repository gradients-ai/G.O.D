"""Claude cheat-check for every non-boss miner entering round 2.

Runs once per entrant when round 1 completes, before round 2 is created.
A flagged repo pauses advancement until an operator agrees (eliminate) or
skips (let them play). The boss is never reviewed.

A review already stored on the tournament row is the old boss-round gate.
Finalization still honors that stored status and does not start a new review.
"""

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
from validator.db.sql.tournaments import get_tournament_participants
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.db.sql.tournaments import update_participant_code_review
from validator.infrastructure.challenger_code_review import review_challenger_code
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.tasks import calculate_miner_ranking_and_scores
from validator.tasks.details import upload_file_to_minio
from validator.tournament.models import GateDecision
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentRoundData
from validator.tournament.notifications import notify_challenger_code_review
from validator.tournament.task_results import get_task_results_for_ranking


logger = get_logger(__name__)

_RESOLVED_ALLOW = frozenset({"clean", "rejected"})
_UNRESOLVED_HOLD = frozenset({"pending", "error"})


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
            "# Round-2 challenger code review\n\n"
            f"- Tournament: `{tournament.tournament_id}`\n"
            f"- Challenger: `{challenger.hotkey}`\n\n"
            f"## Finding\n\n{reason}\n\n"
            "## Evidence\n\n" + "\n".join(f"- {item}" for item in evidence) + "\n"
        )
        object_name = (
            f"tournament-code-reviews/{tournament.tournament_type.value}/"
            f"{tournament.tournament_id}-{challenger.hotkey[:8]}-{int(time.time())}.md"
        )
        return await upload_file_to_minio(str(report), BUCKET_NAME, object_name)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


async def _record(tournament_id: str, hotkey: str, status: str, psql_db: PSQLDB) -> None:
    await update_participant_code_review(tournament_id, hotkey, status, psql_db)


async def evaluate_round2_code_reviews(
    tournament: TournamentData,
    winners: list[str],
    config: Config,
    psql_db: PSQLDB,
) -> GateDecision:
    """Review each non-boss round-2 entrant. Halt until every review is resolved.

    ``accepted`` reviews are returned in ``eliminate`` so those hotkeys never
    enter round 2. ``clean`` and ``rejected`` (operator skip) are allowed through.
    ``pending`` and ``error`` hold the tournament without re-running Claude.
    """
    cohort = [hotkey for hotkey in winners if hotkey != EMISSION_BURN_HOTKEY]
    logger.info(f"Round-2 code review: tournament={tournament.tournament_id}, entrants={len(cohort)}")
    if not cohort:
        return GateDecision(halt=False)

    participants = {
        participant.hotkey: participant for participant in await get_tournament_participants(tournament.tournament_id, psql_db)
    }
    eliminate: set[str] = set()
    halt = False

    for hotkey in cohort:
        participant = participants.get(hotkey)
        if participant is None:
            logger.error(f"Round-2 code review missing participant {hotkey} in {tournament.tournament_id}; holding")
            halt = True
            continue

        status = participant.code_review
        if status in _RESOLVED_ALLOW:
            logger.info(f"Round-2 code review for {hotkey} already {status}; allowing")
            continue
        if status == "accepted":
            logger.warning(f"Accepted round-2 code-review flag for {hotkey}; eliminating before round 2")
            eliminate.add(hotkey)
            continue
        if status in _UNRESOLVED_HOLD:
            logger.warning(f"Round-2 code review for {hotkey} is {status}; holding advancement")
            halt = True
            continue

        try:
            logger.info(f"Running Claude round-2 code review for {hotkey}")
            verdict = await review_challenger_code(participant, tournament.tournament_type.value)
        except Exception as exc:  # noqa: BLE001 - a failed review must hold advancement
            await _record(tournament.tournament_id, hotkey, "error", psql_db)
            logger.error(f"Round-2 code review failed for {hotkey}; holding advancement: {exc}")
            await notify_challenger_code_review(
                tournament.tournament_id,
                tournament.tournament_type.value,
                hotkey,
                f"Code review failed: {exc}",
                None,
                config.discord_url,
            )
            halt = True
            continue

        if not verdict.flagged:
            await _record(tournament.tournament_id, hotkey, "clean", psql_db)
            logger.info(f"Persisted round-2 code review for {hotkey} as clean")
            continue

        try:
            report_url = await _upload_report(tournament, participant, verdict.reason, verdict.evidence)
        except Exception:  # noqa: BLE001 - the Discord finding is still useful without an uploaded report
            report_url = None
        await _record(tournament.tournament_id, hotkey, "pending", psql_db)
        logger.warning(
            f"Persisted round-2 code review for {hotkey} as pending; holding advancement (report={report_url or 'upload failed'})"
        )
        await notify_challenger_code_review(
            tournament.tournament_id,
            tournament.tournament_type.value,
            hotkey,
            verdict.reason,
            report_url,
            config.discord_url,
        )
        halt = True

    return GateDecision(halt=halt, eliminate=eliminate)


async def _best_pre_boss_non_finalist(
    tournament: TournamentData,
    final_round: TournamentRoundData,
    challenger_hotkey: str,
    psql_db: PSQLDB,
) -> str | None:
    """Best non-finalist from the round before the boss, for a legacy disqualification."""
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


async def resolve_legacy_boss_code_review(
    tournament: TournamentData,
    final_round: TournamentRoundData,
    challenger: TournamentParticipant,
    psql_db: PSQLDB,
) -> ChallengerCodeReviewDecision:
    """Apply a boss-round review that was already stored on the tournament.

    Does not call Claude. A null status means the live check ran before round 2,
    so finalization continues.
    """
    status = tournament.code_review
    if status in (None, "clean", "rejected"):
        return ChallengerCodeReviewDecision(halt=False)
    if status == "accepted":
        replacement = await _best_pre_boss_non_finalist(tournament, final_round, challenger.hotkey, psql_db)
        logger.warning(
            f"Legacy boss-round code review accepted for {challenger.hotkey}; "
            f"disqualifying and promoting {replacement} to second place"
        )
        return ChallengerCodeReviewDecision(halt=False, disqualified=True, replacement_hotkey=replacement)
    logger.warning(f"Legacy boss-round code review for {tournament.tournament_id} is {status}; holding completion")
    return ChallengerCodeReviewDecision(halt=True)
