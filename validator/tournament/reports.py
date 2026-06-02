from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.db.sql.tournaments import get_tournament_participant
from validator.db.sql.tournaments import update_tournament_diff_report
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import TournamentData
from validator.tournament.notifications import notify_tournament_completed
from validator.tournament.repo_diff_report import generate_and_upload_repo_diff_report


logger = get_logger(__name__)


async def generate_diff_report_for_result(
    tournament: TournamentData,
    challenger_repo: str | None,
    result_summary: str,
    psql_db: PSQLDB,
    challenger_commit_hash: str | None = None,
    challenger_github_token: str | None = None,
) -> str | None:
    if not challenger_repo:
        logger.warning("Challenger repository is missing; skipping repo diff report")
        return None

    previous_boss = await get_tournament_participant(tournament.tournament_id, EMISSION_BURN_HOTKEY, psql_db)
    previous_boss_repo = previous_boss.backup_repo or previous_boss.training_repo if previous_boss else None
    if not previous_boss_repo:
        logger.warning("Previous boss repository is missing; skipping repo diff report")
        return None
    previous_boss_commit_hash = None if previous_boss.backup_repo else previous_boss.training_commit_hash
    previous_boss_github_token = None if previous_boss.backup_repo else previous_boss.github_token

    report_url = await generate_and_upload_repo_diff_report(
        tournament_id=tournament.tournament_id,
        tournament_type=tournament.tournament_type.value,
        challenger_repo_url=challenger_repo,
        previous_boss_repo_url=previous_boss_repo,
        result_summary=result_summary,
        challenger_commit_hash=challenger_commit_hash,
        challenger_github_token=challenger_github_token,
        previous_boss_commit_hash=previous_boss_commit_hash,
        previous_boss_github_token=previous_boss_github_token,
    )
    if report_url:
        await update_tournament_diff_report(tournament.tournament_id, report_url, psql_db)
    return report_url

async def generate_diff_report_and_notify_tournament_completed(
    tournament: TournamentData,
    challenger_repo: str | None,
    result_summary: str,
    winner: str,
    discord_url: str,
    psql_db: PSQLDB,
    challenger_commit_hash: str | None = None,
    challenger_github_token: str | None = None,
) -> None:
    diff_report = None
    try:
        diff_report = await generate_diff_report_for_result(
            tournament,
            challenger_repo,
            result_summary,
            psql_db,
            challenger_commit_hash=challenger_commit_hash,
            challenger_github_token=challenger_github_token,
        )
    except Exception as exc:
        logger.error(f"Failed to generate tournament diff report: {exc}", exc_info=True)

    try:
        await notify_tournament_completed(
            tournament.tournament_id, tournament.tournament_type.value, winner, discord_url, diff_report
        )
    except Exception as exc:
        logger.error(f"Failed to notify tournament completion: {exc}", exc_info=True)
