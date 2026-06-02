import aiohttp

from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_pairs
from validator.db.sql.tournaments import get_tournament_participant
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.app.config import Config
from validator.tournament.constants import DEFAULT_PARTICIPANT_COMMIT
from validator.tournament.constants import DEFAULT_PARTICIPANT_REPO
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentType


logger = get_logger(__name__)


async def _get_final_round_participants(completed_round: TournamentRoundData, psql_db: PSQLDB) -> tuple[str, str]:
    if completed_round.round_type != RoundType.KNOCKOUT:
        raise ValueError(f"Expected a knockout round, got {completed_round.round_type}")

    pairs = await get_tournament_pairs(completed_round.round_id, psql_db)
    if not pairs:
        raise ValueError(f"No pairs found for final round {completed_round.round_id}")

    pair = pairs[0]
    return pair.hotkey1, pair.hotkey2

async def get_challenger_participant_for_retained_boss(
    tournament: TournamentData,
    completed_round: TournamentRoundData,
    winners: list[str],
    psql_db: PSQLDB,
) -> TournamentParticipant | None:
    challenger_hotkey = next((hotkey for hotkey in winners if hotkey != EMISSION_BURN_HOTKEY), None)
    if not challenger_hotkey and completed_round.round_type == RoundType.KNOCKOUT:
        try:
            participant1, participant2 = await _get_final_round_participants(completed_round, psql_db)
            challenger_hotkey = participant2 if participant1 == EMISSION_BURN_HOTKEY else participant1
        except Exception as exc:
            logger.warning(f"Could not determine retained-boss challenger from final round participants: {exc}")

    if not challenger_hotkey:
        logger.warning("Could not determine retained-boss challenger; diff report will not include challenger repo")
        return None

    challenger = await get_tournament_participant(tournament.tournament_id, challenger_hotkey, psql_db)
    if not challenger or not challenger.training_repo:
        logger.warning(f"Challenger {challenger_hotkey} has no training repository in DB")
        return None
    return challenger

async def get_latest_commit_hash_from_github(repo_url: str) -> str | None:
    """Fetch the latest commit hash from a GitHub repository."""
    # Extract owner/repo from URL: https://github.com/owner/repo
    repo_path = repo_url.split("github.com/")[1].replace(".git", "")
    api_url = f"https://api.github.com/repos/{repo_path}/commits/main"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("sha", "")
                else:
                    logger.error(f"Failed to fetch commit hash from {repo_url}: HTTP {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error fetching commit hash from {repo_url}: {e}")
        return None

async def get_base_contestant(psql_db: PSQLDB, tournament_type: TournamentType, config: Config) -> TournamentParticipant | None:
    """Get a BASE contestant as the last tournament winner."""

    latest_winner = await get_latest_tournament_winner_participant(psql_db, tournament_type, config)
    if latest_winner:
        logger.info(f"Using latest tournament winner as BASE: {latest_winner.hotkey}")

        if latest_winner.backup_repo:
            logger.info(f"Previous winner has backup repo: {latest_winner.backup_repo}")
            commit_hash = await get_latest_commit_hash_from_github(latest_winner.backup_repo)
            if not commit_hash:
                logger.warning(f"Could not fetch commit hash for {latest_winner.backup_repo}, setting to None")

            return TournamentParticipant(
                tournament_id="",
                hotkey=EMISSION_BURN_HOTKEY,
                training_repo=latest_winner.backup_repo,
                training_commit_hash=commit_hash,
            )
        else:
            logger.warning("Could not determine tournament ID for uploaded repo, falling back to original training_repo")
            # Fallback to original training_repo if we can't determine the uploaded repo
            return TournamentParticipant(
                tournament_id="",
                hotkey=EMISSION_BURN_HOTKEY,
                training_repo=latest_winner.training_repo,
                training_commit_hash=latest_winner.training_commit_hash,
            )

    logger.info(
        f"No previous tournament winner found for type {tournament_type.value}, "
        f"using hardcoded base winner: {EMISSION_BURN_HOTKEY}"
    )

    hardcoded_participant = TournamentParticipant(
        tournament_id="",
        hotkey=EMISSION_BURN_HOTKEY,
        training_repo=DEFAULT_PARTICIPANT_REPO,
        training_commit_hash=DEFAULT_PARTICIPANT_COMMIT,
    )

    return hardcoded_participant

async def get_latest_tournament_winner_participant(
    psql_db: PSQLDB, tournament_type: TournamentType, config: Config
) -> TournamentParticipant | None:
    """Get the winner participant from the latest completed tournament of the given type."""
    latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
    if not latest_tournament:
        logger.warning(f"No completed tournaments found for type {tournament_type.value}")
        return None

    winner_hotkey = latest_tournament.winner_hotkey
    if not winner_hotkey:
        logger.warning(f"Tournament {latest_tournament.tournament_id} is completed but has no winner_hotkey stored")
        return None

    logger.info(f"Found latest tournament winner: {winner_hotkey}")
    winner_participant = await get_tournament_participant(latest_tournament.tournament_id, winner_hotkey, psql_db)

    # If we can't find the winner's participant record, check if they were the defending champion
    # who entered as EMISSION_BURN_HOTKEY
    if not winner_participant:
        logger.warning(
            f"Could not find participant record for winner {winner_hotkey} in tournament {latest_tournament.tournament_id}"
        )

        # If the winner was the base_winner (defending champion), try to get their record from EMISSION_BURN_HOTKEY
        if winner_hotkey == latest_tournament.base_winner_hotkey:
            logger.info(f"Winner {winner_hotkey} was the defending champion, checking EMISSION_BURN_HOTKEY participant record")
            emission_participant = await get_tournament_participant(
                latest_tournament.tournament_id, EMISSION_BURN_HOTKEY, psql_db
            )
            if emission_participant:
                # Use the EMISSION_BURN_HOTKEY participant's training info but with the actual winner's hotkey
                emission_participant.hotkey = winner_hotkey
                return emission_participant

        # If still no participant record found, return None to use default
        logger.warning(f"No participant record found for winner {winner_hotkey}, will use default")
        return None

    # If the participant is EMISSION_BURN_HOTKEY but we have a real winner, use the real winner's hotkey
    if winner_participant.hotkey == EMISSION_BURN_HOTKEY and latest_tournament.base_winner_hotkey:
        winner_participant.hotkey = latest_tournament.base_winner_hotkey

    return winner_participant
