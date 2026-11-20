"""Paid places of a tournament the boss kept: each contender in a paid place (2nd, 3rd) against the boss on average rank.

When there is no crown the boss keeps 1st and the contenders take 2nd and 3rd by their standing. Each of those contenders is
compared with the boss, independently: over the rounds of 1-3 in which both have a rank (round_results.round_ranks_with_boss),
when the boss's average rank is strictly lower (better) than the contender's, the contender's share of the pool is paid like
the boss's own share (scoring.tournaments.get_boss_round_pair_weights); otherwise, level included, the contender is paid.

The comparison is computed once, when the tournament completes, and stored (tournament_boss_rank_comparisons) so the weight
setter reads it without recomputing standings. A contender that cannot be compared (no round ranks both it and the boss, or the
ranks cannot be computed) is paid.
"""

from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import BossRankComparison
from validator.tournament.models import TournamentRoundRanks
from validator.tournament.round_results import round_ranks_with_boss


logger = get_logger(__name__)


def compare_with_boss(ranks: TournamentRoundRanks, hotkey: str, final_position: int) -> BossRankComparison:
    """The contender's average rank against the boss's over the rounds that rank both; the share goes to the boss only when
    the boss's average is strictly lower. With no such round the contender is paid."""
    compared = [
        round_ranks for round_ranks in ranks.rounds if hotkey in round_ranks.ranks and EMISSION_BURN_HOTKEY in round_ranks.ranks
    ]
    if not compared:
        return BossRankComparison(hotkey=hotkey, final_position=final_position)
    contender_total = sum(round_ranks.ranks[hotkey] for round_ranks in compared)
    boss_total = sum(round_ranks.ranks[EMISSION_BURN_HOTKEY] for round_ranks in compared)
    return BossRankComparison(
        hotkey=hotkey,
        final_position=final_position,
        contender_average_rank=contender_total / len(compared),
        boss_average_rank=boss_total / len(compared),
        compared_round_numbers=[round_ranks.round_number for round_ranks in compared],
        # Same rounds on both sides, so comparing the integer totals is comparing the averages, without float rounding.
        share_to_boss=boss_total < contender_total,
    )


async def boss_rank_comparisons(
    tournament_id: str, second_place: str | None, third_place: str | None, psql_db: PSQLDB
) -> list[BossRankComparison]:
    """The comparison of each contender in 2nd and 3rd place (an empty place, or the boss's own placeholder, is skipped).

    Never raises: when the round ranks cannot be computed it logs a warning and returns nothing, so every contender is paid.
    """
    placed = [
        (position, hotkey)
        for position, hotkey in ((2, second_place), (3, third_place))
        if hotkey is not None and hotkey != EMISSION_BURN_HOTKEY
    ]
    if not placed:
        return []
    try:
        ranks = await round_ranks_with_boss(tournament_id, psql_db)
    except Exception as e:
        logger.warning(
            f"Tournament {tournament_id}: round ranks with the boss could not be computed ({type(e).__name__}: {e}); "
            f"places 2 and 3 are paid"
        )
        return []
    comparisons = [compare_with_boss(ranks, hotkey, position) for position, hotkey in placed]
    for comparison in comparisons:
        logger.info(f"Tournament {tournament_id} boss rank comparison: {comparison.describe()}")
    return comparisons
