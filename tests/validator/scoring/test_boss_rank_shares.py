"""Paid places when the boss keeps the crown: each contender in 2nd or 3rd place is compared with the boss on average rank over
rounds 1-3, independently; a contender the boss strictly outranks has its share paid like the boss's own share.

Covers the comparison (boss_rank_shares), the per-type rank weights (get_boss_round_pair_weights) and the node weights set on
chain (get_node_weights_from_tournament_audit_data), including the pool totals.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import validator.scoring.constants as cts
import validator.tournament.boss_rank_shares as boss_rank_shares
from validator.scoring.tournaments import get_boss_round_pair_weights
from validator.scoring.weights import apply_tournament_weights
from validator.scoring.weights import get_node_weights_from_tournament_audit_data
from validator.tournament.boss_rank_shares import boss_rank_comparisons
from validator.tournament.boss_rank_shares import compare_with_boss
from validator.tournament.models import BossRankComparison
from validator.tournament.models import RoundRanks
from validator.tournament.models import TournamentAuditData
from validator.tournament.models import TournamentResultsWithWinners
from validator.tournament.models import TournamentRoundRanks


BOSS = cts.EMISSION_BURN_HOTKEY
CHAMPION = "5DefendingChampion"  # the boss's real hotkey (base_winner_hotkey)
SECOND = "5Second"
THIRD = "5Third"
OTHER = "5Other"
TID = "tourn"

# The rank-weight ladder over three paid places.
FIRST_SHARE, SECOND_SHARE, THIRD_SHARE = 16 / 21, 4 / 21, 1 / 21


def _ranks(*rounds: dict[str, int]) -> TournamentRoundRanks:
    return TournamentRoundRanks(
        tournament_id=TID,
        rounds=[
            RoundRanks(round_number=number, round_id=f"r{number}", task_ids=[f"t{number}"], ranks=ranks)
            for number, ranks in enumerate(rounds, start=1)
        ],
    )


def _comparisons(ranks: TournamentRoundRanks) -> list[BossRankComparison]:
    return [compare_with_boss(ranks, SECOND, 2), compare_with_boss(ranks, THIRD, 3)]


def _kept(comparisons: list[BossRankComparison], winner: str = BOSS, base: str | None = CHAMPION):
    return TournamentResultsWithWinners(
        tournament_id=TID,
        rounds=[],
        winner_hotkey=winner,
        base_winner_hotkey=base,
        final_positions={BOSS: 1, SECOND: 2, THIRD: 3, OTHER: 4} if winner == BOSS else {SECOND: 1, BOSS: 2, THIRD: 3},
        boss_rank_comparisons=comparisons,
    )


class TestTheComparison:
    def test_the_boss_strictly_better_sends_the_share_to_the_boss(self):
        comparison = compare_with_boss(_ranks({BOSS: 1, THIRD: 2}, {BOSS: 2, THIRD: 3}, {BOSS: 1, THIRD: 2}), THIRD, 3)

        assert comparison.share_to_boss
        assert comparison.boss_average_rank == pytest.approx(4 / 3)
        assert comparison.contender_average_rank == pytest.approx(7 / 3)
        assert comparison.compared_round_numbers == [1, 2, 3]

    def test_equal_averages_are_paid(self):
        comparison = compare_with_boss(_ranks({BOSS: 1, SECOND: 2}, {BOSS: 2, SECOND: 1}, {BOSS: 3, SECOND: 3}), SECOND, 2)

        assert not comparison.share_to_boss
        assert comparison.boss_average_rank == comparison.contender_average_rank == pytest.approx(2.0)

    def test_a_contender_better_than_the_boss_is_paid(self):
        ranks = _ranks({BOSS: 3, SECOND: 1}, {BOSS: 2, SECOND: 1}, {BOSS: 2, SECOND: 1})

        assert not compare_with_boss(ranks, SECOND, 2).share_to_boss

    def test_a_quant_contender_cut_after_round_two_is_compared_over_its_two_rounds(self):
        # Round 3 has the boss at rank 1 but the contender did not play it: the boss's average is over rounds 1-2 only.
        ranks = _ranks({BOSS: 4, THIRD: 2}, {BOSS: 2, THIRD: 3}, {BOSS: 1, SECOND: 2})

        comparison = compare_with_boss(ranks, THIRD, 3)

        assert comparison.compared_round_numbers == [1, 2]
        assert comparison.boss_average_rank == pytest.approx(3.0) and comparison.contender_average_rank == pytest.approx(2.5)
        assert not comparison.share_to_boss  # over all three rounds the boss (7/3) would have looked better

    def test_rounds_without_a_boss_rank_are_left_out(self):
        comparison = compare_with_boss(_ranks({SECOND: 1}, {BOSS: 1, SECOND: 2}, {SECOND: 1}), SECOND, 2)

        assert comparison.compared_round_numbers == [2] and comparison.share_to_boss

    def test_no_boss_rank_in_any_round_is_paid(self):
        # e.g. a tournament from before the boss played rounds 1-3
        comparison = compare_with_boss(_ranks({SECOND: 1, THIRD: 2}, {SECOND: 1}, {SECOND: 1}), SECOND, 2)

        assert not comparison.share_to_boss
        assert comparison.boss_average_rank is None and comparison.contender_average_rank is None
        assert comparison.compared_round_numbers == []

    async def test_ranks_that_cannot_be_computed_store_nothing_so_both_are_paid(self, monkeypatch):
        monkeypatch.setattr(boss_rank_shares, "round_ranks_with_boss", AsyncMock(side_effect=ValueError("no rounds")))

        assert await boss_rank_comparisons(TID, SECOND, THIRD, None) == []

    async def test_an_empty_place_and_the_boss_placeholder_are_skipped(self, monkeypatch):
        ranks = _ranks({BOSS: 1, SECOND: 2}, {BOSS: 1, SECOND: 2}, {BOSS: 1, SECOND: 2})
        monkeypatch.setattr(boss_rank_shares, "round_ranks_with_boss", AsyncMock(return_value=ranks))

        comparisons = await boss_rank_comparisons(TID, SECOND, None, None)

        assert [(c.hotkey, c.final_position, c.share_to_boss) for c in comparisons] == [(SECOND, 2, True)]
        assert await boss_rank_comparisons(TID, BOSS, None, None) == []


# Rounds 1-3 where the boss sits between the two contenders: it outranks 3rd (avg 2 vs 3) but not 2nd (avg 1 vs 2).
BOSS_BETWEEN = _ranks({SECOND: 1, BOSS: 2, THIRD: 3}, {SECOND: 1, BOSS: 2, THIRD: 3}, {SECOND: 1, BOSS: 2, THIRD: 3})
BOSS_ON_TOP = _ranks({BOSS: 1, SECOND: 2, THIRD: 3}, {BOSS: 1, SECOND: 2, THIRD: 3}, {BOSS: 1, SECOND: 2, THIRD: 3})
BOSS_LAST = _ranks({SECOND: 1, THIRD: 2, BOSS: 3}, {SECOND: 1, THIRD: 2, BOSS: 3}, {SECOND: 1, THIRD: 2, BOSS: 3})
BOSS_ABOVE_SECOND_ONLY = _ranks({THIRD: 1, BOSS: 2, SECOND: 3}, {THIRD: 1, BOSS: 2, SECOND: 3}, {THIRD: 1, BOSS: 2, SECOND: 3})
BOSS_LEVEL = _ranks({SECOND: 1, BOSS: 2, THIRD: 2}, {BOSS: 1, SECOND: 2, THIRD: 3}, {BOSS: 3, SECOND: 3, THIRD: 1})


class TestThePairWeights:
    @pytest.mark.parametrize(
        "ranks, expected",
        [
            (BOSS_BETWEEN, {CHAMPION: FIRST_SHARE + THIRD_SHARE, SECOND: SECOND_SHARE}),  # 2nd paid, 3rd to the boss
            (BOSS_ABOVE_SECOND_ONLY, {CHAMPION: FIRST_SHARE + SECOND_SHARE, THIRD: THIRD_SHARE}),  # 2nd to the boss, 3rd paid
            (BOSS_ON_TOP, {CHAMPION: 1.0}),  # both to the boss
            (BOSS_LAST, {CHAMPION: FIRST_SHARE, SECOND: SECOND_SHARE, THIRD: THIRD_SHARE}),  # both paid
            (BOSS_LEVEL, {CHAMPION: FIRST_SHARE, SECOND: SECOND_SHARE, THIRD: THIRD_SHARE}),  # equal averages: paid
        ],
    )
    def test_each_contender_is_decided_on_its_own(self, ranks, expected):
        weights = get_boss_round_pair_weights(_kept(_comparisons(ranks)))

        assert weights == {hotkey: pytest.approx(share) for hotkey, share in expected.items()}
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_a_crown_changes_nothing_whatever_is_stored(self):
        weights = get_boss_round_pair_weights(_kept(_comparisons(BOSS_ON_TOP), winner=SECOND))

        expected = {SECOND: FIRST_SHARE, CHAMPION: SECOND_SHARE, THIRD: THIRD_SHARE}
        assert weights == {hotkey: pytest.approx(share) for hotkey, share in expected.items()}

    def test_no_stored_comparison_pays_both(self):
        weights = get_boss_round_pair_weights(_kept([]))

        expected = {CHAMPION: FIRST_SHARE, SECOND: SECOND_SHARE, THIRD: THIRD_SHARE}
        assert weights == {hotkey: pytest.approx(share) for hotkey, share in expected.items()}

    def test_missing_ranks_pay_both(self):
        no_boss = _ranks({SECOND: 1, THIRD: 2}, {SECOND: 1, THIRD: 2}, {SECOND: 1})

        weights = get_boss_round_pair_weights(_kept(_comparisons(no_boss)))

        assert weights[SECOND] == pytest.approx(SECOND_SHARE) and weights[THIRD] == pytest.approx(THIRD_SHARE)

    def test_a_comparison_for_a_place_the_contender_no_longer_holds_is_ignored(self):
        stale = BossRankComparison(
            hotkey=OTHER, final_position=3, contender_average_rank=5, boss_average_rank=1, share_to_boss=True
        )

        weights = get_boss_round_pair_weights(_kept([stale]))

        assert weights[THIRD] == pytest.approx(THIRD_SHARE) and weights[CHAMPION] == pytest.approx(FIRST_SHARE)

    def test_a_comparison_that_cannot_be_read_pays_both(self):
        data = _kept([])
        data.boss_rank_comparisons = [None]  # a malformed entry must not break weight setting

        weights = get_boss_round_pair_weights(data)

        assert weights[SECOND] == pytest.approx(SECOND_SHARE) and weights[THIRD] == pytest.approx(THIRD_SHARE)

    def test_with_no_known_champion_the_redirected_share_goes_to_the_burn_hotkey(self):
        weights = get_boss_round_pair_weights(_kept(_comparisons(BOSS_BETWEEN), base=None))

        assert weights == {BOSS: pytest.approx(FIRST_SHARE + THIRD_SHARE), SECOND: pytest.approx(SECOND_SHARE)}


class TestTheWeightsSetOnChain:
    async def _node_weights(self, data: TournamentResultsWithWinners, nodes: list[str]) -> dict[str, float]:
        audit = TournamentAuditData(
            text_tournament_data=data,
            text_tournament_weight=0.5,
            quant_tournament_weight=0.25,
            image_tournament_weight=0.25,
            text_base_weight=0.5,
            quant_base_weight=0.25,
            image_base_weight=0.25,
        )
        metagraph = [MagicMock(hotkey=hotkey, node_id=index) for index, hotkey in enumerate(nodes)]
        with patch("validator.scoring.weights.fetch_nodes") as fetch:
            fetch._get_nodes_for_uid.return_value = metagraph
            result = await get_node_weights_from_tournament_audit_data(MagicMock(), 1, audit)
        return dict(zip(nodes, result.node_weights))

    async def test_the_redirected_share_reaches_the_champion_and_the_pool_is_unchanged(self):
        nodes = [CHAMPION, SECOND, THIRD, OTHER, BOSS]

        paid = await self._node_weights(_kept([]), nodes)
        redirected = await self._node_weights(_kept(_comparisons(BOSS_BETWEEN)), nodes)

        assert redirected[THIRD] == 0.0
        assert redirected[CHAMPION] == pytest.approx(paid[CHAMPION] + paid[THIRD])
        assert redirected[SECOND] == pytest.approx(paid[SECOND])
        text_pool = sum(redirected[hotkey] for hotkey in (CHAMPION, SECOND, THIRD, OTHER))
        assert text_pool == pytest.approx(0.5)
        # The quant and image pools have no tournament: they are undistributed and burned, as before.
        assert redirected[BOSS] == pytest.approx(paid[BOSS]) == pytest.approx(0.5)
        assert sum(redirected.values()) == pytest.approx(1.0)

    async def test_with_no_known_champion_the_boss_and_redirected_shares_are_burned(self):
        nodes = [SECOND, THIRD, OTHER, BOSS]

        weights = await self._node_weights(_kept(_comparisons(BOSS_BETWEEN), base=None), nodes)

        assert weights[SECOND] == pytest.approx(0.5 * SECOND_SHARE)
        assert weights[THIRD] == 0.0
        assert weights[BOSS] == pytest.approx(0.5 + 0.5 * (FIRST_SHARE + THIRD_SHARE))
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_the_burn_hotkeys_share_is_undistributed_not_added_to_its_node(self):
        node_weights = [0.0, 0.0]

        undistributed = apply_tournament_weights(
            {BOSS: 0.8, SECOND: 0.2}, {}, {BOSS: 0, SECOND: 1}, node_weights, 0.5, 0.0, 0.5, 0.0, BOSS, None
        )

        assert node_weights == [0.0, pytest.approx(0.1)]
        assert undistributed == pytest.approx(0.4)
