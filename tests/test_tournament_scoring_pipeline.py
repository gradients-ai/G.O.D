"""Tests for the tournament scoring pipeline: PvP/MCTS results → pairwise → points → weights.

Exercises the full chain that determines tournament outcomes and emission weights.
"""

import pytest

from core.constants import EnvironmentName
from core.models.pvp_models import PvPEnvironmentResult
from core.models.pvp_models import PvPEvalMetadata
from core.models.pvp_models import PvPGroupResults
from core.models.pvp_models import PvPPairResult
from core.models.tournament_models import EnvironmentWeight
from core.models.tournament_models import GroupStagePoints
from core.models.tournament_models import PairwiseOutcome
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentRoundResult
from core.models.tournament_models import TournamentScore
from core.models.tournament_models import TournamentTaskScore
from core.models.tournament_models import TournamentType
from validator.evaluation.tournament_scoring import accumulate_points
from validator.evaluation.tournament_scoring import compute_pvp_tournament_points
from validator.evaluation.tournament_scoring import exponential_decline_mapping
from validator.evaluation.tournament_scoring import individual_scores_to_pairwise
from validator.evaluation.tournament_scoring import pvp_results_to_pairwise
from validator.evaluation.tournament_scoring import tournament_scores_to_weights
from validator.evaluation.tournament_scoring import calculate_tournament_type_scores_from_data
from validator.tournament.utils import determine_boss_round_winner
from validator.tournament.utils import get_real_winner_hotkey
import validator.core.constants as cts


# --- Fixtures ---


ALICE = "5GAlice"
BOB = "5GBob"
CAROL = "5GCarol"

ENV_DICE = EnvironmentName.LIARS_DICE
ENV_POKER = EnvironmentName.LEDUC_POKER
ENV_RUMMY = EnvironmentName.GIN_RUMMY


def _make_group_results(
    hotkeys: list[str],
    pair_results: list[PvPPairResult],
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
) -> PvPGroupResults:
    return PvPGroupResults(
        base_model=base_model,
        hotkeys=hotkeys,
        pair_results=pair_results,
        metadata=PvPEvalMetadata(seed=42, temperature=0.0),
    )


def _make_pair_result(
    hotkey_a: str,
    hotkey_b: str,
    envs: dict[EnvironmentName, tuple[int, int, int]],
) -> PvPPairResult:
    """Build a PvPPairResult. envs maps env → (a_wins, b_wins, draws)."""
    results = {}
    for env_name, (a_wins, b_wins, draws) in envs.items():
        results[env_name] = PvPEnvironmentResult(
            model_a_wins=a_wins,
            model_b_wins=b_wins,
            draws=draws,
            total_games=a_wins + b_wins + draws,
        )
    return PvPPairResult(hotkey_a=hotkey_a, hotkey_b=hotkey_b, results=results)


# --- 1a: pvp_results_to_pairwise ---


class TestPvpResultsToPairwise:
    def test_clear_winner_per_env(self):
        """A beats B in dice, B beats A in poker → 2 outcomes with correct winners."""
        pair = _make_pair_result(ALICE, BOB, {
            ENV_DICE: (8, 2, 0),
            ENV_POKER: (3, 7, 0),
        })
        group = _make_group_results([ALICE, BOB], [pair])
        outcomes = pvp_results_to_pairwise(group)

        assert len(outcomes) == 2
        dice_outcome = next(o for o in outcomes if o.environment == ENV_DICE)
        poker_outcome = next(o for o in outcomes if o.environment == ENV_POKER)
        assert dice_outcome.winner == ALICE
        assert poker_outcome.winner == BOB

    def test_draw_when_equal_wins(self):
        """Equal wins → winner is None (draw)."""
        pair = _make_pair_result(ALICE, BOB, {
            ENV_DICE: (5, 5, 0),
        })
        group = _make_group_results([ALICE, BOB], [pair])
        outcomes = pvp_results_to_pairwise(group)

        assert len(outcomes) == 1
        assert outcomes[0].winner is None

    def test_three_player_round_robin(self):
        """3 players, 2 envs → 6 outcomes (C(3,2) pairs × 2 envs)."""
        pairs = [
            _make_pair_result(ALICE, BOB, {ENV_DICE: (6, 4, 0), ENV_POKER: (3, 7, 0)}),
            _make_pair_result(ALICE, CAROL, {ENV_DICE: (7, 3, 0), ENV_POKER: (8, 2, 0)}),
            _make_pair_result(BOB, CAROL, {ENV_DICE: (5, 5, 0), ENV_POKER: (6, 4, 0)}),
        ]
        group = _make_group_results([ALICE, BOB, CAROL], pairs)
        outcomes = pvp_results_to_pairwise(group)

        assert len(outcomes) == 6  # 3 pairs × 2 envs

        # Alice beats Bob in dice, Bob beats Alice in poker
        ab_dice = next(o for o in outcomes if o.hotkey_a == ALICE and o.hotkey_b == BOB and o.environment == ENV_DICE)
        ab_poker = next(o for o in outcomes if o.hotkey_a == ALICE and o.hotkey_b == BOB and o.environment == ENV_POKER)
        assert ab_dice.winner == ALICE
        assert ab_poker.winner == BOB

        # Bob vs Carol in dice is a draw
        bc_dice = next(o for o in outcomes if o.hotkey_a == BOB and o.hotkey_b == CAROL and o.environment == ENV_DICE)
        assert bc_dice.winner is None


# --- 1b: accumulate_points (no weights) ---


class TestAccumulatePoints:
    def test_win_loss_draw_points(self):
        """Winner gets 3, loser gets 0, draws get 1 each."""
        outcomes = [
            PairwiseOutcome(hotkey_a=ALICE, hotkey_b=BOB, environment=ENV_DICE, winner=ALICE),
        ]
        standings = accumulate_points(outcomes, [ALICE, BOB])

        alice_pts = next(s for s in standings if s.hotkey == ALICE).points
        bob_pts = next(s for s in standings if s.hotkey == BOB).points
        assert alice_pts == cts.PVP_ENV_WIN_POINTS  # 3
        assert bob_pts == 0.0

    def test_draw_gives_one_point_each(self):
        outcomes = [
            PairwiseOutcome(hotkey_a=ALICE, hotkey_b=BOB, environment=ENV_DICE, winner=None),
        ]
        standings = accumulate_points(outcomes, [ALICE, BOB])

        alice_pts = next(s for s in standings if s.hotkey == ALICE).points
        bob_pts = next(s for s in standings if s.hotkey == BOB).points
        assert alice_pts == cts.PVP_ENV_DRAW_POINTS  # 1
        assert bob_pts == cts.PVP_ENV_DRAW_POINTS

    def test_three_player_accumulation(self):
        """Alice wins 2 envs, Bob wins 1, Carol draws 1 — verify total points."""
        outcomes = [
            # Alice vs Bob: Alice wins dice, Bob wins poker
            PairwiseOutcome(hotkey_a=ALICE, hotkey_b=BOB, environment=ENV_DICE, winner=ALICE),
            PairwiseOutcome(hotkey_a=ALICE, hotkey_b=BOB, environment=ENV_POKER, winner=BOB),
            # Alice vs Carol: Alice wins both
            PairwiseOutcome(hotkey_a=ALICE, hotkey_b=CAROL, environment=ENV_DICE, winner=ALICE),
            PairwiseOutcome(hotkey_a=ALICE, hotkey_b=CAROL, environment=ENV_POKER, winner=ALICE),
            # Bob vs Carol: draw in dice, Bob wins poker
            PairwiseOutcome(hotkey_a=BOB, hotkey_b=CAROL, environment=ENV_DICE, winner=None),
            PairwiseOutcome(hotkey_a=BOB, hotkey_b=CAROL, environment=ENV_POKER, winner=BOB),
        ]
        standings = accumulate_points(outcomes, [ALICE, BOB, CAROL])

        alice_pts = next(s for s in standings if s.hotkey == ALICE).points
        bob_pts = next(s for s in standings if s.hotkey == BOB).points
        carol_pts = next(s for s in standings if s.hotkey == CAROL).points

        # Alice: 3 wins × 3 = 9
        assert alice_pts == 9.0
        # Bob: 2 wins × 3 + 1 draw × 1 = 7
        assert bob_pts == 7.0
        # Carol: 1 draw × 1 = 1
        assert carol_pts == 1.0

        # Sorted descending
        assert standings[0].hotkey == ALICE
        assert standings[1].hotkey == BOB
        assert standings[2].hotkey == CAROL


# --- 1c: accumulate_points with env weights ---


class TestAccumulatePointsWeighted:
    def test_weight_flips_winner(self):
        """Alice wins dice (weight 1), Bob wins poker (weight 3). Bob wins overall."""
        outcomes = [
            PairwiseOutcome(hotkey_a=ALICE, hotkey_b=BOB, environment=ENV_DICE, winner=ALICE),
            PairwiseOutcome(hotkey_a=ALICE, hotkey_b=BOB, environment=ENV_POKER, winner=BOB),
        ]
        weights = [
            EnvironmentWeight(environment=ENV_DICE, weight=1.0),
            EnvironmentWeight(environment=ENV_POKER, weight=3.0),
        ]
        standings = accumulate_points(outcomes, [ALICE, BOB], weights)

        alice_pts = next(s for s in standings if s.hotkey == ALICE).points
        bob_pts = next(s for s in standings if s.hotkey == BOB).points

        # Alice: 3 × 1.0 = 3
        assert alice_pts == 3.0
        # Bob: 3 × 3.0 = 9
        assert bob_pts == 9.0
        assert standings[0].hotkey == BOB

    def test_missing_weight_defaults_to_one(self):
        """Environment not in weight list → default multiplier 1.0."""
        outcomes = [
            PairwiseOutcome(hotkey_a=ALICE, hotkey_b=BOB, environment=ENV_DICE, winner=ALICE),
            PairwiseOutcome(hotkey_a=ALICE, hotkey_b=BOB, environment=ENV_RUMMY, winner=ALICE),
        ]
        # Only provide weight for dice
        weights = [EnvironmentWeight(environment=ENV_DICE, weight=2.0)]
        standings = accumulate_points(outcomes, [ALICE, BOB], weights)

        alice_pts = next(s for s in standings if s.hotkey == ALICE).points
        # dice: 3 × 2.0 = 6, rummy: 3 × 1.0 = 3 → total 9
        assert alice_pts == 9.0


# --- 1d: compute_pvp_tournament_points (full chain) ---


class TestComputePvpTournamentPoints:
    def test_matches_manual_chain(self):
        """Full chain from PvPGroupResults matches manually calling pvp_results_to_pairwise + accumulate_points."""
        pairs = [
            _make_pair_result(ALICE, BOB, {ENV_DICE: (8, 2, 0)}),
            _make_pair_result(ALICE, CAROL, {ENV_DICE: (3, 7, 0)}),
            _make_pair_result(BOB, CAROL, {ENV_DICE: (6, 4, 0)}),
        ]
        group = _make_group_results([ALICE, BOB, CAROL], pairs)

        full_chain = compute_pvp_tournament_points(group)
        manual_outcomes = pvp_results_to_pairwise(group)
        manual_standings = accumulate_points(manual_outcomes, [ALICE, BOB, CAROL])

        for full, manual in zip(full_chain, manual_standings):
            assert full.hotkey == manual.hotkey
            assert full.points == manual.points


# --- 1e: individual_scores_to_pairwise ---


class TestIndividualScoresToPairwise:
    def test_clear_winner(self):
        """Score difference exceeds margin → winner."""
        scores = {ALICE: 100.0, BOB: 80.0}
        outcomes = individual_scores_to_pairwise(scores, ENV_DICE, win_margin=0.015)

        assert len(outcomes) == 1
        # Alice: 100 > 80 + 80 * 0.015 = 81.2 → Alice wins
        assert outcomes[0].winner == ALICE

    def test_within_margin_is_draw(self):
        """Scores within margin → draw."""
        scores = {ALICE: 100.0, BOB: 99.0}
        outcomes = individual_scores_to_pairwise(scores, ENV_DICE, win_margin=0.015)

        # Alice: 100 > 99 + 99 * 0.015 = 100.485? No. → Draw
        assert outcomes[0].winner is None

    def test_three_players_produce_three_outcomes(self):
        """C(3,2) = 3 pairwise comparisons."""
        scores = {ALICE: 100.0, BOB: 50.0, CAROL: 75.0}
        outcomes = individual_scores_to_pairwise(scores, ENV_DICE, win_margin=0.015)
        assert len(outcomes) == 3

    def test_zero_score_margin_threshold(self):
        """When score_b is 0, threshold = 0. Any positive score_a wins."""
        scores = {ALICE: 1.0, BOB: 0.0}
        outcomes = individual_scores_to_pairwise(scores, ENV_DICE, win_margin=0.5)

        # threshold = abs(0) * 0.5 = 0, so 1.0 > 0.0 + 0 → Alice wins
        assert outcomes[0].winner == ALICE


# --- 1f: MCTS outcomes are compatible with accumulate_points ---


class TestIndividualAccumulateCompatibility:
    def test_mcts_outcomes_feed_into_accumulate(self):
        """MCTS-generated outcomes produce valid standings."""
        scores = {ALICE: 100.0, BOB: 50.0, CAROL: 80.0}
        outcomes = individual_scores_to_pairwise(scores, ENV_DICE, win_margin=0.015)
        standings = accumulate_points(outcomes, [ALICE, BOB, CAROL])

        assert len(standings) == 3
        # Alice beats both, Carol beats Bob, Bob beats nobody
        assert standings[0].hotkey == ALICE
        assert standings[1].hotkey == CAROL


# --- 1g: exponential_decline_mapping ---


class TestExponentialDeclineMapping:
    def test_rank_one_is_highest(self):
        w1 = exponential_decline_mapping(5, 1)
        w2 = exponential_decline_mapping(5, 2)
        assert w1 > w2

    def test_monotonic_decrease(self):
        weights = [exponential_decline_mapping(5, r) for r in range(1, 6)]
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1]

    def test_single_participant(self):
        assert exponential_decline_mapping(1, 1) == 1.0

    def test_weights_sum_to_one(self):
        """All ranks' weights should sum to approximately 1.0 (normalization)."""
        n = 10
        total = sum(exponential_decline_mapping(n, r) for r in range(1, n + 1))
        assert abs(total - 1.0) < 1e-9


# --- 1h: tournament_scores_to_weights ---


class TestTournamentScoresToWeights:
    def test_prev_winner_won_final_gets_rank_one(self):
        scores = [
            TournamentScore(hotkey=BOB, score=10.0),
            TournamentScore(hotkey=CAROL, score=5.0),
        ]
        weights = tournament_scores_to_weights(scores, prev_winner_hotkey=ALICE, prev_winner_won_final=True)

        assert ALICE in weights
        assert weights[ALICE] > weights[BOB] > weights[CAROL]

    def test_prev_winner_lost_final_gets_rank_two(self):
        """Prev winner in scores but lost final → placed 2nd."""
        scores = [
            TournamentScore(hotkey=ALICE, score=8.0),  # prev winner participated
            TournamentScore(hotkey=BOB, score=10.0),
        ]
        weights = tournament_scores_to_weights(scores, prev_winner_hotkey=ALICE, prev_winner_won_final=False)

        # BOB had highest score → rank 1, ALICE forced to rank 2
        assert weights[BOB] > weights[ALICE]

    def test_prev_winner_default_win_gets_rank_one(self):
        """Prev winner not in scores (won by default) → placed 1st."""
        scores = [
            TournamentScore(hotkey=BOB, score=10.0),
        ]
        weights = tournament_scores_to_weights(scores, prev_winner_hotkey=ALICE, prev_winner_won_final=False)
        # Alice not in scores → won by default → rank 1
        assert weights[ALICE] > weights[BOB]

    def test_tied_players_get_equal_weights(self):
        scores = [
            TournamentScore(hotkey=ALICE, score=10.0),
            TournamentScore(hotkey=BOB, score=10.0),
        ]
        weights = tournament_scores_to_weights(scores, prev_winner_hotkey=None, prev_winner_won_final=False)

        assert abs(weights[ALICE] - weights[BOB]) < 1e-9

    def test_zero_scores_excluded(self):
        scores = [
            TournamentScore(hotkey=ALICE, score=0.0),
            TournamentScore(hotkey=BOB, score=10.0),
        ]
        weights = tournament_scores_to_weights(scores, prev_winner_hotkey=None, prev_winner_won_final=False)

        assert ALICE not in weights
        assert BOB in weights

    def test_empty_scores_no_winner(self):
        assert tournament_scores_to_weights([], None, False) == {}


# --- 1i: determine_boss_round_winner (environment = must win ALL) ---


class TestDetermineBossRoundWinnerEnv:
    """Environment tournaments handled separately in determine_env_tournament_winner,
    which requires DB access. TEXT/IMAGE go through determine_boss_round_winner, which
    now requires a comprehensive victory: the challenger may lose at most one boss-round
    task (e.g. 5/6). ENV uses a different path.
    """

    def test_empty_task_winners_boss_retains(self):
        assert determine_boss_round_winner([], "boss", TournamentType.TEXT) == "boss"

    def test_challenger_loses_at_most_one_wins_text(self):
        """Text: 5/6 tasks (one loss) → challenger wins."""
        winners = ["challenger"] * 5 + ["boss"]
        assert determine_boss_round_winner(winners, "boss", TournamentType.TEXT) == "challenger"

    def test_challenger_two_losses_boss_retains_text(self):
        """Text: 4/6 tasks (two losses) is no longer enough → boss retains."""
        winners = ["challenger"] * 4 + ["boss", "boss"]
        assert determine_boss_round_winner(winners, "boss", TournamentType.TEXT) == "boss"

    def test_two_of_three_one_loss_wins_text(self):
        """Text: 2/3 tasks is exactly one loss → challenger wins (lose-at-most-one)."""
        winners = ["challenger", "boss", "challenger"]
        assert determine_boss_round_winner(winners, "boss", TournamentType.TEXT) == "challenger"

    def test_one_of_three_two_losses_boss_retains_text(self):
        """Text: 1/3 tasks (two losses) → boss retains."""
        winners = ["challenger", "boss", "boss"]
        assert determine_boss_round_winner(winners, "boss", TournamentType.TEXT) == "boss"

    def test_image_same_rules_as_text(self):
        """Image uses the same lose-at-most-one rule as text."""
        winners = ["challenger"] * 5 + ["boss"]
        assert determine_boss_round_winner(winners, "boss", TournamentType.IMAGE) == "challenger"

    def test_all_boss_wins(self):
        winners = ["boss", "boss", "boss"]
        assert determine_boss_round_winner(winners, "boss", TournamentType.TEXT) == "boss"

    def test_all_challenger_wins(self):
        winners = ["challenger", "challenger", "challenger"]
        assert determine_boss_round_winner(winners, "boss", TournamentType.TEXT) == "challenger"


# --- 1j: get_real_winner_hotkey ---


class TestGetRealWinnerHotkey:
    def test_emission_burn_resolves_to_base(self):
        assert get_real_winner_hotkey(cts.EMISSION_BURN_HOTKEY, "real_champ") == "real_champ"

    def test_regular_hotkey_passes_through(self):
        assert get_real_winner_hotkey("regular_winner", "old_champ") == "regular_winner"

    def test_none_winner(self):
        assert get_real_winner_hotkey(None, "old_champ") is None

    def test_emission_burn_no_base(self):
        """EMISSION_BURN_HOTKEY but no base → returns EMISSION_BURN_HOTKEY itself."""
        assert get_real_winner_hotkey(cts.EMISSION_BURN_HOTKEY, None) == cts.EMISSION_BURN_HOTKEY


# --- Integration: calculate_tournament_type_scores_from_data ---


class TestCalculateTournamentTypeScores:
    def _make_tournament_data(
        self,
        rounds: list[TournamentRoundResult],
        winner_hotkey: str | None = None,
        base_winner_hotkey: str | None = None,
    ) -> TournamentResultsWithWinners:
        return TournamentResultsWithWinners(
            tournament_id="test-tourn-001",
            rounds=rounds,
            winner_hotkey=winner_hotkey,
            base_winner_hotkey=base_winner_hotkey,
        )

    def test_none_tournament_data(self):
        result = calculate_tournament_type_scores_from_data(TournamentType.TEXT, None)
        assert result.scores == []
        assert result.prev_winner_hotkey is None
        assert result.prev_winner_won_final is False

    def test_text_tournament_winner_excluded_from_scores(self):
        """In text tournaments, the actual winner doesn't earn points — only non-winners do."""
        data = self._make_tournament_data(
            rounds=[
                TournamentRoundResult(
                    round_id="r1", round_number=1, round_type="group", is_final_round=True,
                    tasks=[
                        TournamentTaskScore(task_id="t1", group_id=None, pair_id=None, winner=ALICE, participant_scores=[]),
                        TournamentTaskScore(task_id="t2", group_id=None, pair_id=None, winner=ALICE, participant_scores=[]),
                    ],
                ),
            ],
            winner_hotkey=ALICE,
        )
        result = calculate_tournament_type_scores_from_data(TournamentType.TEXT, data)
        hotkeys_with_scores = {s.hotkey for s in result.scores}
        assert ALICE not in hotkeys_with_scores

    def test_environment_tournament_ranked_scoring(self):
        """Environment tournaments use ranked participant scoring, not just winner-takes-all."""
        data = self._make_tournament_data(
            rounds=[
                TournamentRoundResult(
                    round_id="r1", round_number=1, round_type="group", is_final_round=False,
                    tasks=[
                        TournamentTaskScore(
                            task_id="t1", group_id="g1", pair_id=None, winner=None,
                            participant_scores=[
                                {"hotkey": ALICE, "test_loss": 10.0},
                                {"hotkey": BOB, "test_loss": 5.0},
                                {"hotkey": CAROL, "test_loss": 1.0},
                            ],
                        ),
                    ],
                ),
            ],
            winner_hotkey=None,
        )
        result = calculate_tournament_type_scores_from_data(TournamentType.ENVIRONMENT, data)
        scores_by_hotkey = {s.hotkey: s.score for s in result.scores}

        # Higher test_loss = better for env → Alice ranked 1st, Bob 2nd, Carol 3rd
        assert scores_by_hotkey[ALICE] > scores_by_hotkey[BOB] > scores_by_hotkey[CAROL]

    def test_emission_burn_hotkey_excluded_from_env_scoring(self):
        """EMISSION_BURN_HOTKEY placeholder excluded from environment scoring."""
        data = self._make_tournament_data(
            rounds=[
                TournamentRoundResult(
                    round_id="r1", round_number=1, round_type="group", is_final_round=False,
                    tasks=[
                        TournamentTaskScore(
                            task_id="t1", group_id="g1", pair_id=None, winner=None,
                            participant_scores=[
                                {"hotkey": cts.EMISSION_BURN_HOTKEY, "test_loss": 100.0},
                                {"hotkey": ALICE, "test_loss": 10.0},
                                {"hotkey": BOB, "test_loss": 5.0},
                            ],
                        ),
                    ],
                ),
            ],
            winner_hotkey=cts.EMISSION_BURN_HOTKEY,
            base_winner_hotkey="real_champ",
        )
        result = calculate_tournament_type_scores_from_data(TournamentType.ENVIRONMENT, data)
        hotkeys_with_scores = {s.hotkey for s in result.scores}

        assert cts.EMISSION_BURN_HOTKEY not in hotkeys_with_scores
        assert "real_champ" not in hotkeys_with_scores
        assert ALICE in hotkeys_with_scores
        assert BOB in hotkeys_with_scores

    def test_prev_winner_won_final_detected(self):
        """prev_winner_won_final is True when winner appears in final round tasks."""
        data = self._make_tournament_data(
            rounds=[
                TournamentRoundResult(
                    round_id="r1", round_number=1, round_type="boss", is_final_round=True,
                    tasks=[
                        TournamentTaskScore(task_id="t1", group_id=None, pair_id=None, winner=ALICE, participant_scores=[]),
                    ],
                ),
            ],
            winner_hotkey=ALICE,
        )
        result = calculate_tournament_type_scores_from_data(TournamentType.TEXT, data)
        assert result.prev_winner_won_final is True
        assert result.prev_winner_hotkey == ALICE
