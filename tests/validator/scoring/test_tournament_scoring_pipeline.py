"""Tests for the tournament scoring pipeline: PvP/MCTS results → pairwise → points → weights.

Exercises the full chain that determines tournament outcomes and emission weights.
"""


import pytest

import validator.scoring.constants as cts
from core.constants.environments import EnvironmentName
from validator.evaluation.pvp.models import PvPEnvironmentResult
from validator.evaluation.pvp.models import PvPEvalMetadata
from validator.evaluation.pvp.models import PvPGroupResults
from validator.evaluation.pvp.models import PvPPairResult
from validator.scoring.tournaments import accumulate_points
from validator.scoring.tournaments import calculate_tournament_type_scores_from_data
from validator.scoring.tournaments import compute_pvp_tournament_points
from validator.scoring.tournaments import exponential_decline_mapping
from validator.scoring.tournaments import get_boss_round_pair_weights
from validator.scoring.tournaments import get_tournament_weights_from_data
from validator.scoring.tournaments import individual_scores_to_pairwise
from validator.scoring.tournaments import pvp_results_to_pairwise
from validator.scoring.tournaments import tournament_scores_to_weights
from validator.scoring.weights import apply_tournament_weights
from validator.tournament.models import EnvironmentWeight
from validator.tournament.models import PairwiseOutcome
from validator.tournament.models import TournamentResultsWithWinners
from validator.tournament.models import TournamentRoundResult
from validator.tournament.models import TournamentScore
from validator.tournament.models import TournamentTaskScore
from validator.tournament.models import TournamentType
from validator.tournament.round_results import determine_boss_round_winner
from validator.tournament.round_results import get_real_tournament_winner
from validator.tournament.round_results import get_real_winner_hotkey


# --- Fixtures ---


ALICE = "5GAlice"
BOB = "5GBob"
CAROL = "5GCarol"
BOSS = "5GBoss"
CHALLENGER = "5GChallenger"
DAVE = "5GDave"

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

    def test_non_increasing_by_rank(self):
        weights = [exponential_decline_mapping(5, r) for r in range(1, 6)]
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]

    def test_only_top_two_ranks_paid(self):
        assert exponential_decline_mapping(5, 1) == pytest.approx(0.8)
        assert exponential_decline_mapping(5, 2) == pytest.approx(0.2)
        assert exponential_decline_mapping(5, 3) == 0.0
        assert exponential_decline_mapping(5, 5) == 0.0

    def test_single_participant(self):
        assert exponential_decline_mapping(1, 1) == 1.0

    def test_weights_sum_to_one(self):
        """Paid ranks' weights should sum to approximately 1.0."""
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
    which requires DB access. TEXT/IMAGE go through determine_boss_round_winner,
    which requires a comprehensive victory: the challenger may lose at most one
    boss-round task. ENV uses a different path.
    """

    def test_empty_task_winners_boss_retains(self):
        assert determine_boss_round_winner([], "boss", TournamentType.TEXT) == "boss"

    def test_challenger_loses_at_most_one_wins_text(self):
        winners = ["challenger"] * 5 + ["boss"]
        assert determine_boss_round_winner(winners, "boss", TournamentType.TEXT) == "challenger"

    def test_challenger_two_losses_boss_retains_text(self):
        winners = ["challenger"] * 4 + ["boss", "boss"]
        assert determine_boss_round_winner(winners, "boss", TournamentType.TEXT) == "boss"

    def test_two_of_three_one_loss_wins_text(self):
        winners = ["challenger", "boss", "challenger"]
        assert determine_boss_round_winner(winners, "boss", TournamentType.TEXT) == "challenger"

    def test_one_of_three_two_losses_boss_retains_text(self):
        winners = ["challenger", "boss", "boss"]
        assert determine_boss_round_winner(winners, "boss", TournamentType.TEXT) == "boss"

    def test_image_same_rules_as_text(self):
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
                    round_id="r2", round_number=2, round_type="boss", is_final_round=True,
                    tasks=[
                        TournamentTaskScore(task_id="t1", group_id=None, pair_id=None, winner=ALICE, participant_scores=[]),
                        TournamentTaskScore(task_id="t2", group_id=None, pair_id=None, winner=BOB, participant_scores=[]),
                    ],
                ),
            ],
            winner_hotkey=ALICE,
        )
        result = calculate_tournament_type_scores_from_data(TournamentType.TEXT, data)
        hotkeys_with_scores = {s.hotkey for s in result.scores}
        assert ALICE not in hotkeys_with_scores
        assert BOB in hotkeys_with_scores

    def test_environment_tournament_ranked_scoring(self):
        """Environment tournaments use ranked participant scoring, not just winner-takes-all."""
        data = self._make_tournament_data(
            rounds=[
                TournamentRoundResult(
                    round_id="r2", round_number=2, round_type="group", is_final_round=False,
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
                    round_id="r2", round_number=2, round_type="group", is_final_round=False,
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
                    round_id="r2", round_number=2, round_type="boss", is_final_round=True,
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

    def test_round_one_tasks_do_not_earn_emissions(self):
        data = self._make_tournament_data(
            rounds=[
                TournamentRoundResult(
                    round_id="r1",
                    round_number=1,
                    round_type="group",
                    is_final_round=False,
                    tasks=[
                        TournamentTaskScore(
                            task_id="text-r1",
                            group_id="g1",
                            pair_id=None,
                            winner=BOB,
                            participant_scores=[],
                        ),
                    ],
                ),
                TournamentRoundResult(
                    round_id="r2",
                    round_number=2,
                    round_type="knockout",
                    is_final_round=False,
                    tasks=[
                        TournamentTaskScore(
                            task_id="text-r2",
                            group_id=None,
                            pair_id="p1",
                            winner=CAROL,
                            participant_scores=[],
                        ),
                    ],
                ),
            ],
            winner_hotkey=ALICE,
        )

        result = calculate_tournament_type_scores_from_data(TournamentType.TEXT, data)
        scores_by_hotkey = {score.hotkey: score.score for score in result.scores}

        assert BOB not in scores_by_hotkey
        assert scores_by_hotkey[CAROL] == 2 * cts.TOURNAMENT_TEXT_WEIGHT

    def test_environment_round_one_tasks_do_not_earn_emissions(self):
        data = self._make_tournament_data(
            rounds=[
                TournamentRoundResult(
                    round_id="r1",
                    round_number=1,
                    round_type="group",
                    is_final_round=False,
                    tasks=[
                        TournamentTaskScore(
                            task_id="env-r1",
                            group_id="g1",
                            pair_id=None,
                            winner=None,
                            participant_scores=[
                                {"hotkey": ALICE, "test_loss": 10.0},
                                {"hotkey": BOB, "test_loss": 5.0},
                            ],
                        ),
                    ],
                ),
            ],
            winner_hotkey=None,
        )

        result = calculate_tournament_type_scores_from_data(TournamentType.ENVIRONMENT, data)

        assert result.scores == []


def _boss_round_tournament(
    final_participants: list[str],
    winner_hotkey: str | None,
    base_winner_hotkey: str | None = None,
    earlier_rounds: list[TournamentRoundResult] | None = None,
    num_final_tasks: int = 1,
) -> TournamentResultsWithWinners:
    """Build a tournament whose final boss round pairs final_participants."""
    final_round = TournamentRoundResult(
        round_id="final",
        round_number=99,
        round_type="knockout",
        is_final_round=True,
        tasks=[
            TournamentTaskScore(
                task_id=f"boss-task-{i}",
                group_id=None,
                pair_id="p1",
                winner=None,
                participant_scores=[{"hotkey": hotkey, "test_loss": 1.0} for hotkey in final_participants],
            )
            for i in range(num_final_tasks)
        ],
    )
    return TournamentResultsWithWinners(
        tournament_id="t",
        rounds=(earlier_rounds or []) + [final_round],
        winner_hotkey=winner_hotkey,
        base_winner_hotkey=base_winner_hotkey,
    )


class TestGetBossRoundPairWeights:
    """Only the two boss-round finalists earn emissions: champion 80%, runner-up 20%."""

    def _tournament(
        self,
        final_participants: list[str],
        winner_hotkey: str | None,
        base_winner_hotkey: str | None = None,
        earlier_rounds: list[TournamentRoundResult] | None = None,
    ) -> TournamentResultsWithWinners:
        return _boss_round_tournament(final_participants, winner_hotkey, base_winner_hotkey, earlier_rounds)

    def test_none_data(self):
        assert get_boss_round_pair_weights(None) == {}

    def test_boss_defends(self):
        data = self._tournament(
            final_participants=[cts.EMISSION_BURN_HOTKEY, CHALLENGER],
            winner_hotkey=cts.EMISSION_BURN_HOTKEY,
            base_winner_hotkey=BOSS,
        )

        weights = get_boss_round_pair_weights(data)

        assert weights == {BOSS: pytest.approx(0.8), CHALLENGER: pytest.approx(0.2)}

    def test_challenger_dethrones_boss(self):
        data = self._tournament(
            final_participants=[cts.EMISSION_BURN_HOTKEY, CHALLENGER],
            winner_hotkey=CHALLENGER,
            base_winner_hotkey=BOSS,
        )

        weights = get_boss_round_pair_weights(data)

        assert weights == {CHALLENGER: pytest.approx(0.8), BOSS: pytest.approx(0.2)}

    def test_semifinalist_earns_nothing_even_when_tied_on_points(self):
        earlier = [
            TournamentRoundResult(
                round_id="semi",
                round_number=2,
                round_type="knockout",
                is_final_round=False,
                tasks=[
                    TournamentTaskScore(
                        task_id="semi-task",
                        group_id=None,
                        pair_id="p0",
                        winner=DAVE,
                        participant_scores=[{"hotkey": DAVE, "test_loss": 1.0}, {"hotkey": "5GEli", "test_loss": 2.0}],
                    )
                ],
            )
        ]
        data = self._tournament(
            final_participants=[cts.EMISSION_BURN_HOTKEY, CHALLENGER],
            winner_hotkey=cts.EMISSION_BURN_HOTKEY,
            base_winner_hotkey=BOSS,
            earlier_rounds=earlier,
        )

        weights = get_boss_round_pair_weights(data)

        assert DAVE not in weights
        assert weights == {BOSS: pytest.approx(0.8), CHALLENGER: pytest.approx(0.2)}

    def test_no_final_round_pays_champion_alone(self):
        data = TournamentResultsWithWinners(
            tournament_id="t",
            rounds=[
                TournamentRoundResult(
                    round_id="r1",
                    round_number=1,
                    round_type="group",
                    is_final_round=False,
                    tasks=[],
                )
            ],
            winner_hotkey=BOSS,
        )

        assert get_boss_round_pair_weights(data) == {BOSS: pytest.approx(1.0)}

    def test_burn_when_no_base_winner(self):
        data = self._tournament(
            final_participants=[cts.EMISSION_BURN_HOTKEY, CHALLENGER],
            winner_hotkey=cts.EMISSION_BURN_HOTKEY,
            base_winner_hotkey=None,
        )

        weights = get_boss_round_pair_weights(data)

        assert weights == {cts.EMISSION_BURN_HOTKEY: pytest.approx(0.8), CHALLENGER: pytest.approx(0.2)}


class TestBossRoundEmissionsProductionPath:
    def test_all_three_types_pay_only_boss_pair(self):
        for tournament_type in [TournamentType.TEXT, TournamentType.IMAGE, TournamentType.ENVIRONMENT]:
            num_tasks = 3 if tournament_type == TournamentType.ENVIRONMENT else 1
            earlier = [
                TournamentRoundResult(
                    round_id="semi",
                    round_number=2,
                    round_type="knockout",
                    is_final_round=False,
                    tasks=[
                        TournamentTaskScore(
                            task_id="s",
                            group_id=None,
                            pair_id="ps",
                            winner=DAVE,
                            participant_scores=[
                                {"hotkey": DAVE, "test_loss": 1.0},
                                {"hotkey": "5GEli", "test_loss": 2.0},
                            ],
                        )
                    ],
                )
            ]
            data = _boss_round_tournament(
                final_participants=[cts.EMISSION_BURN_HOTKEY, CHALLENGER],
                winner_hotkey=cts.EMISSION_BURN_HOTKEY,
                base_winner_hotkey=BOSS,
                earlier_rounds=earlier,
                num_final_tasks=num_tasks,
            )

            text_w, image_w, env_w = get_tournament_weights_from_data(
                data if tournament_type == TournamentType.TEXT else None,
                data if tournament_type == TournamentType.IMAGE else None,
                data if tournament_type == TournamentType.ENVIRONMENT else None,
            )
            weights = {
                TournamentType.TEXT: text_w,
                TournamentType.IMAGE: image_w,
                TournamentType.ENVIRONMENT: env_w,
            }[tournament_type]

            assert weights == {BOSS: pytest.approx(0.8), CHALLENGER: pytest.approx(0.2)}
            assert DAVE not in weights
            assert "5GEli" not in weights

    def test_champion_key_matches_get_real_tournament_winner_when_boss_defends(self):
        data = _boss_round_tournament([cts.EMISSION_BURN_HOTKEY, CHALLENGER], cts.EMISSION_BURN_HOTKEY, BOSS)
        weights = get_boss_round_pair_weights(data)
        champion_key = max(weights, key=lambda hotkey: weights[hotkey])

        assert champion_key == get_real_tournament_winner(data) == BOSS
        assert weights[champion_key] == pytest.approx(0.8)

    def test_champion_key_matches_get_real_tournament_winner_when_dethroned(self):
        data = _boss_round_tournament([cts.EMISSION_BURN_HOTKEY, CHALLENGER], CHALLENGER, BOSS)
        weights = get_boss_round_pair_weights(data)
        champion_key = max(weights, key=lambda hotkey: weights[hotkey])

        assert champion_key == get_real_tournament_winner(data) == CHALLENGER
        assert weights[champion_key] == pytest.approx(0.8)

    def test_apply_tournament_weights_routes_champion_to_boost_runner_up_to_base(self):
        data = _boss_round_tournament([cts.EMISSION_BURN_HOTKEY, CHALLENGER], cts.EMISSION_BURN_HOTKEY, BOSS)
        text_w, image_w, env_w = get_tournament_weights_from_data(data, None, None)

        hotkey_to_node_id = {BOSS: 0, CHALLENGER: 1, DAVE: 2}
        all_node_weights = [0.0, 0.0, 0.0]
        scaled_text_tournament_weight = 0.5
        scaled_text_base_weight = 0.1

        undistributed = apply_tournament_weights(
            text_w,
            image_w,
            env_w,
            hotkey_to_node_id,
            all_node_weights,
            scaled_text_tournament_weight,
            0.0,
            0.0,
            scaled_text_base_weight,
            0.0,
            0.0,
            get_real_tournament_winner(data),
            None,
            None,
        )

        assert all_node_weights[0] == pytest.approx(0.8 * scaled_text_tournament_weight)
        assert all_node_weights[1] == pytest.approx(0.2 * scaled_text_base_weight)
        assert all_node_weights[2] == 0.0
        distributed = 0.8 * scaled_text_tournament_weight + 0.2 * scaled_text_base_weight
        assert undistributed == pytest.approx(scaled_text_tournament_weight - distributed)
