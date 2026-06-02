"""Tests for PvP scoring: determine_outcome from game returns."""


from validator.evaluation.pvp.models import GameOutcome
from validator.evaluation.pvp.models import GameScoringContext
from validator.evaluation.pvp.scoring import determine_outcome


class TestZeroSum:

    def test_win(self) -> None:
        ctx = GameScoringContext(returns=[1.0, -1.0], player_id=0, is_zero_sum=True, min_utility=-1.0, max_utility=1.0)
        assert determine_outcome(ctx) == GameOutcome.WIN

    def test_loss(self) -> None:
        ctx = GameScoringContext(returns=[-1.0, 1.0], player_id=0, is_zero_sum=True, min_utility=-1.0, max_utility=1.0)
        assert determine_outcome(ctx) == GameOutcome.LOSS

    def test_draw(self) -> None:
        ctx = GameScoringContext(returns=[0.0, 0.0], player_id=0, is_zero_sum=True, min_utility=-1.0, max_utility=1.0)
        assert determine_outcome(ctx) == GameOutcome.DRAW

    def test_exact_boundary_is_draw(self) -> None:
        """Score of exactly 0.5 after normalization should be DRAW, not WIN."""
        ctx = GameScoringContext(returns=[0.0, 0.0], player_id=0, is_zero_sum=True, min_utility=-1.0, max_utility=1.0)
        assert determine_outcome(ctx) == GameOutcome.DRAW

    def test_degenerate_equal_utility_is_draw(self) -> None:
        """When max_utility == min_utility, normalization can't discriminate — always DRAW."""
        ctx = GameScoringContext(returns=[0.0, 0.0], player_id=0, is_zero_sum=True, min_utility=0.0, max_utility=0.0)
        assert determine_outcome(ctx) == GameOutcome.DRAW

    def test_player_id_1(self) -> None:
        """Scoring from player 1's perspective."""
        ctx = GameScoringContext(returns=[-1.0, 1.0], player_id=1, is_zero_sum=True, min_utility=-1.0, max_utility=1.0)
        assert determine_outcome(ctx) == GameOutcome.WIN


class TestGeneralSum:

    def test_ahead(self) -> None:
        ctx = GameScoringContext(returns=[10.0, 5.0], player_id=0, is_zero_sum=False, min_utility=0.0, max_utility=10.0)
        assert determine_outcome(ctx) == GameOutcome.WIN

    def test_behind(self) -> None:
        ctx = GameScoringContext(returns=[3.0, 7.0], player_id=0, is_zero_sum=False, min_utility=0.0, max_utility=10.0)
        assert determine_outcome(ctx) == GameOutcome.LOSS

    def test_tied(self) -> None:
        ctx = GameScoringContext(returns=[5.0, 5.0], player_id=0, is_zero_sum=False, min_utility=0.0, max_utility=10.0)
        assert determine_outcome(ctx) == GameOutcome.DRAW
