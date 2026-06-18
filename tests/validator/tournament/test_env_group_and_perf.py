"""Tests for environment tournament group formation, group winner advancement,
performance calculator win percentage, and env tournament winner determination.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import validator.scoring.constants as cts
import validator.tournament.constants as t_cst
from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import GroupRound
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType


# =============================================================================
# 1. Group formation logic (organise_tournament_round)
# =============================================================================


class TestEnvGroupFormation:
    """Test organise_tournament_round for environment tournaments."""

    def _make_nodes(self, n: int) -> list:
        """Create n mock nodes with unique hotkeys."""
        nodes = []
        for i in range(n):
            node = MagicMock()
            node.hotkey = f"hk_{i:03d}"
            nodes.append(node)
        return nodes

    def _form_groups(self, nodes) -> GroupRound:
        from validator.tournament.tournament_manager import organise_tournament_round
        config = MagicMock()
        return organise_tournament_round(
            nodes, config,
            tournament_type=TournamentType.ENVIRONMENT,
            round_id="test_round",
            round_number=1,
        )

    def test_6_participants_small_field(self):
        """6 plus reserved boss slot -> 3 round-1 groups of 2."""
        nodes = self._make_nodes(6)
        result = self._form_groups(nodes)
        assert len(result.groups) == 3
        assert sorted(len(group.member_ids) for group in result.groups) == [2, 2, 2]

    def test_7_participants_three_small_groups(self):
        """7 plus reserved boss slot -> 3 round-1 groups sized 3+2+2."""
        nodes = self._make_nodes(7)
        result = self._form_groups(nodes)
        assert len(result.groups) == 3
        sizes = sorted([len(g.member_ids) for g in result.groups])
        assert sizes == [2, 2, 3]

    def test_12_participants_four_groups(self):
        """12 plus reserved boss slot -> 4 groups of 3."""
        nodes = self._make_nodes(12)
        result = self._form_groups(nodes)
        assert len(result.groups) == 4
        assert all(len(g.member_ids) == 3 for g in result.groups)

    def test_13_participants_four_groups(self):
        """13 plus reserved boss slot -> 4 roughly balanced groups."""
        nodes = self._make_nodes(13)
        result = self._form_groups(nodes)
        assert len(result.groups) == 4
        sizes = sorted(len(g.member_ids) for g in result.groups)
        assert sizes == [3, 3, 3, 4]
        assert sum(sizes) == 13
        assert min(sizes) >= t_cst.MIN_ENVIRONMENT_GROUP_SIZE

    def test_2_participants_minimum(self):
        """2 participants is the minimum."""
        nodes = self._make_nodes(2)
        result = self._form_groups(nodes)
        assert len(result.groups) == 1
        assert len(result.groups[0].member_ids) == 2

    def test_1_participant_raises(self):
        """1 participant < MIN_ENVIRONMENT_GROUP_SIZE → ValueError."""
        nodes = self._make_nodes(1)
        with pytest.raises(ValueError, match="minimum"):
            self._form_groups(nodes)

    def test_all_participants_assigned(self):
        """Every participant ends up in exactly one group."""
        for n in [2, 5, 6, 7, 11, 12, 18, 25]:
            nodes = self._make_nodes(n)
            result = self._form_groups(nodes)
            all_members = []
            for g in result.groups:
                all_members.extend(g.member_ids)
            assert len(all_members) == n, f"n={n}: expected {n} members, got {len(all_members)}"
            assert len(set(all_members)) == n, f"n={n}: duplicate assignments"

    def test_no_group_exceeds_max(self):
        for n in [2, 6, 7, 12, 13, 18, 24, 30]:
            nodes = self._make_nodes(n)
            result = self._form_groups(nodes)
            for g in result.groups:
                assert len(g.member_ids) <= t_cst.MAX_ENVIRONMENT_GROUP_SIZE, (
                    f"n={n}: group has {len(g.member_ids)} > max {t_cst.MAX_ENVIRONMENT_GROUP_SIZE}"
                )

    def test_smallest_group_fits_boss(self):
        for n in [2, 4, 5, 6, 7, 8, 11, 12, 13, 16, 18, 24, 30]:
            nodes = self._make_nodes(n)
            result = self._form_groups(nodes)
            min_size = min(len(group.member_ids) for group in result.groups)
            assert min_size + 1 <= t_cst.MAX_ENVIRONMENT_GROUP_SIZE

    def test_no_group_below_min(self):
        for n in [2, 3, 5, 6, 7, 11, 12, 18]:
            nodes = self._make_nodes(n)
            result = self._form_groups(nodes)
            for g in result.groups:
                assert len(g.member_ids) >= t_cst.MIN_ENVIRONMENT_GROUP_SIZE, (
                    f"n={n}: group has {len(g.member_ids)} < min {t_cst.MIN_ENVIRONMENT_GROUP_SIZE}"
                )

    def test_round_metadata_preserved(self):
        nodes = self._make_nodes(4)
        result = self._form_groups(nodes)
        assert result.round_id == "test_round"
        assert result.round_number == 1


# =============================================================================
# 2. Environment group winner advancement
# =============================================================================


class TestEnvGroupWinnerAdvancement:
    """Test get_environment_group_winners logic with mocked DB."""

    def _mock_miner_result(self, hotkey: str, score: float):
        result = MagicMock()
        result.hotkey = hotkey
        result.adjusted_loss = score
        result.test_loss = score
        result.is_finetune = True
        result.score = 0.0
        result.score_reason = None
        result.synth_loss = 0.0
        result.task_type = TaskType.ENVIRONMENTTASK
        return result

    @pytest.mark.asyncio
    async def test_top_n_advance_from_group(self):
        """In a group of 4 (excluding boss), top ENV_ADVANCE_PER_GROUP advance."""
        from validator.tournament.round_results import get_environment_group_winners

        round_data = TournamentRoundData(
            round_id="r1", tournament_id="t1", round_number=1,
            round_type="group", is_final_round=False,
        )
        tasks = [TournamentTask(tournament_id="t1", round_id="r1", task_id="task_1", group_id="g1")]

        mock_participants = [MagicMock(hotkey=f"hk_{i}") for i in range(4)]
        mock_results = [
            self._mock_miner_result("hk_0", 90.0),  # 1st
            self._mock_miner_result("hk_1", 70.0),  # 2nd
            self._mock_miner_result("hk_2", 50.0),  # 3rd
            self._mock_miner_result("hk_3", 30.0),  # 4th
        ]

        with (
            patch("validator.tournament.round_results.get_tournament_group_members", return_value=mock_participants),
            patch("validator.tournament.round_results.get_task_results_for_ranking", return_value=mock_results),
        ):
            config = MagicMock()
            psql_db = MagicMock()
            winners = await get_environment_group_winners(round_data, tasks, psql_db, config)

        assert len(winners) == t_cst.ENV_ADVANCE_PER_GROUP
        assert "hk_0" in winners

    @pytest.mark.asyncio
    async def test_single_group_boss_retains_when_boss_tops_group(self):
        """Single-group env rounds return no challenger when the boss wins or ties the group."""
        from validator.tournament.round_results import get_environment_group_winners

        round_data = TournamentRoundData(
            round_id="r1", tournament_id="t1", round_number=1,
            round_type="group", is_final_round=False,
        )
        tasks = [TournamentTask(tournament_id="t1", round_id="r1", task_id="task_1", group_id="g1")]

        mock_participants = [
            MagicMock(hotkey=EMISSION_BURN_HOTKEY),
            MagicMock(hotkey="hk_0"),
            MagicMock(hotkey="hk_1"),
            MagicMock(hotkey="hk_2"),
        ]
        mock_results = [
            self._mock_miner_result(EMISSION_BURN_HOTKEY, 100.0),  # Boss scores highest
            self._mock_miner_result("hk_0", 80.0),
            self._mock_miner_result("hk_1", 60.0),
            self._mock_miner_result("hk_2", 40.0),
        ]

        with (
            patch("validator.tournament.round_results.get_tournament_group_members", return_value=mock_participants),
            patch("validator.tournament.round_results.get_task_results_for_ranking", return_value=mock_results),
        ):
            winners = await get_environment_group_winners(round_data, tasks, MagicMock(), MagicMock())

        assert winners == []

    @pytest.mark.asyncio
    async def test_at_least_one_eliminated(self):
        """With 3 non-boss participants and ENV_ADVANCE_PER_GROUP=1,
        at least 1 must be eliminated to guarantee convergence."""
        from validator.tournament.round_results import get_environment_group_winners

        round_data = TournamentRoundData(
            round_id="r1", tournament_id="t1", round_number=1,
            round_type="group", is_final_round=False,
        )
        tasks = [TournamentTask(tournament_id="t1", round_id="r1", task_id="task_1", group_id="g1")]

        mock_participants = [MagicMock(hotkey=f"hk_{i}") for i in range(3)]
        mock_results = [
            self._mock_miner_result("hk_0", 90.0),
            self._mock_miner_result("hk_1", 70.0),
            self._mock_miner_result("hk_2", 50.0),
        ]

        with (
            patch("validator.tournament.round_results.get_tournament_group_members", return_value=mock_participants),
            patch("validator.tournament.round_results.get_task_results_for_ranking", return_value=mock_results),
        ):
            winners = await get_environment_group_winners(round_data, tasks, MagicMock(), MagicMock())

        assert len(winners) < 3, "Must eliminate at least 1 to converge"
        assert len(winners) >= 1

    @pytest.mark.asyncio
    async def test_final_round_returns_all_participants(self):
        """Final round: returns everyone (boss + contender) for determine_env_tournament_winner."""
        from validator.tournament.round_results import get_environment_group_winners

        round_data = TournamentRoundData(
            round_id="r_final", tournament_id="t1", round_number=4,
            round_type="group", is_final_round=True,
        )
        tasks = [TournamentTask(tournament_id="t1", round_id="r_final", task_id="task_1", group_id="g_boss")]

        mock_participants = [
            MagicMock(hotkey=EMISSION_BURN_HOTKEY),
            MagicMock(hotkey="contender"),
        ]

        with patch("validator.tournament.round_results.get_tournament_group_members", return_value=mock_participants):
            winners = await get_environment_group_winners(round_data, tasks, MagicMock(), MagicMock())

        assert EMISSION_BURN_HOTKEY in winners
        assert "contender" in winners


# =============================================================================
# 3. Performance calculator: env win percentage → perf_diff
# =============================================================================


class TestEnvWinPercentage:
    """Test the win_pct formula and perf_diff mapping for environment tasks.

    Formula: win_pct = (2 * challenger_score + boss_score - 3 * num_envs) / (3 * num_envs)
    PvP scoring: each env gives 3 pts for win, 1 for draw, 0 for loss.
    With 3 envs: max points per player = 9, min = 0.
    """

    def _compute_win_pct(self, challenger_score: float, boss_score: float, num_envs: int) -> float:
        win_pct = (2 * challenger_score + boss_score - 3 * num_envs) / (3 * num_envs)
        return max(0.0, win_pct)

    def _compute_perf_diff(self, win_pct: float) -> float:
        if win_pct < cts.PVP_WIN_PCT_THRESHOLD:
            return 0.0
        return cts.EMISSION_MULTIPLIER_THRESHOLD + (win_pct - cts.PVP_WIN_PCT_THRESHOLD) * cts.PVP_PERF_DIFF_SLOPE

    def test_perfect_win_all_envs(self):
        """Challenger wins all 3 envs (9 pts), boss wins none (0 pts).
        win_pct = (2*9 + 0 - 9) / 9 = 1.0"""
        win_pct = self._compute_win_pct(9.0, 0.0, 3)
        assert win_pct == 1.0
        perf_diff = self._compute_perf_diff(win_pct)
        assert perf_diff > 0

    def test_perfect_loss_all_envs(self):
        """Boss wins all 3 envs (9 pts), challenger none (0 pts).
        win_pct = (0 + 9 - 9) / 9 = 0.0"""
        win_pct = self._compute_win_pct(0.0, 9.0, 3)
        assert win_pct == 0.0
        perf_diff = self._compute_perf_diff(win_pct)
        assert perf_diff == 0.0

    def test_all_draws(self):
        """All draws: both get 1 pt per env = 3 pts each for 3 envs.
        win_pct = (2*3 + 3 - 9) / 9 = 0.0"""
        win_pct = self._compute_win_pct(3.0, 3.0, 3)
        assert win_pct == 0.0
        perf_diff = self._compute_perf_diff(win_pct)
        assert perf_diff == 0.0

    def test_below_threshold_zero_perf_diff(self):
        """win_pct just below PVP_WIN_PCT_THRESHOLD → perf_diff = 0."""
        win_pct = cts.PVP_WIN_PCT_THRESHOLD - 0.01
        perf_diff = self._compute_perf_diff(win_pct)
        assert perf_diff == 0.0

    def test_at_threshold_gets_emission_multiplier(self):
        """win_pct exactly at threshold → perf_diff = EMISSION_MULTIPLIER_THRESHOLD."""
        win_pct = cts.PVP_WIN_PCT_THRESHOLD
        perf_diff = self._compute_perf_diff(win_pct)
        assert abs(perf_diff - cts.EMISSION_MULTIPLIER_THRESHOLD) < 1e-9

    def test_above_threshold_linear_scaling(self):
        """Above threshold, perf_diff increases linearly with win_pct."""
        wp1 = cts.PVP_WIN_PCT_THRESHOLD + 0.1
        wp2 = cts.PVP_WIN_PCT_THRESHOLD + 0.2
        pd1 = self._compute_perf_diff(wp1)
        pd2 = self._compute_perf_diff(wp2)
        assert pd2 > pd1
        # Check linearity: difference should be 0.1 * PVP_PERF_DIFF_SLOPE
        expected_delta = 0.1 * cts.PVP_PERF_DIFF_SLOPE
        assert abs((pd2 - pd1) - expected_delta) < 1e-9

    def test_win_pct_clamped_at_zero(self):
        """Negative raw win_pct clamped to 0."""
        # Both score 0: win_pct = (0 + 0 - 9) / 9 = -1.0 → clamped to 0
        win_pct = self._compute_win_pct(0.0, 0.0, 3)
        assert win_pct == 0.0


# =============================================================================
# 4. determine_env_tournament_winner: must win ALL boss round tasks
# =============================================================================


class TestDetermineEnvTournamentWinner:
    """Test the 'must beat boss on ALL tasks' rule with mocked DB."""

    @pytest.mark.asyncio
    async def test_contender_wins_all_three(self):
        from validator.tournament.round_results import determine_env_tournament_winner

        tournament = MagicMock(spec=TournamentData)
        tournament.tournament_id = "t1"

        mock_rounds = [
            MagicMock(is_final_round=True, round_id="r_final"),
        ]
        mock_tasks = [
            MagicMock(task_id="task_1"),
            MagicMock(task_id="task_2"),
            MagicMock(task_id="task_3"),
        ]

        # Contender beats boss on all 3
        scores_by_task = {
            "task_1": {EMISSION_BURN_HOTKEY: 3.0, "contender": 6.0},
            "task_2": {EMISSION_BURN_HOTKEY: 4.0, "contender": 7.0},
            "task_3": {EMISSION_BURN_HOTKEY: 2.0, "contender": 5.0},
        }

        async def mock_get_scores(task_id, psql_db):
            return scores_by_task[task_id]

        with (
            patch("validator.tournament.round_results.get_tournament_rounds", return_value=mock_rounds),
            patch("validator.tournament.round_results.get_tournament_tasks", return_value=mock_tasks),
            patch("validator.tournament.round_results._get_scores_for_task", side_effect=mock_get_scores),
        ):
            result = await determine_env_tournament_winner(tournament, [], MagicMock(), MagicMock())

        # Winner is first in list
        assert result[0] == "contender"
        assert result[1] == EMISSION_BURN_HOTKEY

    @pytest.mark.asyncio
    async def test_contender_loses_one_boss_retains(self):
        from validator.tournament.round_results import determine_env_tournament_winner

        tournament = MagicMock(spec=TournamentData)
        tournament.tournament_id = "t1"

        mock_rounds = [MagicMock(is_final_round=True, round_id="r_final")]
        mock_tasks = [
            MagicMock(task_id="task_1"),
            MagicMock(task_id="task_2"),
            MagicMock(task_id="task_3"),
        ]

        # Contender wins 2/3 but loses 1 → boss retains
        scores_by_task = {
            "task_1": {EMISSION_BURN_HOTKEY: 3.0, "contender": 6.0},
            "task_2": {EMISSION_BURN_HOTKEY: 8.0, "contender": 5.0},  # Boss wins this one
            "task_3": {EMISSION_BURN_HOTKEY: 2.0, "contender": 5.0},
        }

        async def mock_get_scores(task_id, psql_db):
            return scores_by_task[task_id]

        with (
            patch("validator.tournament.round_results.get_tournament_rounds", return_value=mock_rounds),
            patch("validator.tournament.round_results.get_tournament_tasks", return_value=mock_tasks),
            patch("validator.tournament.round_results._get_scores_for_task", side_effect=mock_get_scores),
        ):
            result = await determine_env_tournament_winner(tournament, [], MagicMock(), MagicMock())

        # Boss retains — first in list
        assert result[0] == EMISSION_BURN_HOTKEY

    @pytest.mark.asyncio
    async def test_draws_are_allowed_when_contender_has_no_losses(self):
        """Contender wins when they have at least one win and no losses, even with a draw."""
        from validator.tournament.round_results import determine_env_tournament_winner

        tournament = MagicMock(spec=TournamentData)
        tournament.tournament_id = "t1"

        mock_rounds = [MagicMock(is_final_round=True, round_id="r_final")]
        mock_tasks = [
            MagicMock(task_id="task_1"),
            MagicMock(task_id="task_2"),
            MagicMock(task_id="task_3"),
        ]

        scores_by_task = {
            "task_1": {EMISSION_BURN_HOTKEY: 5.0, "contender": 5.0},  # Tie
            "task_2": {EMISSION_BURN_HOTKEY: 3.0, "contender": 6.0},
            "task_3": {EMISSION_BURN_HOTKEY: 2.0, "contender": 7.0},
        }

        async def mock_get_scores(task_id, psql_db):
            return scores_by_task[task_id]

        with (
            patch("validator.tournament.round_results.get_tournament_rounds", return_value=mock_rounds),
            patch("validator.tournament.round_results.get_tournament_tasks", return_value=mock_tasks),
            patch("validator.tournament.round_results._get_scores_for_task", side_effect=mock_get_scores),
        ):
            result = await determine_env_tournament_winner(tournament, [], MagicMock(), MagicMock())

        assert result[0] == "contender"

    @pytest.mark.asyncio
    async def test_no_final_round_boss_wins_default(self):
        from validator.tournament.round_results import determine_env_tournament_winner

        tournament = MagicMock(spec=TournamentData)
        tournament.tournament_id = "t1"

        # No rounds at all
        with patch("validator.tournament.round_results.get_tournament_rounds", return_value=[]):
            result = await determine_env_tournament_winner(tournament, [], MagicMock(), MagicMock())

        assert result[0] == EMISSION_BURN_HOTKEY
