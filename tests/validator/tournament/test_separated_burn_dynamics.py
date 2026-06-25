"""
Test separated burn dynamics functionality.

This test file validates the new separated burn system that applies different
burn rates based on tournament participation and weekly task participation.
"""

from datetime import datetime
from datetime import timezone
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import validator.scoring.constants as cts
from validator.scoring.weights import apply_tournament_weights
from validator.scoring.weights import get_node_weights_from_tournament_audit_data
from validator.scoring.weights import get_tournament_burn_details
from validator.tournament.models import HotkeyTaskParticipation
from validator.tournament.models import HotkeyTournamentParticipation
from validator.tournament.models import NodeWeightsResult
from validator.tournament.models import TournamentAuditData
from validator.tournament.models import TournamentBurnData
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentType


class TestSeparatedBurnDynamics:
    """Test cases for separated burn dynamics functionality."""

    @pytest.fixture
    def mock_psql_db(self):
        """Mock database connection."""
        return AsyncMock()

    @pytest.fixture
    def sample_tournament_participation(self):
        """Sample tournament participation data."""
        return [
            HotkeyTournamentParticipation(
                hotkey="hotkey1",
                participated_in_text=True,
                participated_in_image=False,
                text_proportion=1.0,
                image_proportion=0.0,
            ),
            HotkeyTournamentParticipation(
                hotkey="hotkey2",
                participated_in_text=False,
                participated_in_image=True,
                text_proportion=0.0,
                image_proportion=1.0,
            ),
            HotkeyTournamentParticipation(
                hotkey="hotkey3",
                participated_in_text=True,
                participated_in_image=True,
                text_proportion=0.6,
                image_proportion=0.4,
            ),
        ]

    @pytest.fixture
    def sample_weekly_participation(self):
        """Sample weekly task participation data."""
        return [
            HotkeyTaskParticipation(hotkey="hotkey1", text_task_proportion=0.8, image_task_proportion=0.2, total_tasks=50),
            HotkeyTaskParticipation(hotkey="hotkey2", text_task_proportion=0.3, image_task_proportion=0.7, total_tasks=30),
            HotkeyTaskParticipation(hotkey="hotkey3", text_task_proportion=0.5, image_task_proportion=0.5, total_tasks=40),
        ]

    @pytest.fixture
    def sample_burn_data(self):
        """Sample separated burn data."""
        return TournamentBurnData(
            text_performance_diff=0.3,
            image_performance_diff=0.1,
            environment_performance_diff=0.2,
            text_burn_proportion=0.3,
            image_burn_proportion=0.1,
            environment_burn_proportion=0.0625,
            text_tournament_weight=0.35,
            image_tournament_weight=0.36,
            environment_tournament_weight=0.15,
            burn_weight=0.14,
        )

    @pytest.mark.asyncio
    async def test_tournament_participation_data_structure(self, sample_tournament_participation):
        """Test that tournament participation data has correct structure."""
        participation = sample_tournament_participation[0]

        assert participation.hotkey == "hotkey1"
        assert participation.participated_in_text is True
        assert participation.participated_in_image is False
        assert participation.text_proportion == 1.0
        assert participation.image_proportion == 0.0

        # Test both tournament participation
        participation_both = sample_tournament_participation[2]
        assert participation_both.participated_in_text is True
        assert participation_both.participated_in_image is True
        assert participation_both.text_proportion == 0.6
        assert participation_both.image_proportion == 0.4

    @pytest.mark.asyncio
    async def test_weekly_participation_data_structure(self, sample_weekly_participation):
        """Test that weekly participation data has correct structure."""
        participation = sample_weekly_participation[0]

        assert participation.hotkey == "hotkey1"
        assert participation.text_task_proportion == 0.8
        assert participation.image_task_proportion == 0.2
        assert participation.total_tasks == 50
        # Verify proportions sum to 1.0
        assert abs((participation.text_task_proportion + participation.image_task_proportion) - 1.0) < 0.001

    def test_burn_data_separated_structure(self, sample_burn_data):
        """Test that separated burn data has correct structure."""
        assert sample_burn_data.text_performance_diff == 0.3
        assert sample_burn_data.image_performance_diff == 0.1
        assert sample_burn_data.environment_performance_diff == 0.2
        assert sample_burn_data.text_burn_proportion == 0.3
        assert sample_burn_data.image_burn_proportion == 0.1
        assert sample_burn_data.environment_burn_proportion == 0.0625
        assert sample_burn_data.text_tournament_weight == 0.35
        assert sample_burn_data.image_tournament_weight == 0.36
        assert sample_burn_data.environment_tournament_weight == 0.15

    def test_apply_tournament_weights(self):
        """Test tournament weight application with separated burn dynamics."""
        text_tournament_weights = {"text_winner": 0.7, "shared": 0.3}
        image_tournament_weights = {"image_winner": 1.0}
        environment_tournament_weights = {"env_winner": 1.0}
        hotkey_to_node_id = {"text_winner": 0, "shared": 1, "image_winner": 2, "env_winner": 3}
        all_node_weights = [0.0, 0.0, 0.0, 0.0]

        undistributed_weight = apply_tournament_weights(
            text_tournament_weights,
            image_tournament_weights,
            environment_tournament_weights,
            hotkey_to_node_id,
            all_node_weights,
            scaled_text_tournament_weight=0.35,
            scaled_image_tournament_weight=0.36,
            scaled_environment_tournament_weight=0.2,
            scaled_text_base_weight=0.15,
            scaled_image_base_weight=0.125,
            scaled_environment_base_weight=0.15,
            text_winner_hotkey="text_winner",
            image_winner_hotkey="image_winner",
            environment_winner_hotkey="env_winner",
        )

        assert all_node_weights == pytest.approx([0.245, 0.045, 0.36, 0.2])
        assert undistributed_weight == pytest.approx(0.06)

    def test_node_weights_result_model(self):
        """Test NodeWeightsResult model functionality."""
        node_ids = [0, 1, 2]
        node_weights = [0.5, 0.3, 0.7]

        result = NodeWeightsResult(node_ids=node_ids, node_weights=node_weights)

        assert result.node_ids == node_ids
        assert result.node_weights == node_weights

        # Test tuple conversion for backward compatibility
        tuple_result = result.to_tuple()
        assert tuple_result == (node_ids, node_weights)
        assert isinstance(tuple_result, tuple)

    @pytest.mark.asyncio
    async def test_separated_burn_calculation_logic(self, mock_psql_db):
        """Test that separated burn calculation tracks each tournament type independently."""
        championship_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        with (
            patch("validator.scoring.weights.get_latest_completed_tournament") as mock_get_tournament,
            patch("validator.scoring.weights.calculate_performance_difference") as mock_calc_perf,
            patch("validator.scoring.weights.count_champion_consecutive_wins") as mock_count_wins,
            patch("validator.scoring.weights.get_tournament_where_champion_first_won") as mock_first_win,
        ):
            mock_text_tournament = TournamentData(
                tournament_id="text_123",
                tournament_type=TournamentType.TEXT,
                status="completed",
                base_winner_hotkey="winner1",
                winner_hotkey="winner1",
                updated_at=championship_time,
            )
            mock_image_tournament = TournamentData(
                tournament_id="image_456",
                tournament_type=TournamentType.IMAGE,
                status="completed",
                base_winner_hotkey="winner2",
                winner_hotkey="winner2",
                updated_at=championship_time,
            )

            def mock_get_tournament_side_effect(psql_db, tournament_type, exclude_tournament_id=None):
                if exclude_tournament_id is not None:
                    return None
                if tournament_type == TournamentType.TEXT:
                    return mock_text_tournament
                if tournament_type == TournamentType.IMAGE:
                    return mock_image_tournament
                return None

            mock_get_tournament.side_effect = mock_get_tournament_side_effect

            def mock_calc_perf_side_effect(tournament_id, psql_db):
                if tournament_id == "text_123":
                    return 0.4
                if tournament_id == "image_456":
                    return 0.1
                return 0.0

            mock_calc_perf.side_effect = mock_calc_perf_side_effect
            mock_count_wins.return_value = 1
            mock_first_win.return_value = TournamentData(
                tournament_id="first_win",
                tournament_type=TournamentType.TEXT,
                status="completed",
                winner_hotkey="winner1",
                updated_at=championship_time,
            )

            result = await get_tournament_burn_details(mock_psql_db)

            assert result.text_performance_diff == 0.4
            assert result.image_performance_diff == 0.1
            assert result.environment_performance_diff is None
            assert result.text_tournament_weight <= cts.MAX_TEXT_TOURNAMENT_WEIGHT
            assert result.image_tournament_weight <= cts.MAX_IMAGE_TOURNAMENT_WEIGHT
            assert result.environment_tournament_weight == min(
                cts.TOURNAMENT_ENVIRONMENT_WEIGHT,
                cts.MAX_ENVIRONMENT_TOURNAMENT_WEIGHT,
            )
            assert result.burn_weight == pytest.approx(
                1.0 - result.text_tournament_weight - result.image_tournament_weight - result.environment_tournament_weight
            )

    def test_participation_proportion_calculation(self):
        """Test that participation proportions are calculated correctly."""
        # Test text-only participation
        text_only = HotkeyTournamentParticipation(
            hotkey="text_miner",
            participated_in_text=True,
            participated_in_image=False,
            text_proportion=1.0,
            image_proportion=0.0,
        )
        assert text_only.text_proportion == 1.0
        assert text_only.image_proportion == 0.0

        # Test both tournaments participation
        both_tournaments = HotkeyTournamentParticipation(
            hotkey="both_miner",
            participated_in_text=True,
            participated_in_image=True,
            text_proportion=0.6,
            image_proportion=0.4,
        )
        assert both_tournaments.text_proportion == 0.6
        assert both_tournaments.image_proportion == 0.4
        # Verify proportions sum to 1.0
        assert abs((both_tournaments.text_proportion + both_tournaments.image_proportion) - 1.0) < 0.001

    def test_edge_case_no_participation(self):
        """Test handling of hotkeys with no participation data."""
        # Test weekly participation with zero tasks
        zero_tasks = HotkeyTaskParticipation(
            hotkey="inactive", text_task_proportion=0.0, image_task_proportion=0.0, total_tasks=0
        )
        assert zero_tasks.total_tasks == 0
        assert zero_tasks.text_task_proportion == 0.0
        assert zero_tasks.image_task_proportion == 0.0

    @pytest.mark.asyncio
    async def test_integration_separated_weight_calculation(self, mock_psql_db):
        """Integration test for the full separated weight calculation."""
        with patch("validator.scoring.weights.fetch_nodes") as mock_fetch_nodes:
            # Setup basic mocks for integration test
            mock_substrate = MagicMock()
            mock_fetch_nodes._get_nodes_for_uid.return_value = [
                MagicMock(hotkey="hotkey1", node_id=0),
                MagicMock(hotkey="hotkey2", node_id=1),
            ]

            # Build tournament audit data
            tournament_audit_data = TournamentAuditData()
            tournament_audit_data.text_tournament_weight = 0.4
            tournament_audit_data.image_tournament_weight = 0.36
            tournament_audit_data.environment_tournament_weight = 0.1
            tournament_audit_data.burn_weight = 0.14
            tournament_audit_data.participants = []

            # This should run without error and return a NodeWeightsResult
            result = await get_node_weights_from_tournament_audit_data(mock_substrate, 1, tournament_audit_data)

            assert isinstance(result, NodeWeightsResult)
            assert len(result.node_ids) == 2
            assert len(result.node_weights) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
