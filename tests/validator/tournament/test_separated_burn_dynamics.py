"""
Test separated burn dynamics functionality.

This test file validates the new separated burn system that applies different
burn rates based on tournament participation and weekly task participation.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

import validator.scoring.constants as cts
from validator.scoring.models import PeriodScore
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
                hotkey="hotkey3", participated_in_text=True, participated_in_image=True, text_proportion=0.6, image_proportion=0.4
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
            text_burn_proportion=0.3,
            image_burn_proportion=0.1,
            text_tournament_weight=0.35,
            image_tournament_weight=0.36,
            text_regular_weight=0.59,
            image_regular_weight=0.52,
            burn_weight=0.18,
        )

    @pytest.fixture
    def sample_period_scores(self):
        """Sample period scores for testing."""
        return [
            PeriodScore(
                hotkey="hotkey1", normalised_score=0.8, weight_multiplier=1.0, average_score=0.75, std_score=0.1, period_tasks=10
            ),
            PeriodScore(
                hotkey="hotkey2", normalised_score=0.6, weight_multiplier=1.0, average_score=0.55, std_score=0.12, period_tasks=8
            ),
            PeriodScore(
                hotkey="hotkey3", normalised_score=0.9, weight_multiplier=1.0, average_score=0.85, std_score=0.08, period_tasks=12
            ),
        ]

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
        assert participation_both.text_proportion == cts.TOURNAMENT_TEXT_WEIGHT
        assert participation_both.image_proportion == cts.TOURNAMENT_IMAGE_WEIGHT

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
        assert sample_burn_data.text_burn_proportion == 0.3
        assert sample_burn_data.image_burn_proportion == 0.1
        assert sample_burn_data.text_tournament_weight == 0.35
        assert sample_burn_data.image_tournament_weight == 0.36

    def test_apply_tournament_weights(self, sample_tournament_participation):
        """Test tournament weight application with separated burn dynamics."""
        # Setup
        tournament_weights = {"hotkey1": 0.5, "hotkey2": 0.3, "hotkey3": 0.7}
        hotkey_to_node_id = {"hotkey1": 0, "hotkey2": 1, "hotkey3": 2}
        all_node_weights = [0.0, 0.0, 0.0]
        tournament_participation_map = {p.hotkey: p for p in sample_tournament_participation}
        scaled_text_tournament_weight = 0.35
        scaled_image_tournament_weight = 0.36

        # Apply weights
        apply_tournament_weights(
            tournament_weights,
            hotkey_to_node_id,
            all_node_weights,
            tournament_participation_map,
            scaled_text_tournament_weight,
            scaled_image_tournament_weight,
        )

        # Verify weights were applied
        assert all_node_weights[0] > 0  # hotkey1 (text only)
        assert all_node_weights[1] > 0  # hotkey2 (image only)
        assert all_node_weights[2] > 0  # hotkey3 (both)

        # hotkey3 participated in both tournaments, should have highest weight
        assert all_node_weights[2] > all_node_weights[0]
        assert all_node_weights[2] > all_node_weights[1]

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
        """Test that separated burn calculation produces different results than combined."""
        # Mock the dependencies
        with (
            pytest.mock.patch("validator.scoring.weights.get_latest_completed_tournament") as mock_get_tournament,
            pytest.mock.patch("validator.scoring.weights.calculate_performance_difference") as mock_calc_perf,
        ):
            # Setup mocks - different performance for text vs image
            mock_text_tournament = TournamentData(
                tournament_id="text_123",
                tournament_type=TournamentType.TEXT,
                status="completed",
                base_winner_hotkey="winner1",
                winner_hotkey="winner1",
            )
            mock_image_tournament = TournamentData(
                tournament_id="image_456",
                tournament_type=TournamentType.IMAGE,
                status="completed",
                base_winner_hotkey="winner2",
                winner_hotkey="winner2",
            )

            def mock_get_tournament_side_effect(psql_db, tournament_type):
                if tournament_type == TournamentType.TEXT:
                    return mock_text_tournament
                else:
                    return mock_image_tournament

            mock_get_tournament.side_effect = mock_get_tournament_side_effect

            def mock_calc_perf_side_effect(tournament_id, psql_db):
                if tournament_id == "text_123":
                    return 0.4  # Higher performance diff for text
                else:
                    return 0.1  # Lower performance diff for image

            mock_calc_perf.side_effect = mock_calc_perf_side_effect

            # Run separated burn calculation
            result = await get_tournament_burn_details(mock_psql_db)

            # Verify different burn rates were calculated
            assert result.text_performance_diff == 0.4
            assert result.image_performance_diff == 0.1
            assert result.text_burn_proportion > result.image_burn_proportion
            assert result.text_tournament_weight < result.image_tournament_weight  # Higher burn = lower weight

    def test_participation_proportion_calculation(self):
        """Test that participation proportions are calculated correctly."""
        # Test text-only participation
        text_only = HotkeyTournamentParticipation(
            hotkey="text_miner", participated_in_text=True, participated_in_image=False, text_proportion=1.0, image_proportion=0.0
        )
        assert text_only.text_proportion == 1.0
        assert text_only.image_proportion == 0.0

        # Test both tournaments participation
        both_tournaments = HotkeyTournamentParticipation(
            hotkey="both_miner",
            participated_in_text=True,
            participated_in_image=True,
            text_proportion=cts.TOURNAMENT_TEXT_WEIGHT,
            image_proportion=cts.TOURNAMENT_IMAGE_WEIGHT,
        )
        assert both_tournaments.text_proportion == cts.TOURNAMENT_TEXT_WEIGHT
        assert both_tournaments.image_proportion == cts.TOURNAMENT_IMAGE_WEIGHT
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
        # This would require mocking many dependencies, but demonstrates the integration point
        with (
            pytest.mock.patch("validator.scoring.weights.fetch_nodes") as mock_fetch_nodes,
            pytest.mock.patch("validator.scoring.weights.get_tournament_burn_details") as mock_burn_data,
            pytest.mock.patch("validator.scoring.weights.get_weekly_task_participation_data") as mock_weekly_part,
            pytest.mock.patch("validator.scoring.weights.get_active_tournament_participants") as mock_participants,
        ):
            # Setup basic mocks for integration test
            mock_substrate = MagicMock()
            mock_fetch_nodes.get_nodes_for_netuid.return_value = [
                MagicMock(hotkey="hotkey1", node_id=0),
                MagicMock(hotkey="hotkey2", node_id=1),
            ]

            mock_burn_data.return_value = TournamentBurnData(
                text_performance_diff=0.2,
                image_performance_diff=0.1,
                text_burn_proportion=0.2,
                image_burn_proportion=0.1,
                text_tournament_weight=0.4,
                image_tournament_weight=0.36,
                text_regular_weight=0.54,
                image_regular_weight=0.52,
                burn_weight=0.14,
            )

            mock_weekly_part.return_value = []
            mock_participants.return_value = []

            # Build tournament audit data
            tournament_audit_data = TournamentAuditData()
            tournament_audit_data.text_tournament_weight = 0.4
            tournament_audit_data.image_tournament_weight = 0.36
            tournament_audit_data.burn_weight = 0.14
            tournament_audit_data.participants = []

            # This should run without error and return a NodeWeightsResult
            result = await get_node_weights_from_tournament_audit_data(mock_substrate, 1, tournament_audit_data)

            assert isinstance(result, NodeWeightsResult)
            assert len(result.node_ids) == 2
            assert len(result.node_weights) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
