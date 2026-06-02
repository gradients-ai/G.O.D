from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import validator.scoring.constants as cts
from validator.scoring.weights import calculate_emission_boost_from_perf
from validator.scoring.weights import get_tournament_burn_details


class TestTournamentBurnSeparated:
    @pytest.fixture
    def mock_psql_db(self):
        return AsyncMock()

    # ========== Test calculate_emission_boost_from_perf ==========

    def test_calculate_emission_boost_from_perf_below_threshold(self):
        """Test emission multiplier with performance below threshold"""
        result = calculate_emission_boost_from_perf(0.03)  # 3% below 5% threshold
        assert result == 0.0

    def test_calculate_emission_boost_from_perf_at_threshold(self):
        """Test emission multiplier at exact threshold"""
        result = calculate_emission_boost_from_perf(cts.EMISSION_MULTIPLIER_THRESHOLD)
        assert result == 0.0

    def test_calculate_emission_boost_from_perf_above_threshold(self):
        """Test emission multiplier above threshold"""
        # 10% performance, 5% threshold -> 5% excess * 2.0 = 10% emission increase
        result = calculate_emission_boost_from_perf(0.10)
        expected = (0.10 - cts.EMISSION_MULTIPLIER_THRESHOLD) * 2.0
        assert abs(result - expected) < 0.0001

    def test_calculate_emission_boost_from_perf_high_performance(self):
        """Test emission multiplier with high performance difference"""
        # 20% performance, 5% threshold -> 15% excess * 2.0 = 30% emission increase
        result = calculate_emission_boost_from_perf(0.20)
        expected = (0.20 - cts.EMISSION_MULTIPLIER_THRESHOLD) * 2.0
        assert abs(result - expected) < 0.0001

    @pytest.mark.asyncio
    async def test_get_burn_details_separated_no_tournaments(self, mock_psql_db):
        """Test burn details with no tournament data at all"""
        with patch("validator.scoring.weights.get_latest_completed_tournament", return_value=None):
            result = await get_tournament_burn_details(mock_psql_db)

            # With no tournaments, both should be None
            assert result.text_performance_diff is None
            assert result.image_performance_diff is None

            # Base weights are still allocated (no emission increase)
            text_base = cts.TOURNAMENT_TEXT_WEIGHT
            image_base = cts.TOURNAMENT_IMAGE_WEIGHT

            assert abs(result.text_tournament_weight - text_base) < 0.0001
            assert abs(result.image_tournament_weight - image_base) < 0.0001
            assert abs(result.burn_weight - (1.0 - text_base - image_base)) < 0.0001

    @pytest.mark.asyncio
    async def test_new_winner_calculates_fresh_performance(self, mock_psql_db):
        """Test that performance difference is calculated fresh when winner changes"""
        mock_latest = MagicMock()
        mock_latest.tournament_id = "latest_tournament"
        mock_latest.winner_hotkey = "new_winner"
        mock_latest.winning_performance_difference = 0.08  # Stored old value

        mock_previous = MagicMock()
        mock_previous.winner_hotkey = "old_winner"  # Different winner
        mock_previous.winning_performance_difference = 0.08

        with patch("validator.scoring.weights.get_latest_completed_tournament", side_effect=[mock_latest, None]):
            with patch("validator.scoring.weights.get_latest_completed_tournament", return_value=mock_previous):
                with patch("validator.scoring.weights.calculate_performance_difference", return_value=0.12) as mock_calc:
                    result = await get_tournament_burn_details(mock_psql_db)

                    # Should calculate fresh performance because winner changed
                    assert mock_calc.called
                    assert result.text_performance_diff == 0.12  # Uses freshly calculated value

    @pytest.mark.asyncio
    async def test_same_winner_uses_stored_performance(self, mock_psql_db):
        """Test that stored performance difference is used when winner stays the same"""
        mock_latest_text = MagicMock()
        mock_latest_text.tournament_id = "latest_text_tournament"
        mock_latest_text.winner_hotkey = "same_winner"
        mock_latest_text.winning_performance_difference = 0.08  # Stored value

        mock_previous_text = MagicMock()
        mock_previous_text.winner_hotkey = "same_winner"  # Same winner
        mock_previous_text.winning_performance_difference = 0.06

        mock_latest_image = MagicMock()
        mock_latest_image.tournament_id = "latest_image_tournament"
        mock_latest_image.winner_hotkey = "image_winner"
        mock_latest_image.winning_performance_difference = 0.10

        mock_previous_image = MagicMock()
        mock_previous_image.winner_hotkey = "image_winner"
        mock_previous_image.winning_performance_difference = 0.09

        with patch(
            "validator.scoring.weights.get_latest_completed_tournament",
            side_effect=[mock_latest_text, mock_latest_image],
        ):
            with patch(
                "validator.scoring.weights.get_latest_completed_tournament",
                side_effect=[mock_previous_text, mock_previous_image],
            ):
                with patch("validator.scoring.weights.calculate_performance_difference") as mock_calc:
                    result = await get_tournament_burn_details(mock_psql_db)

                    # Should NOT calculate fresh performance - uses stored values
                    assert not mock_calc.called
                    assert result.text_performance_diff == 0.08
                    assert result.image_performance_diff == 0.10

    @pytest.mark.asyncio
    async def test_first_winner_calculates_performance(self, mock_psql_db):
        """Test that performance is calculated for first-time winner"""
        mock_latest = MagicMock()
        mock_latest.tournament_id = "first_tournament"
        mock_latest.winner_hotkey = "first_winner"
        mock_latest.winning_performance_difference = None

        with patch("validator.scoring.weights.get_latest_completed_tournament", side_effect=[mock_latest, None]):
            with patch("validator.scoring.weights.get_latest_completed_tournament", return_value=None):
                with patch("validator.scoring.weights.calculate_performance_difference", return_value=0.07) as mock_calc:
                    result = await get_tournament_burn_details(mock_psql_db)

                    # Should calculate fresh performance for first winner
                    assert mock_calc.called
                    assert result.text_performance_diff == 0.07

    @pytest.mark.asyncio
    async def test_only_text_tournament_completed(self, mock_psql_db):
        """Test burn calculation with only TEXT tournament completed"""
        mock_text = MagicMock()
        mock_text.tournament_id = "text_tournament"
        mock_text.winner_hotkey = "text_winner"
        mock_text.winning_performance_difference = 0.08  # 8% performance

        mock_previous_text = MagicMock()
        mock_previous_text.winner_hotkey = "text_winner"
        mock_previous_text.winning_performance_difference = 0.08

        with patch("validator.scoring.weights.get_latest_completed_tournament", side_effect=[mock_text, None]):
            with patch("validator.scoring.weights.get_latest_completed_tournament", return_value=mock_previous_text):
                result = await get_tournament_burn_details(mock_psql_db)

                # TEXT tournament exists
                assert result.text_performance_diff == 0.08
                # IMAGE tournament doesn't exist
                assert result.image_performance_diff is None

                # 8% performance > 5% threshold -> 3% excess * 2.0 = 6% emission increase
                text_emission_increase = (0.08 - cts.EMISSION_MULTIPLIER_THRESHOLD) * 2.0
                text_base = cts.TOURNAMENT_TEXT_WEIGHT
                expected_text_weight = text_base + text_emission_increase

                # IMAGE base weight is still allocated (no emission increase)
                image_base = cts.TOURNAMENT_IMAGE_WEIGHT

                assert abs(result.text_tournament_weight - expected_text_weight) < 0.0001
                assert abs(result.image_tournament_weight - image_base) < 0.0001
                assert abs(result.burn_weight - (1.0 - expected_text_weight - image_base)) < 0.0001

    @pytest.mark.asyncio
    async def test_only_image_tournament_completed(self, mock_psql_db):
        """Test burn calculation with only IMAGE tournament completed"""
        mock_image = MagicMock()
        mock_image.tournament_id = "image_tournament"
        mock_image.winner_hotkey = "image_winner"
        mock_image.winning_performance_difference = 0.12  # 12% performance

        mock_previous_image = MagicMock()
        mock_previous_image.winner_hotkey = "image_winner"
        mock_previous_image.winning_performance_difference = 0.12

        with patch("validator.scoring.weights.get_latest_completed_tournament", side_effect=[None, mock_image]):
            with patch("validator.scoring.weights.get_latest_completed_tournament", return_value=mock_previous_image):
                result = await get_tournament_burn_details(mock_psql_db)

                # TEXT tournament doesn't exist
                assert result.text_performance_diff is None
                # IMAGE tournament exists
                assert result.image_performance_diff == 0.12

                # 12% performance > 5% threshold -> 7% excess * 2.0 = 14% emission increase
                image_emission_increase = (0.12 - cts.EMISSION_MULTIPLIER_THRESHOLD) * 2.0
                image_base = cts.TOURNAMENT_IMAGE_WEIGHT
                expected_image_weight = image_base + image_emission_increase

                # TEXT base weight is still allocated (no emission increase)
                text_base = cts.TOURNAMENT_TEXT_WEIGHT

                assert abs(result.text_tournament_weight - text_base) < 0.0001
                assert abs(result.image_tournament_weight - expected_image_weight) < 0.0001
                assert abs(result.burn_weight - (1.0 - expected_image_weight - text_base)) < 0.0001

    @pytest.mark.asyncio
    async def test_both_tournaments_completed(self, mock_psql_db):
        """Test burn calculation with both TEXT and IMAGE tournaments completed"""
        mock_text = MagicMock()
        mock_text.tournament_id = "text_tournament"
        mock_text.winner_hotkey = "text_winner"
        mock_text.winning_performance_difference = 0.08

        mock_previous_text = MagicMock()
        mock_previous_text.winner_hotkey = "text_winner"
        mock_previous_text.winning_performance_difference = 0.08

        mock_image = MagicMock()
        mock_image.tournament_id = "image_tournament"
        mock_image.winner_hotkey = "image_winner"
        mock_image.winning_performance_difference = 0.10

        mock_previous_image = MagicMock()
        mock_previous_image.winner_hotkey = "image_winner"
        mock_previous_image.winning_performance_difference = 0.10

        with patch("validator.scoring.weights.get_latest_completed_tournament", side_effect=[mock_text, mock_image]):
            with patch(
                "validator.scoring.weights.get_latest_completed_tournament",
                side_effect=[mock_previous_text, mock_previous_image],
            ):
                result = await get_tournament_burn_details(mock_psql_db)

                assert result.text_performance_diff == 0.08
                assert result.image_performance_diff == 0.10

                # Calculate expected weights
                text_emission = (0.08 - cts.EMISSION_MULTIPLIER_THRESHOLD) * 2.0
                image_emission = (0.10 - cts.EMISSION_MULTIPLIER_THRESHOLD) * 2.0

                text_base = cts.TOURNAMENT_TEXT_WEIGHT
                image_base = cts.TOURNAMENT_IMAGE_WEIGHT

                expected_text_weight = text_base + text_emission
                expected_image_weight = image_base + image_emission
                expected_burn_weight = 1.0 - expected_text_weight - expected_image_weight

                assert abs(result.text_tournament_weight - expected_text_weight) < 0.0001
                assert abs(result.image_tournament_weight - expected_image_weight) < 0.0001
                assert abs(result.burn_weight - expected_burn_weight) < 0.0001

    # ========== Test burn account won scenarios ==========

    @pytest.mark.asyncio
    async def test_burn_account_winner_with_calculated_performance(self, mock_psql_db):
        """Test when burn account wins with calculated performance data"""
        mock_text = MagicMock()
        mock_text.tournament_id = "text_tournament"
        mock_text.winner_hotkey = cts.EMISSION_BURN_HOTKEY
        mock_text.winning_performance_difference = 0.25

        mock_previous = MagicMock()
        mock_previous.winner_hotkey = cts.EMISSION_BURN_HOTKEY
        mock_previous.winning_performance_difference = 0.25

        with patch("validator.scoring.weights.get_latest_completed_tournament", side_effect=[mock_text, None]):
            with patch("validator.scoring.weights.get_latest_completed_tournament", return_value=mock_previous):
                result = await get_tournament_burn_details(mock_psql_db)

                # Burn account won, uses stored performance
                assert result.text_performance_diff == 0.25

                # High performance difference -> large emission increase
                text_emission = (0.25 - cts.EMISSION_MULTIPLIER_THRESHOLD) * 2.0
                text_base = cts.TOURNAMENT_TEXT_WEIGHT
                expected_text_weight = text_base + text_emission

                assert abs(result.text_tournament_weight - expected_text_weight) < 0.0001

    @pytest.mark.asyncio
    async def test_winner_with_zero_performance_difference(self, mock_psql_db):
        """Test when winner performs perfectly (0% difference)"""
        mock_text = MagicMock()
        mock_text.tournament_id = "text_tournament"
        mock_text.winner_hotkey = "perfect_winner"
        mock_text.winning_performance_difference = 0.0

        mock_previous = MagicMock()
        mock_previous.winner_hotkey = "perfect_winner"
        mock_previous.winning_performance_difference = 0.0

        with patch("validator.scoring.weights.get_latest_completed_tournament", side_effect=[mock_text, None]):
            with patch("validator.scoring.weights.get_latest_completed_tournament", return_value=mock_previous):
                result = await get_tournament_burn_details(mock_psql_db)

                # Perfect performance -> no emission increase
                assert result.text_performance_diff == 0.0

                text_base = cts.TOURNAMENT_TEXT_WEIGHT
                assert abs(result.text_tournament_weight - text_base) < 0.0001

    # ========== Test performance below threshold (no emission increase) ==========

    @pytest.mark.asyncio
    async def test_performance_below_threshold_no_emission_increase(self, mock_psql_db):
        """Test that performance below 5% threshold doesn't increase emissions"""
        mock_text = MagicMock()
        mock_text.tournament_id = "text_tournament"
        mock_text.winner_hotkey = "text_winner"
        mock_text.winning_performance_difference = 0.03  # 3% < 5% threshold

        mock_previous = MagicMock()
        mock_previous.winner_hotkey = "text_winner"
        mock_previous.winning_performance_difference = 0.03

        with patch("validator.scoring.weights.get_latest_completed_tournament", side_effect=[mock_text, None]):
            with patch("validator.scoring.weights.get_latest_completed_tournament", return_value=mock_previous):
                result = await get_tournament_burn_details(mock_psql_db)

                # Performance below threshold -> no emission increase for TEXT
                text_base = cts.TOURNAMENT_TEXT_WEIGHT
                image_base = cts.TOURNAMENT_IMAGE_WEIGHT

                assert abs(result.text_tournament_weight - text_base) < 0.0001
                assert abs(result.image_tournament_weight - image_base) < 0.0001
                assert abs(result.burn_weight - (1.0 - text_base - image_base)) < 0.0001

    # ========== Test weight distribution sums to 1.0 ==========

    @pytest.mark.asyncio
    async def test_weights_sum_to_one(self, mock_psql_db):
        """Test that all weights sum to 1.0"""
        mock_text = MagicMock()
        mock_text.tournament_id = "text_tournament"
        mock_text.winner_hotkey = "text_winner"
        mock_text.winning_performance_difference = 0.15

        mock_previous_text = MagicMock()
        mock_previous_text.winner_hotkey = "text_winner"
        mock_previous_text.winning_performance_difference = 0.15

        mock_image = MagicMock()
        mock_image.tournament_id = "image_tournament"
        mock_image.winner_hotkey = "image_winner"
        mock_image.winning_performance_difference = 0.12

        mock_previous_image = MagicMock()
        mock_previous_image.winner_hotkey = "image_winner"
        mock_previous_image.winning_performance_difference = 0.12

        with patch("validator.scoring.weights.get_latest_completed_tournament", side_effect=[mock_text, mock_image]):
            with patch(
                "validator.scoring.weights.get_latest_completed_tournament",
                side_effect=[mock_previous_text, mock_previous_image],
            ):
                result = await get_tournament_burn_details(mock_psql_db)

                total_weight = result.text_tournament_weight + result.image_tournament_weight + result.burn_weight
                assert abs(total_weight - 1.0) < 0.0001


# NOTE: Tests for the deprecated get_tournament_burn_details function have been removed.
# The function was removed as part of the migration to separated burn dynamics.
# All relevant test coverage is provided by TestTournamentBurnSeparated above.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
