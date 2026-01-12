"""
Test script to demonstrate tournament burn mechanics with mock data.
This script creates mock tournaments and nodes to show how the burn/emission system works.
"""

import asyncio
from datetime import datetime
from datetime import timezone
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from dotenv import load_dotenv

from validator.core.weight_setting import get_tournament_burn_details


load_dotenv(".vali.env", override=True)

import validator.core.constants as cts
from core.models.tournament_models import TournamentBurnData
from core.models.tournament_models import TournamentType


class MockNode:
    """Mock node to simulate miners in the network"""

    def __init__(self, node_id: int, hotkey: str, incentive: float = 0.0):
        self.node_id = node_id
        self.hotkey = hotkey
        self.incentive = incentive


def create_mock_tournament(
    tournament_id: str,
    tournament_type: TournamentType,
    winner_hotkey: str,
    base_winner_hotkey: str,
    performance_diff: float,
    created_at: datetime = None,
) -> MagicMock:
    """Create a mock tournament with specified parameters"""
    tournament = MagicMock()
    tournament.tournament_id = tournament_id
    tournament.tournament_type = tournament_type
    tournament.winner_hotkey = winner_hotkey
    tournament.base_winner_hotkey = base_winner_hotkey
    tournament.winning_performance_difference = performance_diff
    tournament.created_at = created_at or datetime.now(timezone.utc)
    return tournament


def print_separator(title: str = ""):
    """Print a nice separator"""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}\n")
    else:
        print(f"{'=' * 80}\n")


def print_burn_data(burn_data: TournamentBurnData, scenario: str):
    """Pretty print burn data results"""
    print_separator(f"BURN DATA RESULTS - {scenario}")

    print("📊 Performance Differences:")
    print(
        f"   TEXT Performance: {burn_data.text_performance_diff:.2%}"
        if burn_data.text_performance_diff is not None
        else "   TEXT Performance: None"
    )
    print(
        f"   IMAGE Performance: {burn_data.image_performance_diff:.2%}"
        if burn_data.image_performance_diff is not None
        else "   IMAGE Performance: None"
    )

    print("\n💰 Weight Allocations:")
    print(f"   TEXT Tournament Weight:  {burn_data.text_tournament_weight:.6f} ({burn_data.text_tournament_weight * 100:.2f}%)")
    print(f"   IMAGE Tournament Weight: {burn_data.image_tournament_weight:.6f} ({burn_data.image_tournament_weight * 100:.2f}%)")
    print(f"   Burn Weight:             {burn_data.burn_weight:.6f} ({burn_data.burn_weight * 100:.2f}%)")

    total = burn_data.text_tournament_weight + burn_data.image_tournament_weight + burn_data.burn_weight
    print(f"\n   Total:                   {total:.6f} ({'✅' if abs(total - 1.0) < 0.0001 else '❌'})")

    print("\n🔥 Burn Proportions:")
    print(f"   TEXT Burn Proportion:  {burn_data.text_burn_proportion:.6f}")
    print(f"   IMAGE Burn Proportion: {burn_data.image_burn_proportion:.6f}")

    # Calculate emission increases
    text_base = cts.TOURNAMENT_TEXT_WEIGHT
    image_base = cts.TOURNAMENT_IMAGE_WEIGHT

    text_emission_increase = burn_data.text_tournament_weight - text_base
    image_emission_increase = burn_data.image_tournament_weight - image_base

    print("\n📈 Emission Changes (from base):")
    print(f"   TEXT Emission Change:  {text_emission_increase:+.6f} ({text_emission_increase * 100:+.2f}%)")
    print(f"   IMAGE Emission Change: {image_emission_increase:+.6f} ({image_emission_increase * 100:+.2f}%)")
    print(
        f"   Total Emission Change: {(text_emission_increase + image_emission_increase):+.6f} ({(text_emission_increase + image_emission_increase) * 100:+.2f}%)"
    )

    print()


async def scenario_1_perfect_performance():
    """Scenario 1: Both tournaments with perfect performance (0% difference)"""
    print_separator("SCENARIO 1: Perfect Performance")
    print("Both TEXT and IMAGE winners perform perfectly against synthetic tasks")
    print("Performance Difference: 0% for both")
    print("Expected: Base weights only, no emission increase\n")

    mock_db = AsyncMock()

    # Create mock tournaments with 0% performance difference
    text_tournament = create_mock_tournament(
        "text_2025_01", TournamentType.TEXT, "winner_text_1", "winner_text_1", performance_diff=0.0
    )

    image_tournament = create_mock_tournament(
        "image_2025_01", TournamentType.IMAGE, "winner_image_1", "winner_image_1", performance_diff=0.0
    )

    # Mock previous tournaments (same winners)
    prev_text = create_mock_tournament(
        "text_2024_12", TournamentType.TEXT, "winner_text_1", "winner_text_1", performance_diff=0.0
    )

    prev_image = create_mock_tournament(
        "image_2024_12", TournamentType.IMAGE, "winner_image_1", "winner_image_1", performance_diff=0.0
    )

    with patch("validator.core.weight_setting.get_latest_completed_tournament", side_effect=[text_tournament, image_tournament]):
        with patch("validator.core.weight_setting.get_latest_completed_tournament", side_effect=[prev_text, prev_image]):
            burn_data = await get_tournament_burn_details(mock_db)
            print_burn_data(burn_data, "Perfect Performance")


async def scenario_2_good_performance():
    """Scenario 2: Good performance (7% difference) - just above threshold"""
    print_separator("SCENARIO 2: Good Performance (Above 5% Threshold)")
    print("TEXT winner: 7% performance difference")
    print("IMAGE winner: 6% performance difference")
    print("Expected: Small emission increase (7%-5%)*2 and (6%-5%)*2\n")

    mock_db = AsyncMock()

    text_tournament = create_mock_tournament(
        "text_2025_01", TournamentType.TEXT, "winner_text_1", "winner_text_1", performance_diff=0.07
    )

    image_tournament = create_mock_tournament(
        "image_2025_01", TournamentType.IMAGE, "winner_image_1", "winner_image_1", performance_diff=0.06
    )

    prev_text = create_mock_tournament(
        "text_2024_12", TournamentType.TEXT, "winner_text_1", "winner_text_1", performance_diff=0.07
    )

    prev_image = create_mock_tournament(
        "image_2024_12", TournamentType.IMAGE, "winner_image_1", "winner_image_1", performance_diff=0.06
    )

    with patch("validator.core.weight_setting.get_latest_completed_tournament", side_effect=[text_tournament, image_tournament]):
        with patch("validator.core.weight_setting.get_latest_completed_tournament", side_effect=[prev_text, prev_image]):
            burn_data = await get_tournament_burn_details(mock_db)
            print_burn_data(burn_data, "Good Performance")


async def scenario_3_high_performance():
    """Scenario 3: High performance differences - major emission increase"""
    print_separator("SCENARIO 3: High Performance Differences")
    print("TEXT winner: 15% performance difference")
    print("IMAGE winner: 20% performance difference")
    print("Expected: Large emission increases\n")

    mock_db = AsyncMock()

    text_tournament = create_mock_tournament(
        "text_2025_01", TournamentType.TEXT, "winner_text_1", "winner_text_1", performance_diff=0.15
    )

    image_tournament = create_mock_tournament(
        "image_2025_01", TournamentType.IMAGE, "winner_image_1", "winner_image_1", performance_diff=0.20
    )

    prev_text = create_mock_tournament(
        "text_2024_12", TournamentType.TEXT, "winner_text_1", "winner_text_1", performance_diff=0.15
    )

    prev_image = create_mock_tournament(
        "image_2024_12", TournamentType.IMAGE, "winner_image_1", "winner_image_1", performance_diff=0.20
    )

    with patch("validator.core.weight_setting.get_latest_completed_tournament", side_effect=[text_tournament, image_tournament]):
        with patch("validator.core.weight_setting.get_latest_completed_tournament", side_effect=[prev_text, prev_image]):
            burn_data = await get_tournament_burn_details(mock_db)
            print_burn_data(burn_data, "High Performance")


async def scenario_4_new_winner():
    """Scenario 4: New winner arrives - performance recalculated"""
    print_separator("SCENARIO 4: New Winner Arrives")
    print("TEXT: New winner replaces old winner")
    print("Previous winner had 8% performance, new winner calculated at 12%")
    print("IMAGE: Same winner defends (6% performance)")
    print("Expected: Fresh calculation for TEXT, stored value for IMAGE\n")

    mock_db = AsyncMock()

    # TEXT has new winner
    text_tournament = create_mock_tournament(
        "text_2025_01",
        TournamentType.TEXT,
        "NEW_WINNER_TEXT",
        "NEW_WINNER_TEXT",
        performance_diff=0.08,  # Old stored value (will be ignored)
    )

    prev_text = create_mock_tournament(
        "text_2024_12", TournamentType.TEXT, "old_winner_text", "old_winner_text", performance_diff=0.08
    )

    # IMAGE same winner
    image_tournament = create_mock_tournament(
        "image_2025_01", TournamentType.IMAGE, "winner_image_1", "winner_image_1", performance_diff=0.06
    )

    prev_image = create_mock_tournament(
        "image_2024_12", TournamentType.IMAGE, "winner_image_1", "winner_image_1", performance_diff=0.06
    )

    with patch("validator.core.weight_setting.get_latest_completed_tournament", side_effect=[text_tournament, image_tournament]):
        with patch("validator.core.weight_setting.get_latest_completed_tournament", side_effect=[prev_text, prev_image]):
            # Mock fresh calculation for new TEXT winner
            with patch("validator.core.weight_setting.calculate_performance_difference", return_value=0.12) as mock_calc:
                burn_data = await get_tournament_burn_details(mock_db)

                print(f"🔍 Performance calculation called: {mock_calc.called}")
                print(f"   (Should be True - new winner needs fresh calculation)\n")

                print_burn_data(burn_data, "New Winner Scenario")


async def scenario_5_only_text_tournament():
    """Scenario 5: Only TEXT tournament exists"""
    print_separator("SCENARIO 5: Only TEXT Tournament")
    print("TEXT tournament completed with 10% performance")
    print("IMAGE tournament does not exist yet")
    print("Expected: TEXT gets emission increase, IMAGE gets base weight\n")

    mock_db = AsyncMock()

    text_tournament = create_mock_tournament(
        "text_2025_01", TournamentType.TEXT, "winner_text_1", "winner_text_1", performance_diff=0.10
    )

    prev_text = create_mock_tournament(
        "text_2024_12", TournamentType.TEXT, "winner_text_1", "winner_text_1", performance_diff=0.10
    )

    with patch(
        "validator.core.weight_setting.get_latest_completed_tournament", side_effect=[text_tournament, None]
    ):  # None for IMAGE
        with patch("validator.core.weight_setting.get_latest_completed_tournament", return_value=prev_text):
            burn_data = await get_tournament_burn_details(mock_db)
            print_burn_data(burn_data, "Only TEXT Tournament")


async def scenario_6_below_threshold():
    """Scenario 6: Performance below threshold - no emission increase"""
    print_separator("SCENARIO 6: Below 5% Threshold")
    print("TEXT winner: 3% performance difference (below threshold)")
    print("IMAGE winner: 4% performance difference (below threshold)")
    print("Expected: Base weights only, no emission increase\n")

    mock_db = AsyncMock()

    text_tournament = create_mock_tournament(
        "text_2025_01", TournamentType.TEXT, "winner_text_1", "winner_text_1", performance_diff=0.03
    )

    image_tournament = create_mock_tournament(
        "image_2025_01", TournamentType.IMAGE, "winner_image_1", "winner_image_1", performance_diff=0.04
    )

    prev_text = create_mock_tournament(
        "text_2024_12", TournamentType.TEXT, "winner_text_1", "winner_text_1", performance_diff=0.03
    )

    prev_image = create_mock_tournament(
        "image_2024_12", TournamentType.IMAGE, "winner_image_1", "winner_image_1", performance_diff=0.04
    )

    with patch("validator.core.weight_setting.get_latest_completed_tournament", side_effect=[text_tournament, image_tournament]):
        with patch("validator.core.weight_setting.get_latest_completed_tournament", side_effect=[prev_text, prev_image]):
            burn_data = await get_tournament_burn_details(mock_db)
            print_burn_data(burn_data, "Below Threshold")


def print_constants():
    """Print current configuration constants"""
    print_separator("CONFIGURATION CONSTANTS")
    print(f"📋 Base Configuration:")
    print(f"   TOURNAMENT_TEXT_WEIGHT:        {cts.TOURNAMENT_TEXT_WEIGHT:.2f} ({cts.TOURNAMENT_TEXT_WEIGHT * 100:.0f}%)")
    print(f"   TOURNAMENT_IMAGE_WEIGHT:       {cts.TOURNAMENT_IMAGE_WEIGHT:.2f} ({cts.TOURNAMENT_IMAGE_WEIGHT * 100:.0f}%)")
    print(f"   MAX_TEXT_TOURNAMENT_WEIGHT:    {cts.MAX_TEXT_TOURNAMENT_WEIGHT:.2f} ({cts.MAX_TEXT_TOURNAMENT_WEIGHT * 100:.0f}%)")
    print(f"   MAX_IMAGE_TOURNAMENT_WEIGHT:   {cts.MAX_IMAGE_TOURNAMENT_WEIGHT:.2f} ({cts.MAX_IMAGE_TOURNAMENT_WEIGHT * 100:.0f}%)")
    print(f"   EMISSION_MULTIPLIER_THRESHOLD: {cts.EMISSION_MULTIPLIER_THRESHOLD:.2%}")
    print(f"   EMISSION_BOOST_DECAY_PER_WIN:  {cts.EMISSION_BOOST_DECAY_PER_WIN:.2%}")
    print(f"\n📐 Base Weights:")
    text_base = cts.TOURNAMENT_TEXT_WEIGHT
    image_base = cts.TOURNAMENT_IMAGE_WEIGHT
    print(f"   TEXT Base Weight:  {text_base:.4f} ({text_base * 100:.2f}%)")
    print(f"   IMAGE Base Weight: {image_base:.4f} ({image_base * 100:.2f}%)")
    print(f"   BURN Base Weight:  {1.0 - text_base - image_base:.4f} ({(1.0 - text_base - image_base) * 100:.2f}%)")
    print(f"\n💡 Emission Multiplier Formula:")
    print(f"   If performance > {cts.EMISSION_MULTIPLIER_THRESHOLD:.1%}:")
    print(f"   emission_increase = (performance - {cts.EMISSION_MULTIPLIER_THRESHOLD:.1%}) × 2.0")
    print(f"   emission_increase -= max(0, consecutive_wins - 1) × {cts.EMISSION_BOOST_DECAY_PER_WIN:.1%}")
    print()


async def main():
    """Run all test scenarios"""
    print("\n" + "🔥" * 40)
    print("   TOURNAMENT BURN MECHANICS - TEST SCENARIOS")
    print("🔥" * 40 + "\n")

    print_constants()

    # Run all scenarios
    await scenario_1_perfect_performance()
    await scenario_2_good_performance()
    await scenario_3_high_performance()
    await scenario_4_new_winner()
    await scenario_5_only_text_tournament()
    await scenario_6_below_threshold()

    print_separator("TEST COMPLETE")
    print("✅ All scenarios executed successfully!")
    print("\nKey Takeaways:")
    print("1. Performance < 5% threshold → Base weights only")
    print("2. Performance > 5% threshold → Emission increase = (perf - 5%) × 2.0")
    print("3. New winner → Fresh performance calculation")
    print("4. Same winner defending → Uses stored performance")
    print("5. TEXT and IMAGE calculated separately")
    print("6. Total weights always sum to 1.0")
    print()


if __name__ == "__main__":
    asyncio.run(main())
