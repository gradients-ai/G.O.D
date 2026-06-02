#!/usr/bin/env python3
"""
Test script for the hybrid decay system.

Tests various scenarios:
1. Before cutoff date (old system only)
2. After cutoff, champion won before cutoff (hybrid system)
3. After cutoff, champion won after cutoff (new system only)
4. Edge cases
"""

from datetime import date
from datetime import datetime
from datetime import timezone


# Constants matching validator/scoring/constants.py
EMISSION_BOOST_DECAY_PER_WIN = 0.01  # 1% per win (old system)
EMISSION_DAILY_TIME_DECAY_RATE = 0.0033  # 0.33% per day (new system)
EMISSION_TIME_DECAY_START_DATE = date(2025, 11, 26)
SECONDS_PER_DAY = 86400.0
TOURNAMENT_TEXT_WEIGHT = 0.20
MAX_TEXT_TOURNAMENT_WEIGHT = 0.6
EMISSION_MULTIPLIER_THRESHOLD = 0.05
EMISSION_MULTIPLIER_RATE = 2.0


def calculate_emission_boost_from_perf(performance_diff: float) -> float:
    """Calculate emission boost from performance."""
    if performance_diff <= EMISSION_MULTIPLIER_THRESHOLD:
        return 0.0
    excess_performance = performance_diff - EMISSION_MULTIPLIER_THRESHOLD
    emission_increase = excess_performance * EMISSION_MULTIPLIER_RATE
    return emission_increase


def calculate_hybrid_decays(
    first_championship_time: datetime, consecutive_wins: int, current_time: datetime
) -> tuple[float, float, bool]:
    """Calculate decay components and determine if hybrid logic applies."""
    if first_championship_time is None:
        return (1.0, 1.0, False)

    # Timezone alignment
    cutoff_date = datetime.combine(EMISSION_TIME_DECAY_START_DATE, datetime.min.time(), tzinfo=timezone.utc)
    current_time_utc = current_time.replace(tzinfo=timezone.utc) if current_time.tzinfo is None else current_time
    first_championship_time_utc = (
        first_championship_time.replace(tzinfo=timezone.utc)
        if first_championship_time.tzinfo is None
        else first_championship_time
    )

    # Before cutoff: old system only
    if current_time_utc < cutoff_date:
        old_decay = max(0, consecutive_wins - 1) * EMISSION_BOOST_DECAY_PER_WIN
        return (old_decay, 0.0, False)

    # After cutoff, champion won before cutoff: hybrid system
    if first_championship_time_utc < cutoff_date:
        old_decay = max(0, consecutive_wins - 1) * EMISSION_BOOST_DECAY_PER_WIN
        days_since_cutoff = (current_time_utc - cutoff_date).total_seconds() / SECONDS_PER_DAY
        new_decay = days_since_cutoff * EMISSION_DAILY_TIME_DECAY_RATE
        return (old_decay, new_decay, True)
    else:
        # Champion won after cutoff: new system only
        days_as_champion = (current_time_utc - first_championship_time_utc).total_seconds() / SECONDS_PER_DAY
        new_decay = days_as_champion * EMISSION_DAILY_TIME_DECAY_RATE
        return (0.0, new_decay, False)


def calculate_tournament_weight_with_decay(
    base_weight: float,
    emission_boost: float,
    old_decay: float,
    new_decay: float,
    apply_hybrid: bool,
    max_weight: float,
) -> float:
    """Apply hybrid decay logic and return final capped tournament weight."""
    if apply_hybrid:
        # Pre-cutoff champion after cutoff: hybrid logic
        boost_after_old = max(0.0, emission_boost - old_decay)
        if boost_after_old == 0.0:
            # Boost completely wiped out, now decay from base
            final_weight = max(0.0, base_weight - new_decay)
        else:
            # Boost still exists, don't apply new_decay
            final_weight = max(0.0, base_weight + boost_after_old)
    else:
        # Old regime purely (before cutoff)
        if old_decay > 0.0:
            boost_after_old = max(0.0, emission_boost - old_decay)
            final_weight = max(0.0, base_weight + boost_after_old)
        # New regime purely (after cutoff, champion won after cutoff)
        elif new_decay > 0.0:
            final_weight = max(0.0, base_weight + emission_boost - new_decay)
        else:
            final_weight = base_weight + emission_boost

    final_weight = min(final_weight, max_weight)
    return final_weight


def run_scenario(
    scenario_name: str,
    performance_diff: float,
    first_championship_time: datetime,
    consecutive_wins: int,
    test_times: list[tuple[str, datetime, int]],
):
    """Test a specific scenario over multiple time points."""
    print(f"\n{'=' * 100}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'=' * 100}")
    print(f"Performance difference: {performance_diff:.2%}")
    print(f"First became champion at: {first_championship_time}")
    print()

    emission_boost = calculate_emission_boost_from_perf(performance_diff)
    print(f"Initial emission boost: {emission_boost:.4f} ({emission_boost * 100:.2f}%)")
    print(f"Base weight: {TOURNAMENT_TEXT_WEIGHT:.4f}")
    print()

    print(f"{'Time Point':<30} {'Wins':<6} {'Days':<8} {'Old':<8} {'New':<8} {'Hybrid':<8} {'Final':<10}")
    print("-" * 100)

    for time_label, sim_time, wins_at_time in test_times:
        old_decay, new_decay, apply_hybrid = calculate_hybrid_decays(first_championship_time, wins_at_time, sim_time)

        final_weight = calculate_tournament_weight_with_decay(
            TOURNAMENT_TEXT_WEIGHT, emission_boost, old_decay, new_decay, apply_hybrid, MAX_TEXT_TOURNAMENT_WEIGHT
        )

        days_as_champion = (
            sim_time.replace(tzinfo=timezone.utc) - first_championship_time.replace(tzinfo=timezone.utc)
        ).total_seconds() / SECONDS_PER_DAY

        print(
            f"{time_label:<30} {wins_at_time:>4}   {days_as_champion:>6.1f}d  "
            f"{old_decay:>6.2%}  {new_decay:>6.2%}  {str(apply_hybrid):<8}  {final_weight:>8.4f}"
        )


def main():
    print("\n" + "=" * 100)
    print("HYBRID DECAY SYSTEM TEST")
    print("=" * 100)
    print(f"\nCutoff date: {EMISSION_TIME_DECAY_START_DATE}")
    print(f"Old decay rate: {EMISSION_BOOST_DECAY_PER_WIN * 100:.1f}% per win")
    print(f"New decay rate: {EMISSION_DAILY_TIME_DECAY_RATE * 100:.2f}% per day")
    print(f"Base text weight: {TOURNAMENT_TEXT_WEIGHT:.4f}")
    print(f"Max text weight: {MAX_TEXT_TOURNAMENT_WEIGHT:.4f}")

    # Test 1: Before cutoff date (old system only)
    run_scenario(
        "Before Cutoff - Old System Only",
        performance_diff=0.15,  # 15% advantage -> 20% boost
        first_championship_time=datetime(2025, 10, 1, tzinfo=timezone.utc),
        consecutive_wins=10,
        test_times=[
            ("Oct 5, 2025 (1st win)", datetime(2025, 10, 5, tzinfo=timezone.utc), 1),
            ("Oct 10, 2025 (3rd win)", datetime(2025, 10, 10, tzinfo=timezone.utc), 3),
            ("Oct 20, 2025 (6th win)", datetime(2025, 10, 20, tzinfo=timezone.utc), 6),
            ("Oct 28, 2025 (10th win)", datetime(2025, 10, 28, tzinfo=timezone.utc), 10),
        ],
    )

    # Test 2: Hybrid - small old decay, boost survives
    run_scenario(
        "Hybrid - Small Old Decay (Boost Survives)",
        performance_diff=0.15,  # 15% advantage -> 20% boost
        first_championship_time=datetime(2025, 10, 20, tzinfo=timezone.utc),
        consecutive_wins=5,
        test_times=[
            ("Oct 20, 2025 (1st win)", datetime(2025, 10, 20, tzinfo=timezone.utc), 1),
            ("Oct 28, 2025 (3rd win)", datetime(2025, 10, 28, tzinfo=timezone.utc), 3),
            ("Nov 26, 2025 (CUTOFF - 4th)", datetime(2025, 11, 26, tzinfo=timezone.utc), 4),
            ("Dec 5, 2025 (+9d)", datetime(2025, 12, 5, tzinfo=timezone.utc), 5),
            ("Dec 26, 2025 (+30d)", datetime(2025, 12, 26, tzinfo=timezone.utc), 8),
            ("Jan 25, 2026 (+60d)", datetime(2026, 1, 25, tzinfo=timezone.utc), 12),
        ],
    )

    # Test 3: Hybrid - large old decay, boost wiped out, new decay kicks in
    run_scenario(
        "Hybrid - Large Old Decay (Boost Wiped Out)",
        performance_diff=0.15,  # 15% advantage -> 20% boost
        first_championship_time=datetime(2025, 9, 1, tzinfo=timezone.utc),
        consecutive_wins=30,
        test_times=[
            ("Sep 1, 2025 (1st win)", datetime(2025, 9, 1, tzinfo=timezone.utc), 1),
            ("Oct 15, 2025 (15th win)", datetime(2025, 10, 15, tzinfo=timezone.utc), 15),
            ("Nov 26, 2025 (CUTOFF - 21st)", datetime(2025, 11, 26, tzinfo=timezone.utc), 21),
            ("Dec 5, 2025 (+9d)", datetime(2025, 12, 5, tzinfo=timezone.utc), 24),
            ("Dec 26, 2025 (+30d)", datetime(2025, 12, 26, tzinfo=timezone.utc), 30),
            ("Jan 25, 2026 (+60d)", datetime(2026, 1, 25, tzinfo=timezone.utc), 36),
        ],
    )

    # Test 4: New system only - champion won after cutoff
    run_scenario(
        "New System Only - Champion Won After Cutoff",
        performance_diff=0.10,  # 10% advantage -> 10% boost
        first_championship_time=datetime(2025, 11, 28, tzinfo=timezone.utc),
        consecutive_wins=15,
        test_times=[
            ("Nov 28, 2025 (1st win)", datetime(2025, 11, 28, tzinfo=timezone.utc), 1),
            ("Dec 7, 2025 (+9d)", datetime(2025, 12, 7, tzinfo=timezone.utc), 4),
            ("Dec 28, 2025 (+30d)", datetime(2025, 12, 28, tzinfo=timezone.utc), 11),
            ("Jan 27, 2026 (+60d)", datetime(2026, 1, 27, tzinfo=timezone.utc), 15),
            ("Feb 26, 2026 (+90d)", datetime(2026, 2, 26, tzinfo=timezone.utc), 20),
        ],
    )

    # Test 5: Edge case - exactly at cutoff transition
    run_scenario(
        "Edge Case - Transition at Cutoff",
        performance_diff=0.12,  # 12% advantage -> 14% boost
        first_championship_time=datetime(2025, 10, 20, tzinfo=timezone.utc),
        consecutive_wins=8,
        test_times=[
            ("Nov 25, 2025 (day before)", datetime(2025, 11, 25, 23, 59, tzinfo=timezone.utc), 4),
            ("Nov 26, 2025 (cutoff day)", datetime(2025, 11, 26, 0, 0, tzinfo=timezone.utc), 5),
            ("Nov 26, 2025 (cutoff +1h)", datetime(2025, 11, 26, 1, 0, tzinfo=timezone.utc), 5),
            ("Nov 27, 2025 (+1 day)", datetime(2025, 11, 27, tzinfo=timezone.utc), 6),
        ],
    )

    print("\n" + "=" * 100)
    print("KEY INSIGHTS - NEW HYBRID SYSTEM")
    print("=" * 100)
    print()
    print("BEFORE Nov 26, 2025 (Cutoff):")
    print("  → Old system: decay = (consecutive_wins - 1) × 1%")
    print("  → Applied to emission boost only")
    print("  → final_weight = base + max(0, boost - old_decay)")
    print()
    print("AFTER Nov 26, 2025 - Three Cases:")
    print()
    print("1. Champion won BEFORE cutoff (apply_hybrid=True):")
    print("   → Calculate old_decay from consecutive wins")
    print("   → Calculate new_decay from days since cutoff")
    print("   → Apply old_decay to boost first")
    print("   → IF boost completely wiped out (boost - old_decay ≤ 0):")
    print("      • THEN apply new_decay to base weight")
    print("      • final_weight = max(0, base - new_decay)")
    print("   → ELSE boost still exists:")
    print("      • DON'T apply new_decay")
    print("      • final_weight = base + (boost - old_decay)")
    print()
    print("2. Champion won AFTER cutoff (apply_hybrid=False, new_decay > 0):")
    print("   → Only new time-based decay")
    print("   → new_decay = days_as_champion × 0.33%")
    print("   → final_weight = base + boost - new_decay")
    print()
    print("3. No champion or before cutoff (apply_hybrid=False, old_decay > 0):")
    print("   → Only old consecutive wins decay")
    print("   → final_weight = base + max(0, boost - old_decay)")
    print()
    print("CRITICAL OBSERVATION:")
    print("  ✓ For pre-cutoff champions, new_decay ONLY applies if boost is wiped out")
    print("  ✓ This creates a 'cliff' effect once old_decay exceeds emission_boost")
    print("  ✓ After the cliff, base weight starts decaying with new_decay")
    print()
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
