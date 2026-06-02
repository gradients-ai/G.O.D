#!/usr/bin/env python3
"""
Simple script to check scoring for a hotkey using tournament-only system
"""

import asyncio
import sys

from fiber.chain import fetch_nodes

from core.logging import get_logger
from validator.app.config import load_config
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.scoring.weights import build_tournament_audit_data
from validator.scoring.weights import get_node_weights_from_tournament_audit_data
from validator.tasks.details import try_db_connections
from validator.tournament.models import TournamentType


logger = get_logger(__name__)


async def check_hotkey_scoring(hotkey: str):
    """Check the scoring for a specific hotkey in tournament-only system"""

    # Load config
    config = load_config()

    # Connect to database
    await try_db_connections(config)

    print(f"\nChecking tournament scoring for hotkey: {hotkey}")
    print("=" * 80)

    # Build tournament audit data using the centralized function (same as validator does)
    print("\nGathering tournament data...")
    tournament_audit_data = await build_tournament_audit_data(config.psql_db)

    # Fetch tournament data for display purposes
    text_tournament = await get_latest_completed_tournament(config.psql_db, TournamentType.TEXT)
    image_tournament = await get_latest_completed_tournament(config.psql_db, TournamentType.IMAGE)

    if text_tournament:
        print(f"  Text Tournament: {text_tournament.tournament_id}")
        print(f"    Winner: {text_tournament.winner_hotkey}")
    else:
        print("  Text Tournament: None")

    if image_tournament:
        print(f"  Image Tournament: {image_tournament.tournament_id}")
        print(f"    Winner: {image_tournament.winner_hotkey}")
    else:
        print("  Image Tournament: None")

    print("\nWeight Distribution:")
    print(f"  Text tournament weight: {tournament_audit_data.text_tournament_weight:.6f}")
    print(f"  Image tournament weight: {tournament_audit_data.image_tournament_weight:.6f}")
    print(f"  Burn weight: {tournament_audit_data.burn_weight:.6f}")
    print(f"  Active participants: {len(tournament_audit_data.participants)}")

    # Calculate weights
    print("\nCalculating weights...")
    result = await get_node_weights_from_tournament_audit_data(config.substrate, config.netuid, tournament_audit_data)

    all_node_weights = result.node_weights

    # Get all nodes to map hotkey to node_id
    all_nodes = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
    hotkey_to_node = {node.hotkey: node for node in all_nodes}

    target_node = hotkey_to_node.get(hotkey)
    if not target_node:
        print(f"\n❌ Node not found for hotkey {hotkey}")
        return

    # Display results
    print(f"\n{'=' * 80}")
    print(f"Results for Node ID: {target_node.node_id}")
    print(f"{'=' * 80}")

    # Check tournament participation
    is_participant = hotkey in tournament_audit_data.participants
    print(f"\n🏆 Tournament Participation: {'✅ YES' if is_participant else '❌ NO'}")

    # Check tournament wins
    text_winner = text_tournament.winner_hotkey == hotkey if text_tournament else False
    image_winner = image_tournament.winner_hotkey == hotkey if image_tournament else False

    if text_winner or image_winner:
        print("\n🎯 Tournament Winner:")
        if text_winner:
            print("   ✅ Text Tournament Winner!")
        if image_winner:
            print("   ✅ Image Tournament Winner!")

    # Check tournament rankings
    print("\n📊 Tournament Rankings:")

    if tournament_audit_data.text_tournament_data:
        text_position = None
        for round_idx, round_data in enumerate(tournament_audit_data.text_tournament_data.rounds):
            if hotkey in round_data.participants:
                text_position = (round_idx, round_data.round_name)
                break

        if text_position:
            print(f"   Text Tournament: Reached {text_position[1]} (Round {text_position[0] + 1})")
        else:
            print("   Text Tournament: Did not participate")

    if tournament_audit_data.image_tournament_data:
        image_position = None
        for round_idx, round_data in enumerate(tournament_audit_data.image_tournament_data.rounds):
            if hotkey in round_data.participants:
                image_position = (round_idx, round_data.round_name)
                break

        if image_position:
            print(f"   Image Tournament: Reached {image_position[1]} (Round {image_position[0] + 1})")
        else:
            print("   Image Tournament: Did not participate")

    # Get calculated weight
    calculated_weight = all_node_weights[target_node.node_id]

    # Calculate sum of all weights
    total_weight_sum = sum(all_node_weights)

    print("\n💰 Weight Breakdown:")
    print(f"   Calculated weight: {calculated_weight:.6f}")
    print(f"   Sum of ALL node weights: {total_weight_sum:.6f}")
    print(f"   This node's share: {calculated_weight / total_weight_sum:.4%}")

    # Convert chain weight from raw to normalized
    chain_weight_normalized = target_node.incentive / 65535
    print("\n⛓️  Chain Comparison:")
    print(f"   Current chain weight (raw): {target_node.incentive}")
    print(f"   Current chain weight (normalized): {chain_weight_normalized:.6f}")

    if chain_weight_normalized > 0:
        diff_pct = (calculated_weight - chain_weight_normalized) / chain_weight_normalized * 100
        print(f"   Difference: {(calculated_weight - chain_weight_normalized):.6f} ({diff_pct:+.1f}%)")
    else:
        print(f"   Difference: {calculated_weight:.6f} (chain weight is zero)")

    # Show weight sources
    print("\n📈 Weight Sources:")
    weight_sources = []

    if text_winner:
        weight_sources.append(f"   ✅ Text tournament winner: ~{tournament_audit_data.text_tournament_weight:.4f}")
    if image_winner:
        weight_sources.append(f"   ✅ Image tournament winner: ~{tournament_audit_data.image_tournament_weight:.4f}")
    if is_participant:
        # Approximate participation weight
        import validator.scoring.constants as cts

        weight_sources.append(f"   ✅ Participation reward: {cts.TOURNAMENT_PARTICIPATION_WEIGHT:.6f}")

    if not weight_sources:
        print("   ❌ No weight sources (not a tournament winner or participant)")
    else:
        for source in weight_sources:
            print(source)

    print(f"\n{'=' * 80}")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ops.tools.scoring.check_scoring <hotkey>")
        sys.exit(1)

    hotkey = sys.argv[1]
    await check_hotkey_scoring(hotkey)


if __name__ == "__main__":
    asyncio.run(main())
