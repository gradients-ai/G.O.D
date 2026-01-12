import asyncio
import json
import traceback
from datetime import datetime

from fiber.chain import fetch_nodes

from validator.core.config import load_config
from validator.core.constants import EMISSION_BURN_HOTKEY
from validator.core.weight_setting import build_tournament_audit_data
from validator.core.weight_setting import get_node_weights_from_tournament_audit_data
from validator.utils.logging import get_logger
from validator.utils.util import try_db_connections


logger = get_logger(__name__)


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


async def debug_weight_calculation():
    """Run the weight calculation with extensive debug output"""

    print_section("1. LOADING CONFIG")
    config = load_config()
    print(f"✓ Config loaded")
    print(f"  - NetUID: {config.netuid}")
    print(f"  - Substrate URL: {config.substrate.url if hasattr(config.substrate, 'url') else 'N/A'}")

    print_section("2. CONNECTING TO DATABASE")
    await try_db_connections(config)
    print(f"✓ Database connections established")

    print_section("3. BUILDING TOURNAMENT AUDIT DATA")
    print("Calling build_tournament_audit_data()...")
    tournament_audit_data = await build_tournament_audit_data(config.psql_db)
    print(f"✓ Tournament audit data built\n")

    # Print detailed tournament audit data
    print("Tournament Audit Data Details:")
    print("-" * 80)

    print(f"\n📊 Text Tournament Data:")
    if tournament_audit_data.text_tournament_data:
        print(f"  Tournament ID: {tournament_audit_data.text_tournament_data.tournament_id}")
        print(f"  Base Winner: {tournament_audit_data.text_tournament_data.base_winner_hotkey}")
        print(f"  Winner: {tournament_audit_data.text_tournament_data.winner_hotkey}")
        print(f"  Number of rounds: {len(tournament_audit_data.text_tournament_data.rounds)}")
        for idx, round_data in enumerate(tournament_audit_data.text_tournament_data.rounds):
            print(f"    Round {round_data.round_number} ({round_data.round_type}): {round_data.round_id} - {len(round_data.tasks)} tasks")
    else:
        print("  ❌ No text tournament data")

    print(f"\n📊 Image Tournament Data:")
    if tournament_audit_data.image_tournament_data:
        print(f"  Tournament ID: {tournament_audit_data.image_tournament_data.tournament_id}")
        print(f"  Base Winner: {tournament_audit_data.image_tournament_data.base_winner_hotkey}")
        print(f"  Winner: {tournament_audit_data.image_tournament_data.winner_hotkey}")
        print(f"  Number of rounds: {len(tournament_audit_data.image_tournament_data.rounds)}")
        for idx, round_data in enumerate(tournament_audit_data.image_tournament_data.rounds):
            print(f"    Round {round_data.round_number} ({round_data.round_type}): {round_data.round_id} - {len(round_data.tasks)} tasks")
    else:
        print("  ❌ No image tournament data")

    print(f"\n💰 Weight Distribution:")
    print(f"  Text tournament weight: {tournament_audit_data.text_tournament_weight:.6f}")
    print(f"  Image tournament weight: {tournament_audit_data.image_tournament_weight:.6f}")
    print(f"  Burn weight: {tournament_audit_data.burn_weight:.6f}")
    print(
        f"  Total: {tournament_audit_data.text_tournament_weight + tournament_audit_data.image_tournament_weight + tournament_audit_data.burn_weight:.6f}"
    )

    print(f"\n👥 Active Participants:")
    print(f"  Count: {len(tournament_audit_data.participants)}")
    if tournament_audit_data.participants:
        print(f"  Hotkeys (first 5): {tournament_audit_data.participants[:5]}")
        if len(tournament_audit_data.participants) > 5:
            print(f"  ... and {len(tournament_audit_data.participants) - 5} more")

    print(f"\n📈 Weekly Participation Data:")
    print(f"  Count: {len(tournament_audit_data.weekly_participation)}")
    if tournament_audit_data.weekly_participation:
        for participation in tournament_audit_data.weekly_participation[:3]:
            print(f"    {participation.hotkey[:16]}... - {participation.total_tasks} tasks")
        if len(tournament_audit_data.weekly_participation) > 3:
            print(f"  ... and {len(tournament_audit_data.weekly_participation) - 3} more")

    print_section("4. CALCULATING NODE WEIGHTS")
    print("Calling get_node_weights_from_tournament_audit_data()...")
    result = await get_node_weights_from_tournament_audit_data(config.substrate, config.netuid, tournament_audit_data)
    all_node_ids = result.node_ids
    all_node_weights = result.node_weights
    print(f"✓ Node weights calculated")

    print_section("5. WEIGHT CALCULATION RESULTS")

    print(f"Total nodes: {len(all_node_ids)}")
    print(f"Nodes with non-zero weights: {sum(1 for w in all_node_weights if w > 0)}")
    print(f"Sum of all weights: {sum(all_node_weights):.6f}")

    print(f"\n🏆 Top 20 Nodes by Weight:")
    print(f"{'Rank':<6} {'Node ID':<10} {'Weight':<15} {'Weight %':<10}")
    print("-" * 80)

    # Sort by weight descending
    node_weight_pairs = list(zip(all_node_ids, all_node_weights))
    node_weight_pairs.sort(key=lambda x: x[1], reverse=True)

    for idx, (node_id, weight) in enumerate(node_weight_pairs[:20], 1):
        weight_pct = (weight / sum(all_node_weights) * 100) if sum(all_node_weights) > 0 else 0
        print(f"{idx:<6} {node_id:<10} {weight:<15.6f} {weight_pct:<10.4f}%")

    print(f"\n📊 Weight Distribution Analysis:")
    weights_sorted = sorted(all_node_weights, reverse=True)
    non_zero_weights = [w for w in weights_sorted if w > 0]

    if non_zero_weights:
        print(f"  Max weight: {max(non_zero_weights):.6f}")
        print(f"  Min non-zero weight: {min(non_zero_weights):.6f}")
        print(f"  Median weight (non-zero): {non_zero_weights[len(non_zero_weights) // 2]:.6f}")
        print(f"  Average weight (non-zero): {sum(non_zero_weights) / len(non_zero_weights):.6f}")

        # Show weight percentiles
        print(f"\n  Weight Percentiles (non-zero):")
        for percentile in [90, 75, 50, 25, 10]:
            idx = int(len(non_zero_weights) * (100 - percentile) / 100)
            if idx < len(non_zero_weights):
                print(f"    Top {100 - percentile}%: {non_zero_weights[idx]:.6f}")

    # Check for specific hotkeys if in tournament data
    print_section("6. VERIFY SPECIFIC HOTKEYS")

    all_nodes = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
    hotkey_to_node = {node.hotkey: node for node in all_nodes}

    # Check text tournament winner
    if tournament_audit_data.text_tournament_data and tournament_audit_data.text_tournament_data.winner_hotkey:
        winner_hotkey = tournament_audit_data.text_tournament_data.winner_hotkey
        print(f"\n🏆 Text Tournament Winner: {winner_hotkey}")
        node = hotkey_to_node.get(winner_hotkey)
        if node:
            weight = all_node_weights[node.node_id]
            weight_pct = (weight / sum(all_node_weights) * 100) if sum(all_node_weights) > 0 else 0
            print(f"  Node ID: {node.node_id}")
            print(f"  Weight: {weight:.6f} ({weight_pct:.4f}%)")
            print(f"  Current chain incentive: {node.incentive / 65535:.6f}")
        else:
            print(f"  ❌ Node not found on chain")

    # Check image tournament winner
    if tournament_audit_data.image_tournament_data and tournament_audit_data.image_tournament_data.winner_hotkey:
        winner_hotkey = tournament_audit_data.image_tournament_data.winner_hotkey
        print(f"\n🏆 Image Tournament Winner: {winner_hotkey}")
        node = hotkey_to_node.get(winner_hotkey)
        if node:
            weight = all_node_weights[node.node_id]
            weight_pct = (weight / sum(all_node_weights) * 100) if sum(all_node_weights) > 0 else 0
            print(f"  Node ID: {node.node_id}")
            print(f"  Weight: {weight:.6f} ({weight_pct:.4f}%)")
            print(f"  Current chain incentive: {node.incentive / 65535:.6f}")
        else:
            print(f"  ❌ Node not found on chain")

    # Check a few random participants
    if tournament_audit_data.participants:
        print(f"\n👥 Sample Tournament Participants (first 3):")
        for participant_hotkey in tournament_audit_data.participants[:3]:
            node = hotkey_to_node.get(participant_hotkey)
            if node:
                weight = all_node_weights[node.node_id]
                weight_pct = (weight / sum(all_node_weights) * 100) if sum(all_node_weights) > 0 else 0
                print(f"\n  {participant_hotkey[:16]}...")
                print(f"    Node ID: {node.node_id}")
                print(f"    Weight: {weight:.6f} ({weight_pct:.4f}%)")
                print(f"    Current chain incentive: {node.incentive / 65535:.6f}")

    # Check burn address

    print(f"\n🔥 Burn Address: {EMISSION_BURN_HOTKEY}")
    burn_node = hotkey_to_node.get(EMISSION_BURN_HOTKEY)
    if burn_node:
        weight = all_node_weights[burn_node.node_id]
        weight_pct = (weight / sum(all_node_weights) * 100) if sum(all_node_weights) > 0 else 0
        print(f"  Node ID: {burn_node.node_id}")
        print(f"  Weight: {weight:.6f} ({weight_pct:.4f}%)")
        print(f"  Current chain incentive: {burn_node.incentive / 65535:.6f}")
    else:
        print(f"  ❌ Burn node not found on chain")

    print_section("7. EXPORT DATA TO JSON")

    export_data = {
        "timestamp": datetime.now().isoformat(),
        "tournament_audit_data": {
            "text_tournament_weight": tournament_audit_data.text_tournament_weight,
            "image_tournament_weight": tournament_audit_data.image_tournament_weight,
            "burn_weight": tournament_audit_data.burn_weight,
            "participants_count": len(tournament_audit_data.participants),
            "text_tournament_id": tournament_audit_data.text_tournament_data.tournament_id
            if tournament_audit_data.text_tournament_data
            else None,
            "image_tournament_id": tournament_audit_data.image_tournament_data.tournament_id
            if tournament_audit_data.image_tournament_data
            else None,
        },
        "weight_stats": {
            "total_nodes": len(all_node_ids),
            "nodes_with_weight": sum(1 for w in all_node_weights if w > 0),
            "total_weight": sum(all_node_weights),
            "max_weight": max(all_node_weights) if all_node_weights else 0,
            "min_non_zero_weight": min([w for w in all_node_weights if w > 0]) if any(w > 0 for w in all_node_weights) else 0,
        },
        "top_20_nodes": [
            {
                "node_id": node_id,
                "weight": weight,
                "weight_pct": (weight / sum(all_node_weights) * 100) if sum(all_node_weights) > 0 else 0,
            }
            for node_id, weight in node_weight_pairs[:20]
        ],
    }

    output_file = f"weight_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"✓ Debug data exported to: {output_file}")

    print_section("✅ WEIGHT CALCULATION COMPLETE (NO WEIGHTS SET)")
    print("This script DID NOT set any weights on-chain.")
    print("All calculations are for verification purposes only.")


async def main():
    try:
        await debug_weight_calculation()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
