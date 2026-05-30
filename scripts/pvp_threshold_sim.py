"""
PvP minimum competence threshold simulation.

Rule: in a pairwise matchup, you must win >= THRESHOLD fraction of games
in EVERY environment to earn any points from that matchup. Otherwise you get 0.
"""

import itertools
import random
from dataclasses import dataclass, field

# --- Config ---
WIN_PTS = 3
DRAW_PTS = 1
LOSS_PTS = 0
GAMES_PER_ENV = 20  # 10 seeds x 2 (position swap)
THRESHOLD = 0.20  # must win at least 20% of games in each env
NUM_SIMS = 5000  # Monte Carlo runs per scenario


@dataclass
class Miner:
    name: str
    # win probability per environment (against an equal opponent, 0.5 = average)
    env_skill: dict[str, float] = field(default_factory=dict)


def simulate_games(p_a: float, p_b: float, n_games: int) -> tuple[int, int]:
    """Simulate n_games between two miners. Returns (a_wins, b_wins)."""
    a_wins = 0
    b_wins = 0
    for _ in range(n_games):
        # Simple skill-based model: each player's score is skill + noise
        a_score = p_a + random.gauss(0, 0.15)
        b_score = p_b + random.gauss(0, 0.15)
        if a_score > b_score:
            a_wins += 1
        elif b_score > a_score:
            b_wins += 1
        # else: draw (neither gets a game win)
    return a_wins, b_wins


def score_matchup(
    miner_a: Miner,
    miner_b: Miner,
    envs: list[str],
    threshold: float | None,
) -> tuple[float, float]:
    """Play one full matchup. Returns (a_pts, b_pts)."""
    a_env_pts = 0.0
    b_env_pts = 0.0
    a_passes_floor = True
    b_passes_floor = True

    for env in envs:
        skill_a = miner_a.env_skill.get(env, 0.5)
        skill_b = miner_b.env_skill.get(env, 0.5)
        a_wins, b_wins = simulate_games(skill_a, skill_b, GAMES_PER_ENV)
        total_decided = a_wins + b_wins

        # Check competence floor (game win rate in this env)
        if threshold is not None and total_decided > 0:
            if a_wins / total_decided < threshold:
                a_passes_floor = False
            if b_wins / total_decided < threshold:
                b_passes_floor = False

        # Environment winner
        if a_wins > b_wins:
            a_env_pts += WIN_PTS
            b_env_pts += LOSS_PTS
        elif b_wins > a_wins:
            b_env_pts += WIN_PTS
            a_env_pts += LOSS_PTS
        else:
            a_env_pts += DRAW_PTS
            b_env_pts += DRAW_PTS

    # Apply floor
    if threshold is not None:
        if not a_passes_floor:
            a_env_pts = 0
        if not b_passes_floor:
            b_env_pts = 0

    return a_env_pts, b_env_pts


def run_tournament(miners: list[Miner], envs: list[str], threshold: float | None) -> dict[str, float]:
    """Round-robin, return total points per miner."""
    points = {m.name: 0.0 for m in miners}
    for a, b in itertools.combinations(miners, 2):
        a_pts, b_pts = score_matchup(a, b, envs, threshold)
        points[a.name] += a_pts
        points[b.name] += b_pts
    return points


def monte_carlo(miners: list[Miner], envs: list[str], threshold: float | None, n_sims: int = NUM_SIMS):
    """Run n_sims tournaments, return average points and win rates."""
    totals = {m.name: 0.0 for m in miners}
    first_place = {m.name: 0 for m in miners}

    for _ in range(n_sims):
        pts = run_tournament(miners, envs, threshold)
        for name, p in pts.items():
            totals[name] += p
        winner = max(pts, key=pts.get)
        first_place[winner] += 1

    avg_pts = {name: totals[name] / n_sims for name in totals}
    win_rate = {name: first_place[name] / n_sims for name in first_place}
    return avg_pts, win_rate


def print_results(label: str, avg_pts: dict, win_rate: dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'Miner':<20} {'Avg Pts':>10} {'1st Place %':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12}")
    for name in sorted(avg_pts, key=avg_pts.get, reverse=True):
        print(f"  {name:<20} {avg_pts[name]:>10.1f} {win_rate[name]*100:>11.1f}%")


def scenario_header(title: str, desc: str):
    print(f"\n{'#'*60}")
    print(f"# {title}")
    print(f"# {desc}")
    print(f"{'#'*60}")


# =====================================================================
# SCENARIO 1: Specialist exploit (4 miners, 2 envs)
# =====================================================================
scenario_header(
    "SCENARIO 1: Specialist vs Generalists",
    "A focuses only on rummy (skill=0.85), ignores poker (skill=0.15). B,C,D balanced (0.55 both).",
)
envs = ["poker", "rummy"]
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85}),
    Miner("B_balanced", {"poker": 0.55, "rummy": 0.55}),
    Miner("C_balanced", {"poker": 0.55, "rummy": 0.55}),
    Miner("D_balanced", {"poker": 0.55, "rummy": 0.55}),
]
avg, wr = monte_carlo(miners, envs, threshold=None)
print_results("No threshold", avg, wr)
avg, wr = monte_carlo(miners, envs, threshold=THRESHOLD)
print_results(f"With {THRESHOLD:.0%} floor", avg, wr)

# =====================================================================
# SCENARIO 2: Specialist is REALLY good at one game
# =====================================================================
scenario_header(
    "SCENARIO 2: Elite specialist",
    "A is elite at rummy (0.95), terrible at poker (0.05). Others balanced (0.55).",
)
miners = [
    Miner("A_elite_spec", {"poker": 0.05, "rummy": 0.95}),
    Miner("B_balanced", {"poker": 0.55, "rummy": 0.55}),
    Miner("C_balanced", {"poker": 0.55, "rummy": 0.55}),
    Miner("D_balanced", {"poker": 0.55, "rummy": 0.55}),
]
avg, wr = monte_carlo(miners, envs, threshold=None)
print_results("No threshold", avg, wr)
avg, wr = monte_carlo(miners, envs, threshold=THRESHOLD)
print_results(f"With {THRESHOLD:.0%} floor", avg, wr)

# =====================================================================
# SCENARIO 3: Specialist who meets the floor
# =====================================================================
scenario_header(
    "SCENARIO 3: Smart specialist (meets floor)",
    "A trains rummy hard (0.80) but does just enough poker (0.35). Others balanced (0.55).",
)
miners = [
    Miner("A_smart_spec", {"poker": 0.35, "rummy": 0.80}),
    Miner("B_balanced", {"poker": 0.55, "rummy": 0.55}),
    Miner("C_balanced", {"poker": 0.55, "rummy": 0.55}),
    Miner("D_balanced", {"poker": 0.55, "rummy": 0.55}),
]
avg, wr = monte_carlo(miners, envs, threshold=None)
print_results("No threshold", avg, wr)
avg, wr = monte_carlo(miners, envs, threshold=THRESHOLD)
print_results(f"With {THRESHOLD:.0%} floor", avg, wr)

# =====================================================================
# SCENARIO 4: Two specialists splitting games
# =====================================================================
scenario_header(
    "SCENARIO 4: Two specialists, each owns one game",
    "A focuses rummy (0.85/0.15), B focuses poker (0.85/0.15). C,D balanced (0.55).",
)
miners = [
    Miner("A_rummy_spec", {"poker": 0.15, "rummy": 0.85}),
    Miner("B_poker_spec", {"poker": 0.85, "rummy": 0.15}),
    Miner("C_balanced", {"poker": 0.55, "rummy": 0.55}),
    Miner("D_balanced", {"poker": 0.55, "rummy": 0.55}),
]
avg, wr = monte_carlo(miners, envs, threshold=None)
print_results("No threshold", avg, wr)
avg, wr = monte_carlo(miners, envs, threshold=THRESHOLD)
print_results(f"With {THRESHOLD:.0%} floor", avg, wr)

# =====================================================================
# SCENARIO 5: 3 environments (later rounds)
# =====================================================================
scenario_header(
    "SCENARIO 5: 3 environments (later round)",
    "A specializes in one (0.90), weak in other two (0.15). Others balanced (0.55).",
)
envs3 = ["poker", "rummy", "liar_dice"]
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.90, "liar_dice": 0.15}),
    Miner("B_balanced", {"poker": 0.55, "rummy": 0.55, "liar_dice": 0.55}),
    Miner("C_balanced", {"poker": 0.55, "rummy": 0.55, "liar_dice": 0.55}),
    Miner("D_balanced", {"poker": 0.55, "rummy": 0.55, "liar_dice": 0.55}),
]
avg, wr = monte_carlo(miners, envs3, threshold=None)
print_results("No threshold", avg, wr)
avg, wr = monte_carlo(miners, envs3, threshold=THRESHOLD)
print_results(f"With {THRESHOLD:.0%} floor", avg, wr)

# =====================================================================
# SCENARIO 6: Threshold sensitivity sweep
# =====================================================================
scenario_header(
    "SCENARIO 6: Threshold sweep (specialist exploit case)",
    "Same as Scenario 1. Sweeping threshold from 0% to 45%.",
)
envs = ["poker", "rummy"]
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85}),
    Miner("B_balanced", {"poker": 0.55, "rummy": 0.55}),
    Miner("C_balanced", {"poker": 0.55, "rummy": 0.55}),
    Miner("D_balanced", {"poker": 0.55, "rummy": 0.55}),
]
print(f"\n  {'Threshold':>10} {'A 1st%':>10} {'A avg pts':>10} {'B avg pts':>10}")
print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for t in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    avg, wr = monte_carlo(miners, envs, threshold=t if t > 0 else None, n_sims=3000)
    print(f"  {t:>9.0%} {wr['A_specialist']*100:>9.1f}% {avg['A_specialist']:>10.1f} {avg['B_balanced']:>10.1f}")

# =====================================================================
# SCENARIO 7: Legit good player (no exploit, just skilled)
# =====================================================================
scenario_header(
    "SCENARIO 7: Genuinely skilled miner (no exploit)",
    "A is better at both games (0.65/0.65). Others average (0.50). Threshold shouldn't hurt A.",
)
envs = ["poker", "rummy"]
miners = [
    Miner("A_skilled", {"poker": 0.65, "rummy": 0.65}),
    Miner("B_average", {"poker": 0.50, "rummy": 0.50}),
    Miner("C_average", {"poker": 0.50, "rummy": 0.50}),
    Miner("D_average", {"poker": 0.50, "rummy": 0.50}),
]
avg, wr = monte_carlo(miners, envs, threshold=None)
print_results("No threshold", avg, wr)
avg, wr = monte_carlo(miners, envs, threshold=THRESHOLD)
print_results(f"With {THRESHOLD:.0%} floor", avg, wr)

print("\n" + "="*60)
print("  Done. GAMES_PER_ENV={}, SIMS={}".format(GAMES_PER_ENV, NUM_SIMS))
print("="*60)
