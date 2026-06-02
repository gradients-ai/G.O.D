"""
PvP minimum competence threshold simulation v2.

Realistic parameters: 6 miners per group, 2-3 environments.
Tests many miner compositions to find when specialization actually pays off.
"""

import itertools
import random
from dataclasses import dataclass, field

WIN_PTS = 3
DRAW_PTS = 1
GAMES_PER_ENV = 20
NUM_SIMS = 5000

ENVS_2 = ["poker", "rummy"]
ENVS_3 = ["poker", "rummy", "liar_dice"]


@dataclass
class Miner:
    name: str
    env_skill: dict[str, float] = field(default_factory=dict)


def simulate_games(p_a: float, p_b: float, n_games: int) -> tuple[int, int]:
    a_wins = b_wins = 0
    for _ in range(n_games):
        a_score = p_a + random.gauss(0, 0.15)
        b_score = p_b + random.gauss(0, 0.15)
        if a_score > b_score:
            a_wins += 1
        elif b_score > a_score:
            b_wins += 1
    return a_wins, b_wins


def score_matchup(a: Miner, b: Miner, envs: list[str], threshold: float | None) -> tuple[float, float]:
    a_pts = b_pts = 0.0
    a_floor = b_floor = True

    for env in envs:
        aw, bw = simulate_games(a.env_skill.get(env, 0.5), b.env_skill.get(env, 0.5), GAMES_PER_ENV)
        total = aw + bw
        if threshold and total > 0:
            if aw / total < threshold:
                a_floor = False
            if bw / total < threshold:
                b_floor = False
        if aw > bw:
            a_pts += WIN_PTS
        elif bw > aw:
            b_pts += WIN_PTS
        else:
            a_pts += DRAW_PTS
            b_pts += DRAW_PTS

    if threshold:
        if not a_floor:
            a_pts = 0
        if not b_floor:
            b_pts = 0
    return a_pts, b_pts


def run_group(miners: list[Miner], envs: list[str], threshold: float | None) -> dict[str, float]:
    points = {m.name: 0.0 for m in miners}
    for a, b in itertools.combinations(miners, 2):
        ap, bp = score_matchup(a, b, envs, threshold)
        points[a.name] += ap
        points[b.name] += bp
    return points


def monte_carlo(miners: list[Miner], envs: list[str], threshold: float | None, n_sims: int = NUM_SIMS):
    totals = {m.name: 0.0 for m in miners}
    first_place = {m.name: 0 for m in miners}
    top_half = {m.name: 0 for m in miners}  # top 3 of 6

    for _ in range(n_sims):
        pts = run_group(miners, envs, threshold)
        for name, p in pts.items():
            totals[name] += p
        ranked = sorted(pts.items(), key=lambda x: x[1], reverse=True)
        first_place[ranked[0][0]] += 1
        for name, _ in ranked[:3]:
            top_half[name] += 1

    avg = {n: totals[n] / n_sims for n in totals}
    wr = {n: first_place[n] / n_sims for n in first_place}
    th = {n: top_half[n] / n_sims for n in top_half}
    return avg, wr, th


def print_results(label: str, avg: dict, wr: dict, th: dict):
    print(f"\n  {label}")
    print(f"  {'Miner':<25} {'Avg Pts':>8} {'1st %':>8} {'Top 3 %':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for name in sorted(avg, key=avg.get, reverse=True):
        print(f"  {name:<25} {avg[name]:>8.1f} {wr[name]*100:>7.1f}% {th[name]*100:>7.1f}%")


def header(title: str, desc: str):
    print(f"\n{'#'*70}")
    print(f"# {title}")
    print(f"# {desc}")
    print(f"{'#'*70}")


def run_both(miners: list[Miner], envs: list[str], threshold: float = 0.20):
    avg, wr, th = monte_carlo(miners, envs, threshold=None)
    print_results("No threshold", avg, wr, th)
    avg, wr, th = monte_carlo(miners, envs, threshold=threshold)
    print_results(f"With {threshold:.0%} floor", avg, wr, th)


# =====================================================================
print("\n" + "=" * 70)
print("  PART 1: 6 MINERS, 2 ENVIRONMENTS")
print("=" * 70)

# --- 1A: One specialist, five balanced ---
header("1A: One specialist vs five balanced (2 envs)",
       "A: poker=0.15 rummy=0.85. Others: 0.55/0.55")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55}) for i in range(2, 7)],
]
run_both(miners, ENVS_2)

# --- 1B: One specialist, varied field ---
header("1B: Specialist vs varied field (2 envs)",
       "A: 0.15/0.85. B: 0.70/0.45. C: 0.60/0.55. D: 0.50/0.60. E: 0.55/0.50. F: 0.45/0.65")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85}),
    Miner("B_poker_lean", {"poker": 0.70, "rummy": 0.45}),
    Miner("C_slight_pok", {"poker": 0.60, "rummy": 0.55}),
    Miner("D_slight_rum", {"poker": 0.50, "rummy": 0.60}),
    Miner("E_balanced", {"poker": 0.55, "rummy": 0.50}),
    Miner("F_rum_lean", {"poker": 0.45, "rummy": 0.65}),
]
run_both(miners, ENVS_2)

# --- 1C: Two specialists on opposite games ---
header("1C: Two specialists, opposite games (2 envs)",
       "A: 0.15/0.85. B: 0.85/0.15. Rest: 0.55/0.55")
miners = [
    Miner("A_rummy_spec", {"poker": 0.15, "rummy": 0.85}),
    Miner("B_poker_spec", {"poker": 0.85, "rummy": 0.15}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55}) for i in range(3, 7)],
]
run_both(miners, ENVS_2)

# --- 1D: Specialist who's also okay at the other game ---
header("1D: Mild specialist (not terrible at weak game) (2 envs)",
       "A: poker=0.40 rummy=0.80. Others: 0.55/0.55")
miners = [
    Miner("A_mild_spec", {"poker": 0.40, "rummy": 0.80}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55}) for i in range(2, 7)],
]
run_both(miners, ENVS_2)

# --- 1E: Specialist in a field where everyone is bad at rummy ---
header("1E: Only good rummy player, everyone else weak at rummy (2 envs)",
       "A: 0.15/0.85. Others: poker=0.60, rummy=0.35")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85}),
    *[Miner(f"G{i}_pok_only", {"poker": 0.60, "rummy": 0.35}) for i in range(2, 7)],
]
run_both(miners, ENVS_2)

# --- 1F: Crowded field, one game saturated ---
header("1F: Saturated poker field, specialist finds edge in rummy (2 envs)",
       "A: 0.15/0.85. B-F all strong poker (0.70) but mediocre rummy (0.45)")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85}),
    *[Miner(f"G{i}_poker_str", {"poker": 0.70, "rummy": 0.45}) for i in range(2, 7)],
]
run_both(miners, ENVS_2)

# =====================================================================
print("\n\n" + "=" * 70)
print("  PART 2: 6 MINERS, 3 ENVIRONMENTS")
print("=" * 70)

# --- 2A: One specialist in 3 envs ---
header("2A: One specialist vs five balanced (3 envs)",
       "A: 0.15/0.85/0.15. Others: 0.55 all")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85, "liar_dice": 0.15}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55, "liar_dice": 0.55}) for i in range(2, 7)],
]
run_both(miners, ENVS_3)

# --- 2B: Specialist in 2 of 3 games ---
header("2B: Strong in 2, weak in 1 (3 envs)",
       "A: poker=0.15 rummy=0.80 liar_dice=0.80. Others: 0.55 all")
miners = [
    Miner("A_two_strong", {"poker": 0.15, "rummy": 0.80, "liar_dice": 0.80}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55, "liar_dice": 0.55}) for i in range(2, 7)],
]
run_both(miners, ENVS_3)

# --- 2C: Specialist vs varied field (3 envs) ---
header("2C: Specialist vs varied field (3 envs)",
       "A: 0.15/0.85/0.15. Mix of strengths across field.")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85, "liar_dice": 0.15}),
    Miner("B_poker_str", {"poker": 0.70, "rummy": 0.45, "liar_dice": 0.55}),
    Miner("C_balanced", {"poker": 0.55, "rummy": 0.55, "liar_dice": 0.55}),
    Miner("D_liar_str", {"poker": 0.50, "rummy": 0.45, "liar_dice": 0.70}),
    Miner("E_rum_lean", {"poker": 0.45, "rummy": 0.65, "liar_dice": 0.50}),
    Miner("F_allround", {"poker": 0.60, "rummy": 0.55, "liar_dice": 0.60}),
]
run_both(miners, ENVS_3)

# --- 2D: Two specialists splitting 3 games ---
header("2D: Each specialist owns different games (3 envs)",
       "A: rummy=0.85 (rest 0.15). B: poker=0.85 (rest 0.15). Rest: 0.55 all")
miners = [
    Miner("A_rummy_spec", {"poker": 0.15, "rummy": 0.85, "liar_dice": 0.15}),
    Miner("B_poker_spec", {"poker": 0.85, "rummy": 0.15, "liar_dice": 0.15}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55, "liar_dice": 0.55}) for i in range(3, 7)],
]
run_both(miners, ENVS_3)

# --- 2E: Everyone specializes in one game ---
header("2E: Everyone picks a game (3 envs)",
       "A,B: rummy=0.80 rest=0.30. C,D: poker=0.80 rest=0.30. E,F: liar=0.80 rest=0.30")
miners = [
    Miner("A_rummy", {"poker": 0.30, "rummy": 0.80, "liar_dice": 0.30}),
    Miner("B_rummy", {"poker": 0.30, "rummy": 0.80, "liar_dice": 0.30}),
    Miner("C_poker", {"poker": 0.80, "rummy": 0.30, "liar_dice": 0.30}),
    Miner("D_poker", {"poker": 0.80, "rummy": 0.30, "liar_dice": 0.30}),
    Miner("E_liar", {"poker": 0.30, "rummy": 0.30, "liar_dice": 0.80}),
    Miner("F_liar", {"poker": 0.30, "rummy": 0.30, "liar_dice": 0.80}),
]
run_both(miners, ENVS_3)

# =====================================================================
print("\n\n" + "=" * 70)
print("  PART 3: WORST CASE HUNTING — when does the specialist ACTUALLY win?")
print("=" * 70)

# --- 3A: Weak field ---
header("3A: Specialist vs very weak field (2 envs)",
       "A: 0.15/0.85. Others: 0.45/0.45 (below average at everything)")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85}),
    *[Miner(f"G{i}_weak", {"poker": 0.45, "rummy": 0.45}) for i in range(2, 7)],
]
run_both(miners, ENVS_2)

# --- 3B: Specialist is the BEST player overall ---
header("3B: Specialist with highest total skill budget (2 envs)",
       "A: 0.20/0.90 (total=1.10). Others: 0.55/0.55 (total=1.10). Same skill budget.")
miners = [
    Miner("A_specialist", {"poker": 0.20, "rummy": 0.90}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55}) for i in range(2, 7)],
]
run_both(miners, ENVS_2)

# --- 3C: Field is all very close, specialist is only edge ---
header("3C: Extremely tight field, specialist has only differentiator (2 envs)",
       "A: 0.15/0.85. Others: 0.52/0.52 (barely above average)")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85}),
    *[Miner(f"G{i}_tight", {"poker": 0.52, "rummy": 0.52}) for i in range(2, 7)],
]
run_both(miners, ENVS_2)

# --- 3D: Specialist + one strong all-rounder ---
header("3D: Specialist vs one strong all-rounder + average field (2 envs)",
       "A: 0.15/0.85. B: 0.70/0.70. C-F: 0.50/0.50")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85}),
    Miner("B_strong_all", {"poker": 0.70, "rummy": 0.70}),
    *[Miner(f"G{i}_average", {"poker": 0.50, "rummy": 0.50}) for i in range(3, 7)],
]
run_both(miners, ENVS_2)

# --- 3E: The one where everyone is REALLY bad at rummy ---
header("3E: Everyone terrible at rummy except specialist (2 envs)",
       "A: 0.15/0.85. Others: poker=0.65, rummy=0.20")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85}),
    *[Miner(f"G{i}_no_rummy", {"poker": 0.65, "rummy": 0.20}) for i in range(2, 7)],
]
run_both(miners, ENVS_2)

# =====================================================================
print("\n\n" + "=" * 70)
print("  PART 4: THRESHOLD SENSITIVITY (6 miners, worst cases)")
print("=" * 70)

# Sweep for the case where specialist actually does well
header("4A: Threshold sweep — specialist vs weak rummy field (2 envs)",
       "A: 0.15/0.85 vs others at 0.65/0.20")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85}),
    *[Miner(f"G{i}_no_rummy", {"poker": 0.65, "rummy": 0.20}) for i in range(2, 7)],
]
print(f"\n  {'Threshold':>10} {'A 1st%':>8} {'A top3%':>8} {'A avg':>8} {'G2 avg':>8}")
print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for t in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    avg, wr, th = monte_carlo(miners, ENVS_2, threshold=t if t > 0 else None, n_sims=3000)
    print(f"  {t:>9.0%} {wr['A_specialist']*100:>7.1f}% {th['A_specialist']*100:>7.1f}% {avg['A_specialist']:>8.1f} {avg['G2_no_rummy']:>8.1f}")

header("4B: Threshold sweep — specialist vs balanced (2 envs)",
       "A: 0.15/0.85 vs others at 0.55/0.55")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55}) for i in range(2, 7)],
]
print(f"\n  {'Threshold':>10} {'A 1st%':>8} {'A top3%':>8} {'A avg':>8} {'G2 avg':>8}")
print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for t in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    avg, wr, th = monte_carlo(miners, ENVS_2, threshold=t if t > 0 else None, n_sims=3000)
    print(f"  {t:>9.0%} {wr['A_specialist']*100:>7.1f}% {th['A_specialist']*100:>7.1f}% {avg['A_specialist']:>8.1f} {avg['G2_balanced']:>8.1f}")

header("4C: Threshold sweep — specialist vs balanced (3 envs)",
       "A: 0.15/0.85/0.15 vs others at 0.55 all")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85, "liar_dice": 0.15}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55, "liar_dice": 0.55}) for i in range(2, 7)],
]
print(f"\n  {'Threshold':>10} {'A 1st%':>8} {'A top3%':>8} {'A avg':>8} {'G2 avg':>8}")
print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for t in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    avg, wr, th = monte_carlo(miners, ENVS_3, threshold=t if t > 0 else None, n_sims=3000)
    print(f"  {t:>9.0%} {wr['A_specialist']*100:>7.1f}% {th['A_specialist']*100:>7.1f}% {avg['A_specialist']:>8.1f} {avg['G2_balanced']:>8.1f}")


print("\n\n" + "=" * 70)
print(f"  Done. GAMES_PER_ENV={GAMES_PER_ENV}, SIMS={NUM_SIMS}")
print("=" * 70)
