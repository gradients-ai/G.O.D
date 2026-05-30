"""
PvP threshold sim v3 — dominant player scenarios.

Only top 1 advances per group. 6 miners, 2-3 envs.
Focus: what happens when one player wins 100% in one environment?
Does the 20% floor break the group?
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
    # Track how often each miner gets 0 total points
    zero_rounds = {m.name: 0 for m in miners}

    for _ in range(n_sims):
        pts = run_group(miners, envs, threshold)
        for name, p in pts.items():
            totals[name] += p
            if p == 0:
                zero_rounds[name] += 1
        ranked = sorted(pts.items(), key=lambda x: x[1], reverse=True)
        first_place[ranked[0][0]] += 1

    avg = {n: totals[n] / n_sims for n in totals}
    wr = {n: first_place[n] / n_sims for n in first_place}
    zr = {n: zero_rounds[n] / n_sims for n in zero_rounds}
    return avg, wr, zr


def print_results(label: str, avg: dict, wr: dict, zr: dict):
    print(f"\n  {label}")
    print(f"  {'Miner':<25} {'Avg Pts':>8} {'1st %':>8} {'Zero %':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for name in sorted(avg, key=avg.get, reverse=True):
        print(f"  {name:<25} {avg[name]:>8.1f} {wr[name]*100:>7.1f}% {zr[name]*100:>7.1f}%")


def header(title: str, desc: str):
    print(f"\n{'#'*70}")
    print(f"# {title}")
    print(f"# {desc}")
    print(f"{'#'*70}")


def run_all(miners: list[Miner], envs: list[str], threshold: float = 0.20):
    avg, wr, zr = monte_carlo(miners, envs, threshold=None)
    print_results("No threshold (top 1 advances)", avg, wr, zr)
    avg, wr, zr = monte_carlo(miners, envs, threshold=threshold)
    print_results(f"With {threshold:.0%} floor (top 1 advances)", avg, wr, zr)


# =====================================================================
print("=" * 70)
print("  DOMINANT PLAYER SCENARIOS — 6 miners, top 1 advances")
print("=" * 70)

# --- A: Dominant rummy player, balanced field, 2 envs ---
header("A: Dominant rummy player (skill=0.99) vs balanced field (2 envs)",
       "D wins ~100% of rummy games. Balanced at 0.55. D bad at poker (0.15)")
miners = [
    Miner("D_dominant", {"poker": 0.15, "rummy": 0.99}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55}) for i in range(2, 7)],
]
run_all(miners, ENVS_2)

# --- B: Dominant rummy, but others are also decent at rummy ---
header("B: Dominant rummy vs field that also plays rummy (2 envs)",
       "D: 0.15/0.99. Others: 0.55/0.65 (decent at rummy too)")
miners = [
    Miner("D_dominant", {"poker": 0.15, "rummy": 0.99}),
    *[Miner(f"G{i}_rum_ok", {"poker": 0.55, "rummy": 0.65}) for i in range(2, 7)],
]
run_all(miners, ENVS_2)

# --- C: Dominant rummy, field weak at rummy (worst case for threshold) ---
header("C: Dominant rummy vs field terrible at rummy (2 envs)",
       "D: 0.15/0.99. Others: 0.65/0.25 (can't play rummy at all)")
miners = [
    Miner("D_dominant", {"poker": 0.15, "rummy": 0.99}),
    *[Miner(f"G{i}_no_rum", {"poker": 0.65, "rummy": 0.25}) for i in range(2, 7)],
]
run_all(miners, ENVS_2)

# --- D: Dominant AND good at other game ---
header("D: Dominant rummy AND decent poker (2 envs)",
       "D: 0.50/0.99. Others: 0.55/0.55")
miners = [
    Miner("D_dom_decent", {"poker": 0.50, "rummy": 0.99}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55}) for i in range(2, 7)],
]
run_all(miners, ENVS_2)

# --- E: Dominant AND strong at other game ---
header("E: Dominant rummy AND strong poker (2 envs)",
       "D: 0.70/0.99. Others: 0.55/0.55. D is just a better miner overall.")
miners = [
    Miner("D_dom_strong", {"poker": 0.70, "rummy": 0.99}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55}) for i in range(2, 7)],
]
run_all(miners, ENVS_2)

# --- F: 3 envs, dominant in one ---
header("F: Dominant in rummy, weak in other two (3 envs)",
       "D: 0.15/0.99/0.15. Others: 0.55 all")
miners = [
    Miner("D_dominant", {"poker": 0.15, "rummy": 0.99, "liar_dice": 0.15}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55, "liar_dice": 0.55}) for i in range(2, 7)],
]
run_all(miners, ENVS_3)

# --- G: 3 envs, dominant in one, decent in rest ---
header("G: Dominant rummy, decent elsewhere (3 envs)",
       "D: 0.50/0.99/0.50. Others: 0.55 all")
miners = [
    Miner("D_dom_decent", {"poker": 0.50, "rummy": 0.99, "liar_dice": 0.50}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55, "liar_dice": 0.55}) for i in range(2, 7)],
]
run_all(miners, ENVS_3)


# =====================================================================
print("\n\n" + "=" * 70)
print("  COLLATERAL DAMAGE — how much does the dominant player")
print("  distort OTHER miners' rankings?")
print("=" * 70)

# Compare: who wins the group WITH vs WITHOUT the dominant player?
header("H: Does the dominant player change who advances?",
       "Compare group winner with and without D_dominant present")

# With dominant player
miners_with = [
    Miner("D_dominant", {"poker": 0.15, "rummy": 0.99}),
    Miner("G2", {"poker": 0.55, "rummy": 0.55}),
    Miner("G3", {"poker": 0.60, "rummy": 0.50}),
    Miner("G4", {"poker": 0.50, "rummy": 0.60}),
    Miner("G5", {"poker": 0.65, "rummy": 0.45}),
    Miner("G6", {"poker": 0.45, "rummy": 0.65}),
]
# Without dominant player
miners_without = [m for m in miners_with if m.name != "D_dominant"]

print("\n  --- WITH dominant player in group ---")
for thresh in [None, 0.20]:
    avg, wr, zr = monte_carlo(miners_with, ENVS_2, threshold=thresh)
    label = "No threshold" if thresh is None else f"{thresh:.0%} floor"
    print_results(label, avg, wr, zr)

print("\n  --- WITHOUT dominant player (5 miners) ---")
for thresh in [None, 0.20]:
    avg, wr, zr = monte_carlo(miners_without, ENVS_2, threshold=thresh)
    label = "No threshold" if thresh is None else f"{thresh:.0%} floor"
    print_results(label, avg, wr, zr)


# =====================================================================
print("\n\n" + "=" * 70)
print("  ALTERNATIVE: One-sided floor (only YOUR score checked)")
print("  You need 20% in each env to earn YOUR points.")
print("  Opponent's floor doesn't affect YOUR points.")
print("=" * 70)


def score_matchup_onesided(a: Miner, b: Miner, envs: list[str], threshold: float | None) -> tuple[float, float]:
    """Each miner's floor is checked independently — failing only zeros YOUR points."""
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


def score_matchup_selfonly(a: Miner, b: Miner, envs: list[str], threshold: float | None) -> tuple[float, float]:
    """Only check YOUR OWN floor. Opponent failing doesn't affect you."""
    a_pts = b_pts = 0.0
    a_rates = {}
    b_rates = {}

    for env in envs:
        aw, bw = simulate_games(a.env_skill.get(env, 0.5), b.env_skill.get(env, 0.5), GAMES_PER_ENV)
        total = aw + bw
        if total > 0:
            a_rates[env] = aw / total
            b_rates[env] = bw / total
        else:
            a_rates[env] = 0.5
            b_rates[env] = 0.5

        if aw > bw:
            a_pts += WIN_PTS
        elif bw > aw:
            b_pts += WIN_PTS
        else:
            a_pts += DRAW_PTS
            b_pts += DRAW_PTS

    if threshold:
        if any(r < threshold for r in a_rates.values()):
            a_pts = 0
        if any(r < threshold for r in b_rates.values()):
            b_pts = 0
    return a_pts, b_pts


def run_group_selfonly(miners: list[Miner], envs: list[str], threshold: float | None) -> dict[str, float]:
    points = {m.name: 0.0 for m in miners}
    for a, b in itertools.combinations(miners, 2):
        ap, bp = score_matchup_selfonly(a, b, envs, threshold)
        points[a.name] += ap
        points[b.name] += bp
    return points


def monte_carlo_selfonly(miners: list[Miner], envs: list[str], threshold: float | None, n_sims: int = NUM_SIMS):
    totals = {m.name: 0.0 for m in miners}
    first_place = {m.name: 0 for m in miners}
    zero_rounds = {m.name: 0 for m in miners}

    for _ in range(n_sims):
        pts = run_group_selfonly(miners, envs, threshold)
        for name, p in pts.items():
            totals[name] += p
            if p == 0:
                zero_rounds[name] += 1
        ranked = sorted(pts.items(), key=lambda x: x[1], reverse=True)
        first_place[ranked[0][0]] += 1

    avg = {n: totals[n] / n_sims for n in totals}
    wr = {n: first_place[n] / n_sims for n in first_place}
    zr = {n: zero_rounds[n] / n_sims for n in zero_rounds}
    return avg, wr, zr


# Re-run the problem scenarios with self-only floor
header("I: Dominant rummy vs balanced — self-only floor (2 envs)",
       "D: 0.15/0.99. Others: 0.55/0.55. Floor only checks YOUR game win rates.")
miners = [
    Miner("D_dominant", {"poker": 0.15, "rummy": 0.99}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55}) for i in range(2, 7)],
]
avg, wr, zr = monte_carlo(miners, ENVS_2, threshold=None)
print_results("No threshold", avg, wr, zr)
avg, wr, zr = monte_carlo(miners, ENVS_2, threshold=0.20)
print_results("Both-sides 20% floor", avg, wr, zr)
avg, wr, zr = monte_carlo_selfonly(miners, ENVS_2, threshold=0.20)
print_results("Self-only 20% floor", avg, wr, zr)

header("J: Dominant rummy vs weak rummy field — self-only floor (2 envs)",
       "D: 0.15/0.99. Others: 0.65/0.25. Floor only checks YOUR rates.")
miners = [
    Miner("D_dominant", {"poker": 0.15, "rummy": 0.99}),
    *[Miner(f"G{i}_no_rum", {"poker": 0.65, "rummy": 0.25}) for i in range(2, 7)],
]
avg, wr, zr = monte_carlo(miners, ENVS_2, threshold=None)
print_results("No threshold", avg, wr, zr)
avg, wr, zr = monte_carlo(miners, ENVS_2, threshold=0.20)
print_results("Both-sides 20% floor", avg, wr, zr)
avg, wr, zr = monte_carlo_selfonly(miners, ENVS_2, threshold=0.20)
print_results("Self-only 20% floor", avg, wr, zr)

header("K: Genuinely good player — self-only floor shouldn't hurt (2 envs)",
       "D: 0.70/0.99. Others: 0.55/0.55. D deserves to win.")
miners = [
    Miner("D_dom_strong", {"poker": 0.70, "rummy": 0.99}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55}) for i in range(2, 7)],
]
avg, wr, zr = monte_carlo(miners, ENVS_2, threshold=None)
print_results("No threshold", avg, wr, zr)
avg, wr, zr = monte_carlo(miners, ENVS_2, threshold=0.20)
print_results("Both-sides 20% floor", avg, wr, zr)
avg, wr, zr = monte_carlo_selfonly(miners, ENVS_2, threshold=0.20)
print_results("Self-only 20% floor", avg, wr, zr)

header("L: Specialist exploit — does self-only floor still block it? (2 envs)",
       "A: 0.15/0.85. Others: 0.55/0.55. The original exploit case.")
miners = [
    Miner("A_specialist", {"poker": 0.15, "rummy": 0.85}),
    *[Miner(f"G{i}_balanced", {"poker": 0.55, "rummy": 0.55}) for i in range(2, 7)],
]
avg, wr, zr = monte_carlo(miners, ENVS_2, threshold=None)
print_results("No threshold", avg, wr, zr)
avg, wr, zr = monte_carlo(miners, ENVS_2, threshold=0.20)
print_results("Both-sides 20% floor", avg, wr, zr)
avg, wr, zr = monte_carlo_selfonly(miners, ENVS_2, threshold=0.20)
print_results("Self-only 20% floor", avg, wr, zr)


print("\n\n" + "=" * 70)
print(f"  Done. GAMES_PER_ENV={GAMES_PER_ENV}, SIMS={NUM_SIMS}")
print("=" * 70)
