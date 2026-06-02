"""Backtest: old 3/1/0 pairwise scoring vs new dispersion-weighted scoring.

Replays real round-1 data from environment tournament tourn_33c30659e3c920d5_20260601
(liars_dice head-to-head + intercode individual scores) and prints how each group's
winner and ordering change under the new scheme.

Run: .venv/bin/python scripts/sim_env_scoring.py
"""

import statistics

from core.constants import EnvironmentName
from core.models.pvp_models import PvPEnvironmentResult
from core.models.pvp_models import PvPEvalMetadata
from core.models.pvp_models import PvPGroupResults
from core.models.pvp_models import PvPPairResult
from core.models.scoring_models import EnvironmentWeight
from validator.evaluation.tournament_scoring import accumulate_points
from validator.evaluation.tournament_scoring import dispersion_weighted_standings
from validator.evaluation.tournament_scoring import individual_scores_to_pairwise
from validator.evaluation.tournament_scoring import pvp_results_to_pairwise
from validator.evaluation.tournament_scoring import pvp_results_to_winrates


LD = EnvironmentName.LIARS_DICE
IC = EnvironmentName.INTERCODE
WEIGHTS = [EnvironmentWeight(environment=LD, weight=1.0), EnvironmentWeight(environment=IC, weight=1.0)]

# (hotkeys, liars_dice pairs (a, b, a_wins, b_wins), intercode scores)
GROUPS = {
    "group_001": (
        ["CMZQwPd", "CRwFq2Q", "EF2nASX", "FpdSckw", "GNP9XWd", "HWPK9f6"],
        [
            ("CMZQwPd", "CRwFq2Q", 0, 200), ("CMZQwPd", "EF2nASX", 0, 200), ("CMZQwPd", "FpdSckw", 0, 200),
            ("CMZQwPd", "GNP9XWd", 2, 198), ("CMZQwPd", "HWPK9f6", 111, 89), ("CRwFq2Q", "EF2nASX", 1, 199),
            ("CRwFq2Q", "FpdSckw", 1, 199), ("CRwFq2Q", "GNP9XWd", 35, 165), ("CRwFq2Q", "HWPK9f6", 138, 62),
            ("EF2nASX", "FpdSckw", 90, 110), ("EF2nASX", "GNP9XWd", 134, 66), ("EF2nASX", "HWPK9f6", 200, 0),
            ("FpdSckw", "GNP9XWd", 129, 71), ("FpdSckw", "HWPK9f6", 198, 2), ("GNP9XWd", "HWPK9f6", 200, 0),
        ],
        {"CMZQwPd": 0.791, "CRwFq2Q": 0.749, "EF2nASX": 0.729, "GNP9XWd": 0.721, "HWPK9f6": 0.719, "FpdSckw": 0.709},
    ),
    "group_002": (
        ["CoNpVXZ", "CUgn1rt", "D2Qee4V", "EgpWgYv", "FRdgPRd", "HKEAZxF"],
        [
            ("CoNpVXZ", "D2Qee4V", 1, 199), ("CoNpVXZ", "EgpWgYv", 1, 199), ("CoNpVXZ", "FRdgPRd", 1, 199),
            ("CoNpVXZ", "HKEAZxF", 2, 198), ("CUgn1rt", "CoNpVXZ", 200, 0), ("CUgn1rt", "D2Qee4V", 1, 199),
            ("CUgn1rt", "EgpWgYv", 1, 199), ("CUgn1rt", "FRdgPRd", 1, 199), ("CUgn1rt", "HKEAZxF", 1, 199),
            ("D2Qee4V", "EgpWgYv", 99, 101), ("D2Qee4V", "FRdgPRd", 105, 95), ("D2Qee4V", "HKEAZxF", 200, 0),
            ("EgpWgYv", "FRdgPRd", 107, 93), ("EgpWgYv", "HKEAZxF", 121, 79), ("FRdgPRd", "HKEAZxF", 131, 69),
        ],
        {"D2Qee4V": 0.76, "HKEAZxF": 0.74, "FRdgPRd": 0.73, "CUgn1rt": 0.72, "EgpWgYv": 0.72, "CoNpVXZ": 0.71},
    ),
    "group_006": (
        ["Ca32LwM", "CyKEP2X", "CZtSFWz", "FWLwQ2G", "GbMFGZS"],
        [
            ("Ca32LwM", "CyKEP2X", 198, 2), ("Ca32LwM", "FWLwQ2G", 2, 198), ("Ca32LwM", "GbMFGZS", 200, 0),
            ("CyKEP2X", "FWLwQ2G", 0, 200), ("CyKEP2X", "GbMFGZS", 200, 0), ("CZtSFWz", "Ca32LwM", 1, 199),
            ("CZtSFWz", "CyKEP2X", 75, 125), ("CZtSFWz", "FWLwQ2G", 1, 199), ("CZtSFWz", "GbMFGZS", 200, 0),
            ("FWLwQ2G", "GbMFGZS", 200, 0),
        ],
        # GbMFGZS topped intercode (0.80) but went 0-4 at dice
        {"GbMFGZS": 0.80, "FWLwQ2G": 0.76, "CyKEP2X": 0.74, "CZtSFWz": 0.73, "Ca32LwM": 0.73},
    ),
}


def _group(hotkeys, pairs):
    pair_results = [
        PvPPairResult(
            hotkey_a=a, hotkey_b=b,
            results={LD: PvPEnvironmentResult(model_a_wins=aw, model_b_wins=bw, draws=0, total_games=aw + bw)},
        )
        for a, b, aw, bw in pairs
    ]
    return PvPGroupResults(base_model="base", hotkeys=hotkeys, pair_results=pair_results,
                           metadata=PvPEvalMetadata(seed=42, temperature=0.0))


def old_standings(g, intercode, hotkeys):
    outcomes = pvp_results_to_pairwise(g) + individual_scores_to_pairwise(intercode, IC)
    return accumulate_points(outcomes, hotkeys, WEIGHTS)


def new_standings(g, intercode, hotkeys):
    wr = pvp_results_to_winrates(g)[LD]
    return dispersion_weighted_standings({LD: wr, IC: intercode}, hotkeys, weights=WEIGHTS), wr


def main():
    for name, (hotkeys, pairs, intercode) in GROUPS.items():
        g = _group(hotkeys, pairs)
        old = old_standings(g, intercode, hotkeys)
        new, wr = new_standings(g, intercode, hotkeys)
        std_ld = statistics.pstdev(wr.values())
        std_ic = statistics.pstdev(intercode.values())
        ratio = std_ld / std_ic if std_ic else float("inf")

        print(f"\n=== {name} ===")
        print(f"  dispersion: std(dice win-rate)={std_ld:.3f}  std(intercode)={std_ic:.3f}  "
              f"-> dice influence {ratio:.1f}x intercode")
        print(f"  OLD winner: {old[0].hotkey:9s} (dice rank "
              f"{sorted(wr, key=lambda h: -wr[h]).index(old[0].hotkey) + 1})")
        print(f"  NEW winner: {new[0].hotkey:9s} (dice rank "
              f"{sorted(wr, key=lambda h: -wr[h]).index(new[0].hotkey) + 1})")
        flip = "  >>> WINNER CHANGED" if old[0].hotkey != new[0].hotkey else "  (same winner)"
        print(flip)
        print("  ranking  OLD:", " > ".join(s.hotkey for s in old))
        print("  ranking  NEW:", " > ".join(s.hotkey for s in new))


if __name__ == "__main__":
    main()
