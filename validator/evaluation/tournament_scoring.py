
import itertools

import validator.core.constants as cts
from core.constants import EnvironmentName
from core.models.pvp_models import PvPGroupResults
from core.models.scoring_models import EnvironmentWeight
from core.models.scoring_models import GroupStagePoints
from core.models.scoring_models import PairwiseOutcome
from core.models.scoring_models import TournamentScore
from core.models.scoring_models import TournamentTypeResult
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentType
from validator.utils.logging import get_logger


logger = get_logger(__name__)


# --- Universal pairwise scoring ---


def accumulate_points(
    outcomes: list[PairwiseOutcome],
    hotkeys: list[str],
    weights: list[EnvironmentWeight] | None = None,
) -> list[GroupStagePoints]:
    """Universal 3/1/0 accumulator. Works with outcomes from any eval type.

    Weights are per-environment multipliers (default 1.0 = uniform).
    Returns list sorted by points descending.
    """
    weight_map = {w.environment: w.weight for w in weights} if weights else {}
    points = {hotkey: 0.0 for hotkey in hotkeys}

    for outcome in outcomes:
        weight = weight_map.get(outcome.environment, 1.0)

        if outcome.winner == outcome.hotkey_a:
            points[outcome.hotkey_a] += cts.PVP_ENV_WIN_POINTS * weight
        elif outcome.winner == outcome.hotkey_b:
            points[outcome.hotkey_b] += cts.PVP_ENV_WIN_POINTS * weight
        elif outcome.winner is None:
            points[outcome.hotkey_a] += cts.PVP_ENV_DRAW_POINTS * weight
            points[outcome.hotkey_b] += cts.PVP_ENV_DRAW_POINTS * weight

    standings = [GroupStagePoints(hotkey=hk, points=pts) for hk, pts in points.items()]
    standings.sort(key=lambda s: s.points, reverse=True)
    return standings


# --- PvP → pairwise ---


def pvp_results_to_pairwise(group_results: PvPGroupResults) -> list[PairwiseOutcome]:
    """Convert PvP group round-robin results into universal pairwise outcomes."""
    outcomes: list[PairwiseOutcome] = []

    for pair in group_results.pair_results:
        for env_name, env_result in pair.results.items():
            if env_result.model_a_wins > env_result.model_b_wins:
                winner = pair.hotkey_a
            elif env_result.model_b_wins > env_result.model_a_wins:
                winner = pair.hotkey_b
            else:
                winner = None

            outcomes.append(PairwiseOutcome(
                hotkey_a=pair.hotkey_a,
                hotkey_b=pair.hotkey_b,
                environment=env_name,
                winner=winner,
            ))

    return outcomes


# --- MCTS scores → pairwise ---


def mcts_scores_to_pairwise(
    scores_by_hotkey: dict[str, float],
    environment: EnvironmentName,
    win_margin: float = cts.MCTS_WIN_MARGIN,
) -> list[PairwiseOutcome]:
    """Convert independent MCTS scores into pairwise outcomes.

    A must exceed B by win_margin (fractional) to count as a win.
    Scores within the margin = draw.
    """
    hotkeys = list(scores_by_hotkey.keys())
    outcomes: list[PairwiseOutcome] = []

    for hotkey_a, hotkey_b in itertools.combinations(hotkeys, 2):
        score_a = scores_by_hotkey[hotkey_a]
        score_b = scores_by_hotkey[hotkey_b]
        threshold = abs(score_b) * win_margin

        if score_a > score_b + threshold:
            winner = hotkey_a
        elif score_b > score_a + threshold:
            winner = hotkey_b
        else:
            winner = None

        outcomes.append(PairwiseOutcome(
            hotkey_a=hotkey_a,
            hotkey_b=hotkey_b,
            environment=environment,
            winner=winner,
        ))

    return outcomes


# --- Convenience: PvP group results → standings ---


def compute_pvp_tournament_points(
    group_results: PvPGroupResults,
    weights: list[EnvironmentWeight] | None = None,
) -> list[GroupStagePoints]:
    """Convert PvP group results into per-hotkey tournament points.

    Convenience wrapper: pvp_results_to_pairwise → accumulate_points.
    """
    outcomes = pvp_results_to_pairwise(group_results)
    return accumulate_points(outcomes, group_results.hotkeys, weights)


def calculate_tournament_type_scores_from_data(
    tournament_type: TournamentType, tournament_data: TournamentResultsWithWinners | None
) -> TournamentTypeResult:
    """Calculate tournament scores from tournament data without database access."""
    if not tournament_data:
        return TournamentTypeResult(scores=[], prev_winner_hotkey=None, prev_winner_won_final=False)

    if tournament_type == TournamentType.TEXT:
        type_weight = cts.TOURNAMENT_TEXT_WEIGHT
    elif tournament_type == TournamentType.IMAGE:
        type_weight = cts.TOURNAMENT_IMAGE_WEIGHT
    elif tournament_type == TournamentType.ENVIRONMENT:
        type_weight = cts.TOURNAMENT_ENVIRONMENT_WEIGHT
    else:
        raise ValueError(f"Unknown tournament type: {tournament_type}")
    score_dict = {}
    prev_winner_won_final = False

    # Resolve EMISSION_BURN_HOTKEY placeholder to the real defending champion hotkey
    winner_hk = tournament_data.winner_hotkey
    base_hk = tournament_data.base_winner_hotkey
    if winner_hk == cts.EMISSION_BURN_HOTKEY and base_hk:
        actual_winner_hotkey = base_hk
    else:
        actual_winner_hotkey = winner_hk
    if tournament_data.winner_hotkey == cts.EMISSION_BURN_HOTKEY and tournament_data.base_winner_hotkey:
        logger.info(f"Swapped EMISSION_BURN_HOTKEY with actual defending champion: {actual_winner_hotkey}")

    for round_result in tournament_data.rounds:
        round_number = round_result.round_number
        is_final_round = round_result.is_final_round

        for task in round_result.tasks:
            winner = task.winner

            if is_final_round and actual_winner_hotkey and winner == actual_winner_hotkey:
                prev_winner_won_final = True

            # Also check if winner is EMISSION_BURN_HOTKEY (placeholder for defending champion)
            if is_final_round and winner == cts.EMISSION_BURN_HOTKEY and tournament_data.base_winner_hotkey:
                prev_winner_won_final = True

            if tournament_type == TournamentType.ENVIRONMENT:
                ranked_participants = []
                for participant in task.participant_scores:
                    hotkey = participant.get("hotkey")
                    test_loss = participant.get("test_loss")
                    if hotkey == actual_winner_hotkey:
                        continue
                    if hotkey == cts.EMISSION_BURN_HOTKEY and tournament_data.base_winner_hotkey:
                        continue
                    if test_loss is None or test_loss == 0:
                        continue
                    ranked_participants.append((hotkey, test_loss))

                ranked_participants.sort(key=lambda x: x[1], reverse=True)

                total_participants = len(ranked_participants)
                for rank, (hotkey, _) in enumerate(ranked_participants, start=1):
                    points = round_number * type_weight * (total_participants - rank + 1) / total_participants
                    if hotkey not in score_dict:
                        score_dict[hotkey] = 0
                    score_dict[hotkey] += points

            else:
                # Exclude both the actual winner and EMISSION_BURN_HOTKEY (if it's the placeholder) from earning points
                if (
                    winner
                    and winner != actual_winner_hotkey
                    and not (winner == cts.EMISSION_BURN_HOTKEY and tournament_data.base_winner_hotkey)
                ):
                    if winner not in score_dict:
                        score_dict[winner] = 0
                    score_dict[winner] += round_number * type_weight

    scores = [TournamentScore(hotkey=hotkey, score=score) for hotkey, score in score_dict.items()]

    return TournamentTypeResult(
        scores=scores, prev_winner_hotkey=actual_winner_hotkey, prev_winner_won_final=prev_winner_won_final
    )


def exponential_decline_mapping(total_participants: int, rank: float) -> float:
    """Exponential weight decay based on rank."""
    if total_participants <= 1:
        return 1.0

    # Calculate all weights for normalization
    all_weights = [cts.TOURNAMENT_SIMPLE_DECAY_BASE ** (r - 1) for r in range(1, total_participants + 1)]
    total_sum = sum(all_weights)

    # Return normalized weight to ensure sum = 1
    raw_weight = cts.TOURNAMENT_SIMPLE_DECAY_BASE ** (rank - 1)
    return raw_weight / total_sum


def tournament_scores_to_weights(
    tournament_scores: list[TournamentScore], prev_winner_hotkey: str | None, prev_winner_won_final: bool
) -> dict[str, float]:
    if not tournament_scores and not prev_winner_hotkey:
        return {}

    # Filter out zero scores
    non_zero_scores = [score for score in tournament_scores if score.score > 0]

    # If we have a previous winner, place them appropriately
    if prev_winner_hotkey:
        if prev_winner_won_final:
            # Previous winner won final round, place them 1st
            prev_winner_score = TournamentScore(hotkey=prev_winner_hotkey, score=float("inf"))
            non_zero_scores.insert(0, prev_winner_score)
        else:
            # Check if prev_winner is in the scores (meaning they participated and lost)
            # vs won by default (not in scores, won because others failed)
            prev_winner_in_scores = any(score.hotkey == prev_winner_hotkey for score in non_zero_scores)

            if prev_winner_in_scores:
                # Previous winner participated but lost final round, place them 2nd
                if len(non_zero_scores) > 0:
                    max_score = max(score.score for score in non_zero_scores)
                    prev_winner_score = TournamentScore(hotkey=prev_winner_hotkey, score=max_score - 0.1)
                    non_zero_scores.append(prev_winner_score)
            else:
                # Previous winner won by default (not in scores), place them 1st
                prev_winner_score = TournamentScore(hotkey=prev_winner_hotkey, score=float("inf"))
                non_zero_scores.insert(0, prev_winner_score)

    if not non_zero_scores:
        return {}

    # Group by score to handle ties
    score_groups = {}
    for tournament_score in non_zero_scores:
        score = tournament_score.score
        if score not in score_groups:
            score_groups[score] = []
        score_groups[score].append(tournament_score.hotkey)

    # Sort scores in descending order
    sorted_scores = sorted(score_groups.keys(), reverse=True)

    # Calculate weights
    total_participants = len(non_zero_scores)
    weights = {}

    current_rank = 1
    for score in sorted_scores:
        hotkeys_with_score = score_groups[score]

        # Calculate average rank for tied participants
        if len(hotkeys_with_score) == 1:
            avg_rank = current_rank
        else:
            avg_rank = current_rank + (len(hotkeys_with_score) - 1) / 2

        weight = exponential_decline_mapping(total_participants, avg_rank)

        # Assign same weight to all tied participants
        for hotkey in hotkeys_with_score:
            weights[hotkey] = weight

        current_rank += len(hotkeys_with_score)

    return weights


def _compute_weights(tournament_type: TournamentType, data: TournamentResultsWithWinners | None) -> dict[str, float]:
    result = calculate_tournament_type_scores_from_data(tournament_type, data)
    weights = (
        tournament_scores_to_weights(result.scores, result.prev_winner_hotkey, result.prev_winner_won_final)
        if result.scores
        else {}
    )
    logger.info(f"{tournament_type.value} tournament weights: {weights}")
    return weights


def get_tournament_weights_from_data(
    text_tournament_data: TournamentResultsWithWinners | None,
    image_tournament_data: TournamentResultsWithWinners | None,
    environment_tournament_data: TournamentResultsWithWinners | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Get tournament weights keeping text, image, and environment tournaments separate."""
    text_weights = _compute_weights(TournamentType.TEXT, text_tournament_data)
    image_weights = _compute_weights(TournamentType.IMAGE, image_tournament_data)
    environment_weights = _compute_weights(TournamentType.ENVIRONMENT, environment_tournament_data)
    return text_weights, image_weights, environment_weights
