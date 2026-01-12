
import validator.core.constants as cts
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentScore
from core.models.tournament_models import TournamentType
from core.models.tournament_models import TournamentTypeResult
from validator.tournament.utils import get_real_winner_hotkey
from validator.utils.logging import get_logger


logger = get_logger(__name__)


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

    # Get real winner hotkey (handles EMISSION_BURN_HOTKEY placeholder for defending champions)
    actual_winner_hotkey = get_real_winner_hotkey(tournament_data.winner_hotkey, tournament_data.base_winner_hotkey)
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
                    quality_score = participant.get("quality_score")
                    if hotkey == actual_winner_hotkey:
                        continue
                    if hotkey == cts.EMISSION_BURN_HOTKEY and tournament_data.base_winner_hotkey:
                        continue
                    ranked_participants.append((hotkey, quality_score))

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


def get_tournament_weights_from_data(
    text_tournament_data: TournamentResultsWithWinners | None,
    image_tournament_data: TournamentResultsWithWinners | None,
    environment_tournament_data: TournamentResultsWithWinners | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Get tournament weights keeping text, image, and environment tournaments separate."""

    # Calculate text tournament weights
    text_result = calculate_tournament_type_scores_from_data(TournamentType.TEXT, text_tournament_data)
    text_weights = {}
    if text_result.scores:
        text_weights = tournament_scores_to_weights(
            text_result.scores, text_result.prev_winner_hotkey, text_result.prev_winner_won_final
        )
    logger.info(f"Text tournament weights: {text_weights}")

    # Calculate image tournament weights
    image_result = calculate_tournament_type_scores_from_data(TournamentType.IMAGE, image_tournament_data)
    image_weights = {}
    if image_result.scores:
        image_weights = tournament_scores_to_weights(
            image_result.scores, image_result.prev_winner_hotkey, image_result.prev_winner_won_final
        )
    logger.info(f"Image tournament weights: {image_weights}")

    # Calculate environment tournament weights
    environment_result = calculate_tournament_type_scores_from_data(TournamentType.ENVIRONMENT, environment_tournament_data)
    environment_weights = {}
    if environment_result.scores:
        environment_weights = tournament_scores_to_weights(
            environment_result.scores, environment_result.prev_winner_hotkey, environment_result.prev_winner_won_final
        )
    logger.info(f"Environment tournament weights: {environment_weights}")

    return text_weights, image_weights, environment_weights
