"""Score computation for PvP game results.

Determines the winner of a 2-player game from OpenSpiel's terminal returns.
"""

from core.models.pvp_models import GameOutcome, GameScoringContext


def determine_outcome(context: GameScoringContext) -> GameOutcome:
    """Determine win/loss/draw for the given player from terminal returns.

    For zero-sum games, normalizes the player's return to [0, 1] and
    uses 0.5 as the draw threshold. For general-sum, compares raw returns.
    """
    player_return = context.returns[context.player_id]
    opponent_return = context.returns[1 - context.player_id]

    if context.is_zero_sum:
        if context.max_utility > context.min_utility:
            score = (player_return - context.min_utility) / (context.max_utility - context.min_utility)
        else:
            score = 0.5

        if score > 0.5:
            return GameOutcome.WIN
        elif score < 0.5:
            return GameOutcome.LOSS
        return GameOutcome.DRAW

    if player_return > opponent_return:
        return GameOutcome.WIN
    elif player_return < opponent_return:
        return GameOutcome.LOSS
    return GameOutcome.DRAW
