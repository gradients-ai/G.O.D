"""Scoring models for tournament evaluation."""

from pydantic import BaseModel
from pydantic import Field

from core.constants import EnvironmentName


class TournamentScore(BaseModel):
    hotkey: str
    score: float


class EnvironmentWeight(BaseModel):
    """Weight for a single environment in tournament scoring."""

    environment: EnvironmentName
    weight: float = Field(default=1.0, ge=0.0, description="Scoring multiplier for this environment")


class PairwiseOutcome(BaseModel):
    """Universal outcome of a single pair comparison on a single environment.

    Produced by any eval type (PvP, MCTS, etc.) and fed into the universal
    points accumulator. The winner field is the hotkey of the winner, or
    None for a draw.
    """

    hotkey_a: str
    hotkey_b: str
    environment: EnvironmentName
    winner: str | None = Field(description="Hotkey of winner, or None for draw")


class GroupStagePoints(BaseModel):
    """Per-hotkey points from group stage evaluation (any eval type)."""

    hotkey: str
    points: float


class TournamentTypeResult(BaseModel):
    scores: list[TournamentScore]
    prev_winner_hotkey: str | None
    prev_winner_won_final: bool


class EvalHotkeyResults(BaseModel):
    """Outcome of evaluating a batch of hotkeys."""

    evaluated: list[str] = Field(description="Hotkeys that were successfully evaluated")
    failed: list[str] = Field(default_factory=list, description="Hotkeys that failed evaluation")
