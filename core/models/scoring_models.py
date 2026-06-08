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


class MinerRepos(BaseModel):
    """Miner hotkey → HuggingFace model repo mapping for tournament evaluation."""

    by_hotkey: dict[str, str] = Field(description="Mapping of hotkey → repo_id")

    @property
    def hotkeys(self) -> list[str]:
        return list(self.by_hotkey.keys())

    @property
    def repos(self) -> list[str]:
        return list(self.by_hotkey.values())

    def __len__(self) -> int:
        return len(self.by_hotkey)

    def subset(self, hotkeys: list[str]) -> "MinerRepos":
        """Return a new MinerRepos containing only the given hotkeys."""
        return MinerRepos(by_hotkey={hk: self.by_hotkey[hk] for hk in hotkeys if hk in self.by_hotkey})


class IndividualEvalResult(BaseModel):
    """Scores from individual eval containers for one environment."""

    environment_name: EnvironmentName
    scores_by_hotkey: dict[str, float]


class IndividualScoresByEnv(BaseModel):
    """Collected individual scores across multiple environments."""

    results: dict[EnvironmentName, IndividualEvalResult] = Field(default_factory=dict)

    def is_complete(self, envs: list[EnvironmentName], hotkeys: list[str]) -> bool:
        for env in envs:
            result = self.results.get(env)
            if not result or any(hk not in result.scores_by_hotkey for hk in hotkeys):
                return False
        return True

    def missing(self, envs: list[EnvironmentName], hotkeys: list[str]) -> list[tuple[EnvironmentName, list[str]]]:
        incomplete = []
        for env in envs:
            result = self.results.get(env)
            missing_hks = [hk for hk in hotkeys if hk not in (result.scores_by_hotkey if result else {})]
            if missing_hks:
                incomplete.append((env, missing_hks))
        return incomplete


class EnvMinerScores(BaseModel):
    """Continuous per-miner scores for a single environment, before rank normalization.

    Holds either a PvP env's win-rates or an INDIVIDUAL env's raw scores — both are
    just a higher-is-better number per hotkey, combined downstream by rank.
    """

    environment: EnvironmentName
    scores_by_hotkey: dict[str, float]


class EvalHotkeyResults(BaseModel):
    """Outcome of evaluating a batch of hotkeys."""

    evaluated: list[str] = Field(description="Hotkeys that were successfully evaluated")
    failed: list[str] = Field(default_factory=list, description="Hotkeys that failed evaluation")
