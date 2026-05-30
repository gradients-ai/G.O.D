"""
Pydantic models for PvP (Player-vs-Player) environment evaluation.
Defines input configuration and output result contracts.
"""

from enum import Enum
from typing import Protocol

from pydantic import BaseModel
from pydantic import Field

from core.constants import EnvironmentName


class PvPIncompleteError(Exception):
    """Raised when PvP eval has incomplete pairs — completed pairs are persisted in DB."""
    pass


class PvPBaseModel(BaseModel):
    """Base for PvP models that have fields starting with 'model_'."""

    model_config = {"protected_namespaces": ()}


class GameOutcome(str, Enum):
    """Outcome of a single game from a player's perspective."""

    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


class GameScoringContext(BaseModel):
    """Extracted game metadata needed to compute win/loss/draw from returns."""

    returns: list[float] = Field(description="Terminal returns from state.returns(), one per player")
    player_id: int = Field(description="Index of the player whose outcome we're computing")
    is_zero_sum: bool = Field(description="Whether the game is zero-sum")
    min_utility: float = Field(description="Minimum possible return value")
    max_utility: float = Field(description="Maximum possible return value")


class GameInstance(PvPBaseModel):
    """Configuration for a single game to be played."""

    game_name: str = Field(description="OpenSpiel game identifier (e.g. 'liars_dice')")
    game_params: dict[str, int] = Field(description="Parameters passed to pyspiel.load_game()")
    model_a_player_id: int = Field(description="Player index assigned to model A (0 or 1)")
    seed: int = Field(description="Random seed for this game instance")
    is_zero_sum: bool = Field(description="Whether the game is zero-sum")
    min_utility: float = Field(description="Game's minimum utility value")
    max_utility: float = Field(description="Game's maximum utility value")


class PreparedModel(BaseModel):
    """Result of detecting model type and building SGLang flags."""

    sglang_model_path: str = Field(description="HF repo ID passed to SGLang --model-path")
    inference_name: str = Field(description="Model name used in chat completion requests")
    extra_sglang_args: str = Field(default="", description="Additional SGLang CLI flags (e.g. LoRA)")


class PvPModelSpec(PvPBaseModel):
    """Specification for a model participating in PvP evaluation."""

    repo: str = Field(description="HuggingFace model repository (e.g. 'org/model-name')")
    original_model: str = Field(
        description="Base model repository, used for LoRA detection"
    )
    gpu_id: int | None = Field(default=None, ge=0, description="GPU device ID. Defaults to 0 for model_a, 1 for model_b")
    port: int | None = Field(default=None, gt=0, description="SGLang server port. Defaults to 30000 for model_a, 30001 for model_b")


class PvPMatchupConfig(BaseModel):
    """Configuration for a single environment matchup."""

    num_games: int = Field(
        gt=0,
        description="Number of seeds to play. Each seed is played twice (position swap), so total games = num_games * 2",
    )


class PvPGroupModelSpec(BaseModel):
    """A model in a group evaluation. All must share the same base model."""

    repo: str = Field(description="HuggingFace model repository")
    hotkey: str = Field(description="Miner hotkey identifier")


class PvPMode(str, Enum):
    """Evaluation mode: single pair or round-robin group."""

    PAIR = "pair"
    GROUP = "group"


class PvPEvalConfig(PvPBaseModel):
    """Top-level input configuration for a PvP evaluation run.

    Loaded from PVP_EVAL_CONFIG env var or /config/pvp_eval.json.
    mode determines whether this is a pair or group evaluation.
    """

    mode: PvPMode = Field(default=PvPMode.PAIR)
    model_a: PvPModelSpec | None = Field(default=None, description="Pair mode: first model")
    model_b: PvPModelSpec | None = Field(default=None, description="Pair mode: second model")
    models: list[PvPGroupModelSpec] | None = Field(default=None, min_length=2, description="Group mode: models to compete")
    base_model: str | None = Field(default=None, description="Group mode: shared base model")
    gpu_ids: list[int] = Field(default=[0, 1], min_length=2, max_length=2)
    ports: list[int] = Field(default=[30000, 30001], min_length=2, max_length=2)
    matchups: dict[EnvironmentName, PvPMatchupConfig] = Field(
        description="Map of environment name to matchup configuration"
    )
    seed: int = Field(default=42, description="Base seed for deterministic game generation")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class PvPEnvironmentResult(PvPBaseModel):
    """Win/loss/draw result for a single environment."""

    model_a_wins: int = 0
    model_b_wins: int = 0
    draws: int = 0
    total_games: int = 0


class ChatRole(str, Enum):
    """OpenAI-compatible message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single message in an OpenAI-compatible conversation."""

    role: ChatRole
    content: str


class ChatCompletionConfig(BaseModel):
    """Configuration for calling an OpenAI-compatible chat endpoint."""

    inference_model: str = Field(description="Model name as registered in the inference server")
    base_url: str = Field(description="OpenAI-compatible API base (e.g. http://localhost:30000/v1)")
    api_key: str = Field(default="dummy", description="API key (SGLang ignores but SDK requires)")
    temperature: float | None = Field(default=None, description="Sampling temperature, None uses server default")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    max_tokens: int = Field(default=20, gt=0, description="Max tokens to generate per response")
    max_retries: int = Field(default=10, ge=0, description="Retry attempts on transient failures")
    read_timeout: float = Field(default=30.0, gt=0, description="HTTP read timeout in seconds")


class ChatResult(BaseModel):
    """Result from an LLM chat completion."""

    content: str | None = None
    usage: dict[str, int | None] | None = None


class ChatFn(Protocol):
    """Protocol for the chat completion callable, enabling DI for testing."""

    def __call__(self, config: ChatCompletionConfig, messages: list[ChatMessage]) -> ChatResult: ...


class PvPEvalMetadata(BaseModel):
    """Metadata about the evaluation run."""

    seed: int
    temperature: float
    position_swapped: bool = True
    wall_time_seconds: float = 0.0


class PvPEvalResults(PvPBaseModel):
    """Complete output of a PvP evaluation run (single pair)."""

    model_a: str
    model_b: str
    results: dict[EnvironmentName, PvPEnvironmentResult]
    metadata: PvPEvalMetadata


# --- Group evaluation models ---



def _canonical_pair_key(hotkey_a: str, hotkey_b: str) -> str:
    """Sorted pair key so order doesn't matter."""
    a, b = sorted([hotkey_a, hotkey_b])
    return f"{a}:{b}"


class PvPPairResult(PvPBaseModel):
    """Result for one pair within a group evaluation."""

    hotkey_a: str
    hotkey_b: str
    results: dict[EnvironmentName, PvPEnvironmentResult]

    @property
    def pair_key(self) -> str:
        return _canonical_pair_key(self.hotkey_a, self.hotkey_b)


class PvPPairDbRow(BaseModel):
    """A persisted PvP pair result row from the database."""

    task_id: str
    hotkey_a: str
    hotkey_b: str
    environment_name: str
    model_a_wins: int = 0
    model_b_wins: int = 0
    draws: int = 0
    total_games: int = 0
    n_attempts: int = 0
    status: str = "pending"

    @property
    def pair_key(self) -> str:
        return _canonical_pair_key(self.hotkey_a, self.hotkey_b)

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"


class FullWeightContestants(BaseModel):
    """Signal that some contestants submitted full weights instead of LoRA.

    These cannot participate in multi-LoRA group evaluation and need
    1v1 fallback in the tournament orchestrator.
    """

    hotkeys: list[str] = Field(description="Hotkeys of contestants with full-weight submissions")
    repos: list[str] = Field(description="Corresponding repo IDs")


class PvPGroupResults(PvPBaseModel):
    """Complete output of a group round-robin evaluation."""

    base_model: str
    hotkeys: list[str]
    pair_results: list[PvPPairResult]
    full_weight_fallbacks: FullWeightContestants | None = Field(
        default=None,
        description="Present if any contestants were excluded due to full-weight submissions",
    )
    metadata: PvPEvalMetadata
