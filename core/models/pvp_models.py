"""
Pydantic models for PvP (Player-vs-Player) environment evaluation.
Defines input configuration and output result contracts.
"""

import json
from enum import Enum
from typing import Annotated
from typing import Literal
from typing import Protocol

from pydantic import BaseModel
from pydantic import Field

from core.constants import EnvironmentName


class PvPIncompleteError(Exception):
    """Raised when PvP eval has incomplete pairs — completed pairs are persisted in DB."""
    pass


class PvPStatus(str, Enum):
    """Status of a persisted PvP result row."""

    PENDING = "pending"
    COMPLETE = "complete"


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


class GameParams(BaseModel):
    """Base for a game's pyspiel.load_game() parameters.

    Each game has its own subclass with just the fields it accepts; the `game`
    discriminator makes them a tagged union so a GameInstance round-trips to the
    right subclass. to_pyspiel() renders the kwargs dict, dropping the tag (which
    is ours, not a pyspiel parameter). Subclasses declare the `game` tag; the
    base omits it so each subclass owns its literal.
    """

    def to_pyspiel(self) -> dict[str, int | str | bool]:
        return self.model_dump(exclude={"game"})


class LiarsDiceParams(GameParams):
    game: Literal["liars_dice"] = "liars_dice"
    players: int = 2
    numdice: int = 5


class LeducPokerParams(GameParams):
    game: Literal["leduc_poker"] = "leduc_poker"
    players: int = 2


class GinRummyParams(GameParams):
    game: Literal["gin_rummy"] = "gin_rummy"
    hand_size: int
    knock_card: int


class OthelloParams(GameParams):
    game: Literal["othello"] = "othello"


class GoofspielParams(GameParams):
    game: Literal["goofspiel"] = "goofspiel"
    players: int = 2
    num_cards: int
    imp_info: bool = True
    points_order: Literal["random", "ascending", "descending"] = "random"
    returns_type: Literal["win_loss", "total_points", "point_difference"] = "win_loss"


class ClobberParams(GameParams):
    game: Literal["clobber"] = "clobber"
    rows: int
    columns: int


AnyGameParams = Annotated[
    LiarsDiceParams | LeducPokerParams | GinRummyParams | OthelloParams | GoofspielParams | ClobberParams,
    Field(discriminator="game"),
]


class GameInstance(PvPBaseModel):
    """Configuration for a single game to be played."""

    game_name: str = Field(description="OpenSpiel game identifier (e.g. 'liars_dice')")
    game_params: AnyGameParams = Field(description="Parameters passed to pyspiel.load_game()")
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
    tool_call_parser: str | None = Field(
        default=None,
        description=(
            "SGLang --tool-call-parser resolved by the caller, for repos whose id "
            "carries no family substring (opaque full-weight miner repos). When unset "
            "the server resolves from sglang_model_path."
        ),
    )


class PvPModelSpec(PvPBaseModel):
    """Specification for a model participating in PvP evaluation."""

    repo: str = Field(description="HuggingFace model repository (e.g. 'org/model-name')")
    original_model: str = Field(
        description="Foundation model repository (the root base), used for LoRA detection"
    )
    base_chain: list[str] = Field(
        default_factory=list,
        description=(
            "Adapter repos to merge onto `original_model` before applying `repo`, so a "
            "continuation miner is served on the base it actually trained on. Empty for "
            "round-1 models. A list (not a single repo) to support deeper chains."
        ),
    )
    gpu_id: int | None = Field(default=None, ge=0, description="GPU device ID. Defaults to 0 for model_a, 1 for model_b")
    port: int | None = Field(
        default=None,
        gt=0,
        description="SGLang server port. Defaults to 30000 for model_a, 30001 for model_b",
    )


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
    TOOL = "tool"


# A single JSON scalar — the value type of tool-call arguments.
JsonScalar = str | int | float | bool | None


class ToolCall(BaseModel):
    """A tool/function call parsed from a model response.

    arguments is the decoded JSON object the model passed to the tool.
    """

    id: str = Field(description="Provider-assigned id, echoed back in the matching tool result.")
    name: str = Field(description="Tool/function name.")
    arguments: dict[str, JsonScalar] = Field(default_factory=dict, description="Decoded JSON arguments.")

    def to_openai(self) -> dict:
        """Wire form for an assistant message's tool_calls (arguments JSON-encoded)."""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }


class ChatMessage(BaseModel):
    """A single message in an OpenAI-compatible conversation.

    Covers system/user text, assistant turns carrying tool_calls, and tool
    results (role=tool, with tool_call_id). to_openai() renders the wire form.
    """

    role: ChatRole
    content: str | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, description="Set on assistant turns that call tools.")
    tool_call_id: str | None = Field(default=None, description="Set on tool-result messages.")

    def to_openai(self) -> dict:
        out: dict = {"role": self.role.value, "content": self.content}
        if self.tool_calls is not None:
            out["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        return out


class ChatCompletionConfig(BaseModel):
    """Configuration for calling an OpenAI-compatible chat endpoint."""

    inference_model: str = Field(description="Model name as registered in the inference server")
    tokenizer_repo: str | None = Field(
        default=None,
        description=(
            "HF repo / local path for the tokenizer used to budget memory slots. "
            "A LoRA's inference_model carries a ':lora' suffix that is not a loadable "
            "repo, so this points at the base model (or local weights dir) instead. "
            "Falls back to inference_model when unset."
        ),
    )
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
    tool_calls: list[ToolCall] | None = None
    usage: dict[str, int | None] | None = None


class FunctionSchema(BaseModel):
    """One function-tool exposed to the model. parameters is a JSON Schema document."""

    name: str
    description: str
    parameters: dict


class ToolSchema(BaseModel):
    """OpenAI function-tool envelope."""

    type: Literal["function"] = "function"
    function: FunctionSchema

    def to_openai(self) -> dict:
        return self.model_dump()


class ChatFn(Protocol):
    """Protocol for the chat completion callable, enabling DI for testing."""

    def __call__(
        self,
        config: ChatCompletionConfig,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
    ) -> ChatResult: ...


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
    deployment_id: str | None = None
    status: PvPStatus = PvPStatus.PENDING

    @property
    def pair_key(self) -> str:
        return _canonical_pair_key(self.hotkey_a, self.hotkey_b)

    @property
    def is_complete(self) -> bool:
        return self.status == PvPStatus.COMPLETE


class PvPIndividualScoreDbRow(BaseModel):
    """A persisted individual score row from the database."""

    task_id: str
    hotkey: str
    environment_name: str
    score: float = 0.0
    n_attempts: int = 0
    status: PvPStatus = PvPStatus.PENDING

    @property
    def is_complete(self) -> bool:
        return self.status == PvPStatus.COMPLETE


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


# --- Tool-calling memory models ---


class MemoryArea(str, Enum):
    """An area of model-managed memory in the tool-calling harness.

    Add a member here (plus its SlotMemory instance and presentation metadata)
    to introduce a new memory area; the tool layer expands automatically.
    """

    WORKING = "working_memory"
    LONG_TERM = "long_term_memory"

    @property
    def persists_across_games(self) -> bool:
        """Long-term memory survives between games (opponent model); working resets."""
        return self is MemoryArea.LONG_TERM


class MemoryOp(str, Enum):
    """An edit operation on a memory slot.

    The value equals the SlotMemory method name, so dispatch needs no lookup
    table: getattr(slot_memory, op.value)(...).
    """

    REWRITE = "rewrite"
    APPEND = "append"


class MemoryConfig(BaseModel):
    """Sizing for one memory area: a fixed number of fixed-size slots."""

    n_slots: int = Field(gt=0, description="Number of addressable slots.")
    slot_token_budget: int = Field(gt=0, description="Max tokens retained per slot.")


class MemorySlotEdit(BaseModel):
    """Arguments accepted by a memory edit tool (rewrite/append)."""

    slot: int = Field(description="Target slot number.")
    content: str = Field(description="Text content for the slot.")


class GameActionArgs(BaseModel):
    """Arguments accepted by the game_action tool."""

    action_id: int = Field(description="A legal action id for the current state.")
