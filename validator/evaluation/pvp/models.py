"""Back-compat shim: PvP models moved to core.models.pvp_models."""

from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatFn
from core.models.pvp_models import ChatMessage
from core.models.pvp_models import ChatResult
from core.models.pvp_models import ChatRole
from core.models.pvp_models import ClobberParams
from core.models.pvp_models import FullWeightContestants
from core.models.pvp_models import FunctionSchema
from core.models.pvp_models import GameActionArgs
from core.models.pvp_models import GameInstance
from core.models.pvp_models import GameOutcome
from core.models.pvp_models import GameParams
from core.models.pvp_models import GameScoringContext
from core.models.pvp_models import GinRummyParams
from core.models.pvp_models import GoofspielParams
from core.models.pvp_models import JsonScalar
from core.models.pvp_models import LeducPokerParams
from core.models.pvp_models import LiarsDiceParams
from core.models.pvp_models import MemoryArea
from core.models.pvp_models import MemoryConfig
from core.models.pvp_models import MemoryOp
from core.models.pvp_models import MemorySlotEdit
from core.models.pvp_models import OthelloParams
from core.models.pvp_models import PreparedModel
from core.models.pvp_models import PvPBaseModel
from core.models.pvp_models import PvPEnvironmentResult
from core.models.pvp_models import PvPEvalConfig
from core.models.pvp_models import PvPEvalMetadata
from core.models.pvp_models import PvPEvalResults
from core.models.pvp_models import PvPGroupModelSpec
from core.models.pvp_models import PvPGroupResults
from core.models.pvp_models import PvPIncompleteError
from core.models.pvp_models import PvPIndividualScoreDbRow
from core.models.pvp_models import PvPMatchupConfig
from core.models.pvp_models import PvPMode
from core.models.pvp_models import PvPModelSpec
from core.models.pvp_models import PvPPairDbRow
from core.models.pvp_models import PvPPairResult
from core.models.pvp_models import PvPStatus
from core.models.pvp_models import ToolCall
from core.models.pvp_models import ToolSchema
from core.models.pvp_models import _canonical_pair_key


__all__ = [
    "ChatCompletionConfig",
    "ChatFn",
    "ChatMessage",
    "ChatResult",
    "ChatRole",
    "ClobberParams",
    "FullWeightContestants",
    "FunctionSchema",
    "GameActionArgs",
    "GameInstance",
    "GameOutcome",
    "GameParams",
    "GameScoringContext",
    "GinRummyParams",
    "GoofspielParams",
    "JsonScalar",
    "LeducPokerParams",
    "LiarsDiceParams",
    "MemoryArea",
    "MemoryConfig",
    "MemoryOp",
    "MemorySlotEdit",
    "OthelloParams",
    "PreparedModel",
    "PvPBaseModel",
    "PvPEvalConfig",
    "PvPEvalMetadata",
    "PvPEvalResults",
    "PvPEnvironmentResult",
    "PvPGroupModelSpec",
    "PvPGroupResults",
    "PvPIncompleteError",
    "PvPIndividualScoreDbRow",
    "PvPMatchupConfig",
    "PvPMode",
    "PvPModelSpec",
    "PvPPairDbRow",
    "PvPPairResult",
    "PvPStatus",
    "ToolCall",
    "ToolSchema",
    "_canonical_pair_key",
]
