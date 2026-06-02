"""Back-compat shim: the PvP bot moved to core.pvp.bot (shared with model-prep)."""

from core.pvp.bot import ContextOverflowError
from core.pvp.bot import EmptyLegalActionsError
from core.pvp.bot import InvalidActionForfeitError
from core.pvp.bot import LLMBot
from core.pvp.bot import TurnTimeoutError
from core.pvp.bot import default_memories


__all__ = [
    "ContextOverflowError",
    "EmptyLegalActionsError",
    "InvalidActionForfeitError",
    "LLMBot",
    "TurnTimeoutError",
    "default_memories",
]
