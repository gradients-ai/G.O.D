"""Back-compat shim: the PvP bot moved to core.pvp.bot."""

from core.pvp.bot import ChatTimeoutForfeitError
from core.pvp.bot import ContextOverflowError
from core.pvp.bot import EmptyLegalActionsError
from core.pvp.bot import InvalidActionForfeitError
from core.pvp.bot import LLMBot
from core.pvp.bot import ModelUnreachableError
from core.pvp.bot import TurnTimeoutError
from core.pvp.bot import default_memories


__all__ = [
    "ChatTimeoutForfeitError",
    "ContextOverflowError",
    "EmptyLegalActionsError",
    "InvalidActionForfeitError",
    "LLMBot",
    "ModelUnreachableError",
    "TurnTimeoutError",
    "default_memories",
]
