"""Back-compat shim: PvP agents moved to core.pvp.agents."""

from core.pvp.agents import BaseGameAgent
from core.pvp.agents import ClobberAgent
from core.pvp.agents import GinRummyAgent
from core.pvp.agents import GoofspielAgent
from core.pvp.agents import LeducPokerAgent
from core.pvp.agents import LiarsDiceAgent
from core.pvp.agents import OthelloAgent
from core.pvp.agents import load_prompts


__all__ = [
    "BaseGameAgent",
    "ClobberAgent",
    "GinRummyAgent",
    "GoofspielAgent",
    "LeducPokerAgent",
    "LiarsDiceAgent",
    "OthelloAgent",
    "load_prompts",
]
