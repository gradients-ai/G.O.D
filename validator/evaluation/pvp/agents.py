"""Back-compat shim: PvP agents moved to core.pvp.agents (shared with model-prep)."""

from core.pvp.agents import BaseGameAgent
from core.pvp.agents import GinRummyAgent
from core.pvp.agents import LeducPokerAgent
from core.pvp.agents import LiarsDiceAgent
from core.pvp.agents import OthelloAgent
from core.pvp.agents import load_prompts


__all__ = ["BaseGameAgent", "GinRummyAgent", "LeducPokerAgent", "LiarsDiceAgent", "OthelloAgent", "load_prompts"]
