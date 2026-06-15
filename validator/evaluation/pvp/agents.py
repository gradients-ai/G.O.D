"""Back-compat shim: PvP agents moved to core.pvp.agents (shared with model-prep)."""

from core.pvp.agents import BaseGameAgent as BaseGameAgent
from core.pvp.agents import GinRummyAgent as GinRummyAgent
from core.pvp.agents import GoofspielAgent as GoofspielAgent
from core.pvp.agents import LeducPokerAgent as LeducPokerAgent
from core.pvp.agents import LiarsDiceAgent as LiarsDiceAgent
from core.pvp.agents import OthelloAgent as OthelloAgent
from core.pvp.agents import load_prompts as load_prompts
