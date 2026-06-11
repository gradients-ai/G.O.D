"""Back-compat shim: PvP chat client moved to core.pvp.chat (shared with model-prep)."""

from core.pvp.chat import chat_completion
from core.pvp.chat import create_client
from core.pvp.chat import strip_think_tags


__all__ = ["chat_completion", "create_client", "strip_think_tags"]
