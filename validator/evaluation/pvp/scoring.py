"""Back-compat shim: PvP scoring moved to core.pvp.scoring."""

from core.pvp.scoring import determine_outcome


__all__ = ["determine_outcome"]
