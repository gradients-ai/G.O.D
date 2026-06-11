"""Back-compat shim: PvP scoring moved to core.pvp.scoring (shared with model-prep)."""

from core.pvp.scoring import determine_outcome


__all__ = ["determine_outcome"]
