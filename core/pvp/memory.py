"""Fixed-slot working / long-term memory for the tool-calling PvP harness.

Each memory area is a fixed number of fixed-size slots that the model edits
via tools (see tools.py). Slots are 1-indexed.

Design invariants:
  * Bounded by construction: total size <= n_slots * slot_token_budget, so the
    per-turn context can never grow over a game or a matchup.
  * Total operations: a bad slot index or malformed input returns an error
    string and NEVER raises. A fumbled memory op must not crash a game or cost
    a match — only game_action can forfeit.
  * rewrite truncates the TAIL (keep what was written first — "write tighter").
  * append truncates the HEAD (FIFO within the slot — drop the oldest text).

Token budgets are enforced through an injected TokenCounter so this module
stays dependency-free and deterministic in tests; production injects a
tokenizer-backed counter so budgets reflect real model tokens.
"""

from __future__ import annotations

from typing import Literal
from typing import Protocol


class TokenCounter(Protocol):
    """Measures and truncates text in token units."""

    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int, keep: Literal["head", "tail"]) -> str: ...


class WhitespaceTokenCounter:
    """Deterministic, dependency-free counter: one token per whitespace word.

    Used as the default and in tests. Production wraps the served model's
    tokenizer instead so slot budgets map onto real tokens.
    """

    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int, keep: Literal["head", "tail"]) -> str:
        words = text.split()
        if len(words) <= max_tokens:
            return text
        kept = words[:max_tokens] if keep == "head" else words[len(words) - max_tokens:]
        return " ".join(kept)


class SlotMemory:
    """A fixed set of fixed-size, independently addressable memory slots."""

    def __init__(
        self,
        n_slots: int,
        slot_token_budget: int,
        counter: TokenCounter,
        separator: str = "\n",
    ):
        if n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        if slot_token_budget < 1:
            raise ValueError("slot_token_budget must be >= 1")
        self.n_slots = n_slots
        self.slot_token_budget = slot_token_budget
        self._counter = counter
        self._sep = separator
        self.slots: dict[int, str] = {i: "" for i in range(1, n_slots + 1)}

    def _valid(self, slot: int) -> bool:
        return isinstance(slot, int) and not isinstance(slot, bool) and 1 <= slot <= self.n_slots

    def _fit(self, text: str, keep: Literal["head", "tail"]) -> str:
        if self._counter.count(text) <= self.slot_token_budget:
            return text
        return self._counter.truncate(text, self.slot_token_budget, keep)

    def rewrite(self, slot: int, content: str) -> str:
        """Overwrite a slot. Over budget -> drop the tail (keep the front)."""
        if not self._valid(slot):
            return f"error: slot {slot} out of range (1-{self.n_slots})"
        stored = self._fit(content, keep="head")
        self.slots[slot] = stored
        note = " (truncated to budget)" if stored != content else ""
        return f"ok: slot {slot} rewritten, {self._counter.count(stored)} tokens{note}"

    def append(self, slot: int, content: str) -> str:
        """Append to a slot. Over budget -> drop the oldest text (FIFO front)."""
        if not self._valid(slot):
            return f"error: slot {slot} out of range (1-{self.n_slots})"
        existing = self.slots[slot]
        combined = f"{existing}{self._sep}{content}" if existing else content
        stored = self._fit(combined, keep="tail")
        self.slots[slot] = stored
        note = " (oldest dropped)" if stored != combined else ""
        return f"ok: slot {slot} appended, {self._counter.count(stored)} tokens{note}"

    def clear(self, slot: int) -> str:
        if not self._valid(slot):
            return f"error: slot {slot} out of range (1-{self.n_slots})"
        self.slots[slot] = ""
        return f"ok: slot {slot} cleared"

    def read(self, slot: int) -> str:
        if not self._valid(slot):
            return f"error: slot {slot} out of range (1-{self.n_slots})"
        return self.slots[slot] or "(empty)"

    def render(self, title: str | None = None) -> str:
        """Render every slot as a numbered list (empties included), for the prompt."""
        lines = [f"  [{i}] {self.slots[i] or '(empty)'}" for i in range(1, self.n_slots + 1)]
        body = "\n".join(lines)
        return f"{title}\n{body}" if title else body

    def reset(self) -> None:
        """Empty every slot (used when a per-game memory area starts a new game)."""
        self.slots = {i: "" for i in range(1, self.n_slots + 1)}

    def to_dict(self) -> dict[int, str]:
        return dict(self.slots)
