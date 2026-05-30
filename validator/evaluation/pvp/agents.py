"""Game-specific agents for PvP evaluation.

Each agent provides state formatting and parameter generation for its game.
Rules text is loaded from core/config/pvp_game_prompts.yml.
"""

import functools
import re
from abc import ABC, abstractmethod
from pathlib import Path

import pyspiel
import yaml

_PROMPTS_PATH = Path(__file__).resolve().parents[3] / "core" / "config" / "pvp_game_prompts.yml"


@functools.cache
def load_prompts() -> dict[str, str]:
    with open(_PROMPTS_PATH) as f:
        return yaml.safe_load(f)


class BaseGameAgent(ABC):
    """Abstract base for game-specific LLM prompt generation."""

    @property
    @abstractmethod
    def game_name(self) -> str:
        ...

    @property
    @abstractmethod
    def rules_key(self) -> str:
        """Key in pvp_game_prompts.yml for this game's rules."""
        ...

    @abstractmethod
    def generate_params(self, config_id: int) -> dict[str, int]:
        """Generate pyspiel game parameters from a config variant ID."""
        ...

    def get_rules(self) -> str:
        return load_prompts()[self.rules_key]

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        """Format game state as text. Override for game-specific formatting."""
        try:
            return state.observation_string(player_id)
        except (RuntimeError, AttributeError):
            pass
        try:
            return state.information_state_string(player_id)
        except (RuntimeError, AttributeError):
            raise ValueError(
                f"Game {self.game_name} supports neither observation_string nor "
                f"information_state_string — override format_state() for this game"
            )

    def generate_system_prompt(self) -> str:
        prompts = load_prompts()
        return prompts["system_prompt_template"].format(
            game_name=self.game_name, rules=self.get_rules()
        )

    def generate_user_prompt(
        self,
        state: pyspiel.State,
        player_id: int,
        legal_actions: list[int],
    ) -> str:
        state_desc = self.format_state(state, player_id)
        actions_desc = []
        for action in legal_actions:
            try:
                action_str = state.action_to_string(player_id, action)
                actions_desc.append(f"{action} -> {action_str}")
            except (RuntimeError, AttributeError):
                actions_desc.append(str(action))

        return (
            f"Current State:\n{state_desc}\n\n"
            f"You are Player {player_id}.\n"
            f"Legal Actions:\n" + "\n".join(actions_desc) + "\n\n"
            "Your choice (ID only):"
        )


# --- Concrete agents ---


class LiarsDiceAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "liars_dice"

    @property
    def rules_key(self) -> str:
        return "liars_dice_rules"

    def generate_params(self, config_id: int) -> dict[str, int]:
        return {"players": 2, "numdice": 5}

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        try:
            info_str = state.information_state_string(player_id)
        except (RuntimeError, AttributeError):
            return str(state)

        if not info_str:
            return str(state)

        parts = info_str.split()
        dice_part = parts[0]
        bid_parts = [p for p in parts[1:] if "-" in p]

        dice = [int(d) for d in dice_part if d.isdigit()]
        num_dice = len(dice)
        total_dice = num_dice * state.num_players()

        lines = [
            f"Your dice: {dice} (showing: {', '.join(map(str, dice))})",
            f"Dice per player: {num_dice}",
            f"Total dice in game: {total_dice}",
            f"Players: {state.num_players()}",
            f"Current player: Player {state.current_player()}",
        ]

        if bid_parts:
            last_bid = bid_parts[-1]
            quantity, face = last_bid.split("-")
            lines.append(
                f'\nCurrent bid: "{quantity}-{face}" '
                f"(at least {quantity} dice showing {face} across all players)"
            )
            lines.append("You can: (1) Make a higher bid, or (2) Call 'Liar'")
        else:
            lines.append("No bid yet - you must make the first bid")

        return "\n".join(lines)


class LeducPokerAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "leduc_poker"

    @property
    def rules_key(self) -> str:
        return "leduc_poker_rules"

    def generate_params(self, config_id: int) -> dict[str, int]:
        return {"players": 2}

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        try:
            info_str = state.information_state_string(player_id)
        except (RuntimeError, AttributeError):
            return str(state)

        private_card = self._extract(info_str, r"\[Private: (-?\d+)\]")
        round_num = self._extract(info_str, r"\[Round (\d+)\]")
        pot = self._extract(info_str, r"\[Pot: (\d+)\]")
        money = self._extract(info_str, r"\[Money: ([\d ]+)\]")
        public_card = self._extract(info_str, r"\[Public: (-?\d+)\]")
        round1_seq = self._extract(info_str, r"\[Round1: ([^\]]*)\]")
        round2_seq = self._extract(info_str, r"\[Round2: ([^\]]*)\]")

        lines: list[str] = []

        if private_card and private_card != "-10000":
            lines.append(f"Your card: {self._card_name(int(private_card))}")
        else:
            lines.append("Your card: (not dealt yet)")

        if public_card and public_card != "-10000":
            lines.append(f"Public card: {self._card_name(int(public_card))}")
            if private_card and private_card != "-10000":
                if int(private_card) // 2 == int(public_card) // 2:
                    lines.append("Hand: PAIR")

        lines.append(f"Round: {round_num}/2")
        lines.append(f"Pot: {pot} chips")

        if money:
            chips = money.split()
            if len(chips) >= 2:
                lines.append(f"Your chips: {chips[player_id]}")
                lines.append(f"Opponent chips: {chips[1 - player_id]}")

        if round1_seq:
            lines.append(f"Round 1 actions: {self._parse_betting(round1_seq)}")
        if round2_seq:
            lines.append(f"Round 2 actions: {self._parse_betting(round2_seq)}")

        return "\n".join(lines)

    @staticmethod
    def _extract(info_str: str, pattern: str) -> str:
        match = re.search(pattern, info_str)
        return match.group(1) if match else ""

    @staticmethod
    def _card_name(card_id: int) -> str:
        ranks = ["J", "Q", "K", "A"]  # A used only in 3+ player variants
        suits = ["\u2660", "\u2665"]
        rank_idx = card_id // 2
        suit_idx = card_id % 2
        if rank_idx < len(ranks):
            return f"{ranks[rank_idx]}{suits[suit_idx]}"
        return f"Card_{card_id}"

    @staticmethod
    def _parse_betting(seq: str) -> str:
        if not seq or not seq.strip():
            return "(none)"
        actions_map = {0: "Fold", 1: "Call", 2: "Raise"}
        numbers = [int(x) for x in seq.split() if x.isdigit()]
        if not numbers:
            return "(none)"
        return ", ".join(actions_map.get(a, f"Action{a}") for a in numbers)


class GinRummyAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "gin_rummy"

    @property
    def rules_key(self) -> str:
        return "gin_rummy_rules"

    def generate_params(self, config_id: int) -> dict[str, int]:
        hand_var = (config_id // 3) % 3
        knock_var = config_id % 3
        return {
            "hand_size": 7 + hand_var,
            "knock_card": 10 - knock_var,
        }

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        return state.observation_string(player_id)
