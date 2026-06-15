"""Game-specific agents for PvP evaluation.

Each agent provides state formatting and parameter generation for its game.
Rules text is loaded from core/config/pvp_game_prompts.yml.
"""

import functools
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path

import pyspiel
import yaml

from core.models.pvp_models import GameParams
from core.models.pvp_models import GinRummyParams
from core.models.pvp_models import GoofspielParams
from core.models.pvp_models import LeducPokerParams
from core.models.pvp_models import LiarsDiceParams
from core.models.pvp_models import OthelloParams

_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "core" / "config" / "pvp_game_prompts.yml"


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
    def generate_params(self, config_id: int) -> GameParams:
        """Generate pyspiel game parameters from a config variant ID."""
        ...

    def setup_initial_state(self, state: pyspiel.State, seed: int) -> None:
        """Advance the fresh state before the models take over. Default: no-op.

        Games with chance nodes (dice, card deals) get their per-game variety for
        free from the seed passed to evaluate_bots. Deterministic games with no
        chance nodes (e.g. othello) override this to inject seeded variety so the
        same seed reproduces the same start while different seeds diverge.
        """
        return None

    def load_game(self, params: GameParams) -> pyspiel.Game:
        """Build the pyspiel game this agent plays.

        Default: load game_name with params directly. Games whose native dynamics
        are simultaneous (e.g. goofspiel) override this to wrap the game so the
        sequential LLMBot/evaluate_bots harness can drive it.
        """
        return pyspiel.load_game(self.game_name, params.to_pyspiel())

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


# --- Concrete agents ---


class LiarsDiceAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "liars_dice"

    @property
    def rules_key(self) -> str:
        return "liars_dice_rules"

    def generate_params(self, config_id: int) -> GameParams:
        return LiarsDiceParams(players=2, numdice=5)

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

    def generate_params(self, config_id: int) -> GameParams:
        return LeducPokerParams(players=2)

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

    def generate_params(self, config_id: int) -> GameParams:
        hand_var = (config_id // 3) % 3
        knock_var = config_id % 3
        return GinRummyParams(hand_size=7 + hand_var, knock_card=10 - knock_var)

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        return state.observation_string(player_id)


# Number of seeded random opening plies applied to an othello game, sampled from
# this inclusive range. Enough to diverge the opening tree for variety, few
# enough that positions stay balanced and game-like.
_OTHELLO_OPENING_PLIES = (2, 6)


class OthelloAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "othello"

    @property
    def rules_key(self) -> str:
        return "othello_rules"

    def generate_params(self, config_id: int) -> GameParams:
        return OthelloParams()

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        """Prefix the board with the player's colour.

        The observation only says whose turn it is ("Black (x) to play"), so
        without this line the model must infer its own colour — small models
        get it wrong and play for the opponent.
        """
        colour = "x (Black)" if player_id == 0 else "o (White)"
        return f"You play {colour}.\n{state.observation_string(player_id)}"

    def setup_initial_state(self, state: pyspiel.State, seed: int) -> None:
        """Apply a seeded number of uniformly-random legal opening moves.

        Othello is deterministic with no chance nodes, so every game would start
        from the identical board. Deriving the opening plies from the instance
        seed keeps games reproducible (same seed -> same start) while giving each
        seed a distinct mid-game position to play from.
        """
        rng = random.Random(seed)
        num_plies = rng.randint(*_OTHELLO_OPENING_PLIES)
        for _ in range(num_plies):
            if state.is_terminal():
                break
            legal_actions = state.legal_actions()
            if not legal_actions:
                break
            state.apply_action(rng.choice(legal_actions))


# Deck sizes goofspiel is played with, selected per game from the config id so
# each game varies board size (and thus length) for SFT/eval diversity. 5 is a
# short sharp game; 13 is the full standard deck.
_GOOFSPIEL_NUM_CARDS = (5, 8, 10, 13)


class GoofspielAgent(BaseGameAgent):
    """Goofspiel (a.k.a. the Game of Pure Strategy).

    OpenSpiel's goofspiel is a SIMULTANEOUS-move game; the sequential harness
    drives it via convert_to_turn_based, which hides each player's concurrent
    bid from the other (so simultaneity and fairness are preserved). Played with
    imp_info=True (opponent hand hidden) and returns_type=win_loss so terminal
    returns are zero-sum {-1, 0, 1}, mapping straight to win/loss/draw.
    """

    @property
    def game_name(self) -> str:
        return "goofspiel"

    @property
    def rules_key(self) -> str:
        return "goofspiel_rules"

    def generate_params(self, config_id: int) -> GameParams:
        num_cards = _GOOFSPIEL_NUM_CARDS[config_id % len(_GOOFSPIEL_NUM_CARDS)]
        return GoofspielParams(
            players=2,
            num_cards=num_cards,
            imp_info=True,
            points_order="random",
            returns_type="win_loss",
        )

    def load_game(self, params: GameParams) -> pyspiel.Game:
        """Load goofspiel and wrap its simultaneous moves into sequential turns."""
        return pyspiel.convert_to_turn_based(pyspiel.load_game(self.game_name, params.to_pyspiel()))

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        """Render the player's own view; imp_info keeps the opponent's hand hidden.

        observation_string already reports the current point card, the remaining
        point cards, both running scores, this player's hand and the win sequence.
        Prefix the player's identity since the board labels are absolute (P0/P1).
        """
        return f"You are Player {player_id} (P{player_id}).\n{state.observation_string(player_id)}"
