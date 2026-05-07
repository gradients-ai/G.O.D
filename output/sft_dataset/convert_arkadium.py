"""
Convert Arkadium PBN gin rummy game logs to SFT training format.

Reads .pbn files, reconstructs game state at each step, validates
every conversion, and outputs JSONL matching checkpoint.jsonl format.

Validation: any step that fails a consistency check causes the entire
game to be skipped — only fully-validated games make it to output.
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Card encoding — must match the game server exactly
# ---------------------------------------------------------------------------
SUITS_PBN_TO_ENV = {"S": "s", "H": "h", "D": "d", "C": "c"}
SUIT_ORDER = ["s", "c", "d", "h"]  # spades=0, clubs=1, diamonds=2, hearts=3
RANK_ORDER = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
RANK_VALUES = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
               "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13}

# Grid layout positions for the box-drawing display
GRID_COLS = {
    "A": 0, "2": 2, "3": 4, "4": 6, "5": 8, "6": 10,
    "7": 12, "8": 14, "9": 16, "T": 18, "J": 20, "Q": 22, "K": 24,
}
GRID_WIDTH = 26  # inner width of the card display box


def pbn_card_to_env(pbn_card: str) -> str:
    """Convert PBN notation (e.g. 'HK') to env notation (e.g. 'Kh')."""
    suit_char = pbn_card[0]
    rank_char = pbn_card[1:]
    if rank_char == "10":
        rank_char = "T"
    return rank_char + SUITS_PBN_TO_ENV[suit_char]


def card_to_action_id(env_card: str) -> int:
    """Convert env card notation (e.g. 'Kh') to action ID (0-51)."""
    rank = env_card[:-1]
    suit = env_card[-1]
    suit_idx = SUIT_ORDER.index(suit)
    rank_idx = RANK_ORDER.index(rank)
    return suit_idx * 13 + rank_idx


def card_deadwood_value(env_card: str) -> int:
    """Deadwood point value for a card."""
    rank = env_card[:-1]
    return RANK_VALUES[rank]


# ---------------------------------------------------------------------------
# Meld detection and optimal deadwood (DP)
# ---------------------------------------------------------------------------

def find_all_melds(hand: list[str]) -> list[list[str]]:
    """Find all valid melds (sets and runs) from a hand."""
    melds = []

    # Sets: 3+ cards of same rank
    by_rank: dict[str, list[str]] = {}
    for c in hand:
        r = c[:-1]
        by_rank.setdefault(r, []).append(c)
    for r, cards in by_rank.items():
        if len(cards) >= 3:
            from itertools import combinations
            for size in range(3, len(cards) + 1):
                for combo in combinations(cards, size):
                    melds.append(list(combo))

    # Runs: 3+ consecutive same-suit cards
    by_suit: dict[str, list[str]] = {}
    for c in hand:
        s = c[-1]
        by_suit.setdefault(s, []).append(c)
    for s, cards in by_suit.items():
        indices = sorted(RANK_ORDER.index(c[:-1]) for c in cards)
        card_by_idx = {RANK_ORDER.index(c[:-1]): c for c in cards}
        for start in range(len(indices)):
            run = [indices[start]]
            for j in range(start + 1, len(indices)):
                if indices[j] == run[-1] + 1:
                    run.append(indices[j])
                else:
                    break
            if len(run) >= 3:
                for end in range(2, len(run)):
                    for begin in range(len(run) - end):
                        sub = run[begin:begin + end + 1]
                        if len(sub) >= 3:
                            melds.append([card_by_idx[i] for i in sub])

    # Deduplicate
    seen = set()
    unique = []
    for m in melds:
        key = tuple(sorted(m))
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def compute_optimal_deadwood(hand: list[str]) -> int:
    """Compute minimum deadwood using bitmask DP over all valid melds."""
    if not hand:
        return 0

    n = len(hand)
    melds = find_all_melds(hand)

    # Convert melds to bitmasks
    meld_masks = []
    for meld in melds:
        mask = 0
        for card in meld:
            idx = hand.index(card)
            mask |= (1 << idx)
        meld_masks.append(mask)

    # DP: for each subset of used cards, find minimum deadwood
    total_dw = sum(card_deadwood_value(c) for c in hand)
    card_values = [card_deadwood_value(c) for c in hand]

    best = total_dw  # worst case: no melds
    # Try all combinations of non-overlapping melds
    def search(idx: int, used: int, melded_value: int):
        nonlocal best
        best = min(best, total_dw - melded_value)
        for i in range(idx, len(meld_masks)):
            if meld_masks[i] & used == 0:
                mv = sum(card_values[b] for b in range(n) if meld_masks[i] & (1 << b))
                search(i + 1, used | meld_masks[i], melded_value + mv)

    search(0, 0, 0)
    return best


# ---------------------------------------------------------------------------
# Card display box (matches server format)
# ---------------------------------------------------------------------------

def render_card_box(hand: list[str], deadwood: int | None = None, player_id: int = 0, hidden: bool = False) -> str:
    """Render cards in the box-drawing format used by the game server.

    The box is exactly 28 chars wide:  +  26 dashes  +
    Content rows:                      |  26 chars   |
    """
    border = "+" + "-" * GRID_WIDTH + "+"

    if hidden:
        lines = [border]
        for _ in range(4):
            lines.append("|" + " " * GRID_WIDTH + "|")
        lines.append(border)
        header = f"Player{player_id}:"
        return header + "\n" + "\n".join(lines)

    # Group cards by suit row (s=0, c=1, d=2, h=3 — matches SUIT_ORDER)
    rows: list[list[str]] = [[] for _ in range(4)]
    for card in hand:
        suit = card[-1]
        suit_idx = SUIT_ORDER.index(suit)
        rows[suit_idx].append(card)

    lines = [border]
    for row_cards in rows:
        row_chars = [" "] * GRID_WIDTH
        for card in sorted(row_cards, key=lambda c: RANK_ORDER.index(c[:-1])):
            rank = card[:-1]
            col = GRID_COLS[rank]
            row_chars[col] = rank
            if col + 1 < GRID_WIDTH:
                row_chars[col + 1] = card[-1]
        line = "".join(row_chars)
        lines.append("|" + line + "|")
    lines.append(border)

    if deadwood is not None:
        header = f"Player{player_id}: Deadwood={deadwood}"
    else:
        header = f"Player{player_id}:"
    return header + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# PBN parser
# ---------------------------------------------------------------------------

@dataclass
class PBNGame:
    """Parsed PBN game log."""
    gameplay_id: str = ""
    hand_number: str = ""
    game_score: str = ""
    dealer: int = 0
    player1_hand: list[str] = field(default_factory=list)
    player2_hand: list[str] = field(default_factory=list)
    upcard: str = ""
    score: str = ""
    moves: list[dict] = field(default_factory=list)
    groups: dict[int, list[list[str]]] = field(default_factory=dict)
    layoffs: dict[int, list[str]] = field(default_factory=dict)
    deadwood: str = ""
    knock_card: int = 10  # Default


def parse_pbn(text: str) -> PBNGame:
    """Parse a PBN game log into structured data."""
    game = PBNGame()

    # Parse header fields
    for m in re.finditer(r'\[(\w+)\s+"([^"]*)"\]', text):
        key, val = m.group(1), m.group(2)
        if key == "GameplayId":
            game.gameplay_id = val
        elif key == "Hand":
            game.hand_number = val
        elif key == "GameScore":
            game.game_score = val
        elif key == "Dealer":
            game.dealer = int(val)
        elif key == "Deal":
            hands = val.split(", ")
            game.player1_hand = [pbn_card_to_env(c) for c in hands[0].split()]
            game.player2_hand = [pbn_card_to_env(c) for c in hands[1].split()]
        elif key == "Upcard":
            game.upcard = pbn_card_to_env(val)
        elif key == "Score":
            game.score = val
        elif key == "Deadwood":
            game.deadwood = val

    # Parse moves
    play_match = re.search(r'\[Play\]\n(.*?)(?:\n\[|\Z)', text, re.DOTALL)
    if play_match:
        for line in play_match.group(1).strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r'(\d+)\.\s+(\d+)\s+(\w+)\s*(.*)', line)
            if not m:
                continue
            move_num = int(m.group(1))
            player = int(m.group(2))
            action = m.group(3)
            card_str = m.group(4).strip() if m.group(4) else ""

            move = {"num": move_num, "player": player, "action": action}
            if card_str:
                # Remove 'x' suffix (indicates picked from discard)
                from_discard = card_str.endswith("x")
                clean = card_str.rstrip("x")
                if clean:
                    move["card"] = pbn_card_to_env(clean)
                move["from_discard"] = from_discard
            game.moves.append(move)

    # Parse groups (melds)
    groups_match = re.search(r'\[Groups\]\n(.*?)(?:\n\[|\Z)', text, re.DOTALL)
    if groups_match:
        for line in groups_match.group(1).strip().split("\n"):
            m = re.match(r'(\d+):\s*(.*)', line.strip())
            if m:
                player = int(m.group(1))
                melds_str = m.group(2).strip()
                if melds_str:
                    melds = []
                    for meld_group in melds_str.split(", "):
                        cards = [pbn_card_to_env(c) for c in meld_group.split()]
                        melds.append(cards)
                    game.groups[player] = melds

    return game


# ---------------------------------------------------------------------------
# Game state simulator
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Mutable game state reconstructed from PBN moves."""
    hands: dict[int, list[str]]  # player_id -> cards in hand
    stock: list[str]             # remaining stock cards (unknown to us)
    discard_pile: list[str]      # visible discard pile
    upcard: str                  # current top of discard / XX if face-down
    stock_size: int              # cards remaining in stock
    phase: str = "FirstUpcard"   # Current phase
    current_player: int = 1
    knock_card: int = 10
    prev_upcard: str = ""
    repeated_move: int = 0


def build_initial_state(game: PBNGame) -> GameState:
    """Build the initial game state from PBN deal."""
    # All 52 cards
    all_cards = set()
    for s in SUIT_ORDER:
        for r in RANK_ORDER:
            all_cards.add(r + s)

    dealt = set(game.player1_hand + game.player2_hand + [game.upcard])
    stock = list(all_cards - dealt)
    # Stock size = 52 - 10 dealt to players - 1 upcard = 31 (for 10-card hands)
    # or 52 - 14 - 1 = 37 (for 7-card hands), etc.
    hand_size = len(game.player1_hand)

    return GameState(
        hands={1: list(game.player1_hand), 2: list(game.player2_hand)},
        stock=stock,
        discard_pile=[],
        upcard=game.upcard,
        stock_size=52 - hand_size * 2 - 1,
        phase="FirstUpcard",
        current_player=1,  # Non-dealer acts first in FirstUpcard
        knock_card=game.knock_card,
        prev_upcard=game.upcard,
    )


# ---------------------------------------------------------------------------
# Observation rendering
# ---------------------------------------------------------------------------

def render_observation(state: GameState, player_id: int) -> str:
    """Render the full observation text matching the game server format."""
    hand = state.hands[player_id]
    opp_id = 3 - player_id
    deadwood = compute_optimal_deadwood(hand)

    # Discard pile display
    dp_display = "".join(state.discard_pile) if state.discard_pile else ""
    # Use space-separated for readability if pile exists
    if state.discard_pile:
        dp_display = "".join(state.discard_pile)

    # Upcard display
    upcard_display = state.upcard if state.phase in ("FirstUpcard", "Draw") else "XX"

    parts = []
    parts.append(f"Game: gin_rummy")
    parts.append(f"You are Player {player_id - 1}.")  # 0-indexed in env
    parts.append("")
    parts.append("Current State:")
    parts.append("")
    parts.append(f"Knock card: {state.knock_card}")
    parts.append(f"Prev upcard: {state.prev_upcard}")
    parts.append(f"Repeated move: {state.repeated_move}")
    parts.append(f"Current player: {player_id - 1}")
    parts.append(f"Phase: {state.phase}")
    parts.append("")

    # Opponent box (hidden)
    parts.append(render_card_box([], player_id=1 - (player_id - 1), hidden=True))
    parts.append("")
    parts.append(f"Stock size: {state.stock_size}  Upcard: {upcard_display}")
    parts.append(f"Discard pile: {dp_display}")
    parts.append("")

    # Player's hand box
    parts.append(render_card_box(hand, deadwood=deadwood, player_id=player_id - 1))
    parts.append("")

    # Legal actions
    parts.append("")
    parts.append("Legal Actions:")
    for aid, label in get_legal_actions(state, player_id):
        parts.append(f"  {aid} -> Player: {player_id - 1} Action: {label}")
    parts.append("")
    parts.append("Your choice (action ID only):")

    return "\n".join(parts)


def get_legal_actions(state: GameState, player_id: int) -> list[tuple[int, str]]:
    """Compute legal actions for the current phase."""
    actions = []
    hand = state.hands[player_id]

    if state.phase == "FirstUpcard":
        actions.append((52, "Draw upcard"))
        actions.append((54, "Pass"))
    elif state.phase == "Draw":
        actions.append((52, "Draw upcard"))
        actions.append((53, "Draw stock"))
    elif state.phase == "Discard":
        for card in sorted(hand, key=card_to_action_id):
            aid = card_to_action_id(card)
            actions.append((aid, card))
        # Knock is always a legal option during Discard phase if deadwood allows
        deadwood = compute_optimal_deadwood(hand)
        if deadwood <= state.knock_card:
            actions.append((55, "Knock"))
    elif state.phase == "Knock":
        # Meld declaration and discard — complex, skip for now
        pass

    return actions


# ---------------------------------------------------------------------------
# Move-to-action mapping
# ---------------------------------------------------------------------------

def move_to_action(move: dict, state: GameState, player_id: int) -> tuple[int, str] | None:
    """Convert a PBN move to (action_id, action_description).

    Returns None if the move can't be mapped (validation failure).
    """
    action = move["action"]
    card = move.get("card")
    from_discard = move.get("from_discard", False)

    if action == "passes":
        if state.phase == "FirstUpcard":
            return (54, "Pass")
        return None  # Unexpected pass

    if action == "picks":
        if from_discard:
            return (52, "Draw upcard")
        else:
            return (53, "Draw stock")

    if action == "discards":
        if card is None:
            return None
        aid = card_to_action_id(card)
        return (aid, card)

    if action == "knocks":
        return (55, "Knock")

    return None


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert game-playing AI competing in Gin Rummy.\n\n"
    "# Game Rules\n\n"
    "SETUP:\n"
    "- 52-card deck, each player receives 7-10 cards (variant dependent)\n"
    "- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\n"
    "MELDS (Valid Combinations):\n"
    "1. SET: 3+ cards of SAME RANK (e.g., 7\u2660 7\u2665 7\u2663)\n"
    "2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5\u2666 6\u2666 7\u2666)\n"
    "Examples:\n"
    "- Valid runs: A\u2660-2\u2660-3\u2660, 9\u2665-10\u2665-J\u2665-Q\u2665, 10\u2663-J\u2663-Q\u2663-K\u2663\n"
    "- Invalid: K\u2660-A\u2660-2\u2660 (Ace is LOW only, not wraparound)\n\n"
    "CARD NOTATION:\n"
    "- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n"
    "- Suits: s(spades\u2660), h(hearts\u2665), d(diamonds\u2666), c(clubs\u2663)\n"
    "- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades\n\n"
    "GAME PHASES:\n"
    "1. FirstUpcard: Choose to draw first upcard or pass (action IDs: 52=Draw upcard, 54=Pass)\n"
    "2. Draw: Choose to draw from upcard or stock pile (action IDs: 52=Draw upcard, 53=Draw stock)\n"
    "3. Discard: Choose which card to discard (action ID = card's index number, shown in Legal Actions)\n"
    "4. Layoff: After opponent knocks, add cards to their melds or pass (action IDs: card indices or 54=Pass)\n"
    "5. Knock: Declare end of hand when deadwood \u2264 knock_card value\n\n"
    "EACH TURN:\n"
    "1. DRAW phase: Pick from stock pile (53) OR discard pile upcard (52)\n"
    "2. DISCARD phase: Choose ONE card from hand to discard (use card's action ID from Legal Actions)\n\n"
    "KNOCKING:\n"
    "- When deadwood \u2264 knock_card value (8-10), you MAY knock to end hand\n"
    "- Gin: ALL cards form melds (0 deadwood) = 25-point bonus\n\n"
    "SCORING: Winner scores difference in deadwood point values.\n"
    "Card Values: A=1, 2-10=face value, J=11, Q=12, K=13\n\n"
    "# Strategy Tips\n"
    "- Early game: Draw from stock pile to see more cards and hide information\n"
    "- Build runs and sets to reduce deadwood\n"
    "- Track opponent's discards to guess their hand\n"
    "- Knock when you have \u226410 deadwood points and think you're ahead\n"
    "- Go for Gin (0 deadwood) when close for bonus points\n"
    "- Discard high-value cards that don't contribute to melds\n"
    "- Be careful picking up from the discard pile \u2014 it reveals information to opponent\n"
    "- IMPORTANT: YOU MUST PICK THE ACTION ID FROM THE LEGAL ACTIONS\n\n"
    "# Output Format\n"
    "You must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    "Examples:\n"
    "- For action \"0 -> roll\": respond \"0\"\n"
    "- For action \"89 -> a3\": respond \"89\""
)


def generate_thought(move: dict, state: GameState, player_id: int) -> str:
    """Generate a brief strategy thought for the action (template-based)."""
    action = move["action"]
    card = move.get("card")
    hand = state.hands[player_id]
    deadwood = compute_optimal_deadwood(hand)

    if action == "passes":
        return f"The upcard {state.upcard} doesn't improve my hand. I'll pass."

    if action == "picks":
        from_discard = move.get("from_discard", False)
        if from_discard:
            return f"The upcard {state.upcard} could help form melds. Drawing it."
        else:
            return f"The upcard {state.upcard} doesn't help. Drawing from stock to keep my hand hidden."

    if action == "discards":
        return f"Discarding {card} ({card_deadwood_value(card)} points) as it doesn't contribute to melds. Deadwood: {deadwood}."

    if action == "knocks":
        return f"My deadwood is {deadwood}, under the knock threshold of {state.knock_card}. Knocking to end the hand."

    return "Making the best available play."


@dataclass
class ValidationStats:
    total_games: int = 0
    valid_games: int = 0
    skipped_reasons: dict = field(default_factory=dict)

    def skip(self, reason: str):
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1


def convert_game(game: PBNGame, stats: ValidationStats) -> dict | None:
    """Convert a single PBN game to SFT format.

    Returns None if any validation check fails.
    """
    stats.total_games += 1

    if not game.moves:
        stats.skip("no_moves")
        return None

    if not game.player1_hand or not game.player2_hand:
        stats.skip("no_deal")
        return None

    state = build_initial_state(game)
    conversations = [{"from": "system", "value": SYSTEM_PROMPT}]

    # Track which player we generate data for.
    # We generate from BOTH players' perspectives for more data.
    # But we process one player at a time — pick the one with more moves.
    p1_moves = sum(1 for m in game.moves if m["player"] == 1)
    p2_moves = sum(1 for m in game.moves if m["player"] == 2)
    target_player = 1 if p1_moves >= p2_moves else 2

    first_obs = True
    num_turns = 0

    for move in game.moves:
        player = move["player"]

        # Only generate training data for the target player's turns
        if player != target_player:
            # Still need to update state for opponent moves
            if not apply_move_to_state(state, move, player):
                stats.skip("state_update_failed_opponent")
                return None
            continue

        # Validate: is this player actually the current player?
        # (Relaxed — PBN doesn't always track alternation perfectly)

        # Render observation
        obs = render_observation(state, player)

        # Get the action mapping
        action_result = move_to_action(move, state, player)
        if action_result is None:
            stats.skip("unmappable_action")
            return None
        action_id, action_label = action_result

        # Validate: is the action in the legal action set?
        legal = get_legal_actions(state, player)
        legal_ids = {a[0] for a in legal}

        # Knock (55) is always trusted from PBN — Arkadium may use slightly
        # different knock thresholds or deadwood evaluation than our DP.
        # All other actions must appear in the legal set.
        if action_id not in legal_ids and action_id != 55 and state.phase != "Knock":
            stats.skip(f"illegal_action_{state.phase}_{move['action']}")
            return None

        # Build conversation turn
        if first_obs:
            # First observation includes rules preamble (matching checkpoint format)
            obs_with_rules = "# Game Rules\n" + SYSTEM_PROMPT.split("# Game Rules\n")[1].split("# Strategy Tips")[0].strip() + "\n\n# Current Game State\n" + obs
            obs_with_rules += '\n\nYour output must strictly follow this format: "Thought:\nyour thoughts ONLY in text.\n\nAction:\nONLY your action ID (a single number)."'
            conversations.append({"from": "user", "value": obs_with_rules})
            first_obs = False
        else:
            conversations.append({"from": "user", "value": obs})

        # Assistant responds with just the action ID — matches what the
        # training environment expects ("respond with ONLY the action ID").
        conversations.append({"from": "assistant", "value": str(action_id)})

        num_turns += 1

        # Apply move to state
        if not apply_move_to_state(state, move, player):
            stats.skip("state_update_failed_self")
            return None

    if num_turns < 2:
        stats.skip("too_few_turns")
        return None

    # Determine reward from PBN score
    score_parts = game.score.split(":")
    if len(score_parts) == 2:
        s1, s2 = int(score_parts[0]), int(score_parts[1])
        if target_player == 1:
            reward = 1.0 if s1 > 0 else (0.0 if s1 == 0 and s2 == 0 else -1.0)
            outcome = "win" if s1 > 0 else ("draw" if s2 == 0 else "loss")
        else:
            reward = 1.0 if s2 > 0 else (0.0 if s1 == 0 and s2 == 0 else -1.0)
            outcome = "win" if s2 > 0 else ("draw" if s1 == 0 else "loss")
    else:
        reward = 0.0
        outcome = "unknown"

    stats.valid_games += 1
    return {
        "conversations": conversations,
        "game": "gin_rummy",
        "reward": reward,
        "outcome": outcome,
        "model": "arkadium_human",
        "seed": hash(game.gameplay_id + game.hand_number) % (2**31),
        "task_id": hash(game.gameplay_id) % (2**31),
        "num_turns": num_turns,
    }


def apply_move_to_state(state: GameState, move: dict, player: int) -> bool:
    """Apply a move to the game state. Returns False on inconsistency."""
    action = move["action"]
    card = move.get("card")
    from_discard = move.get("from_discard", False)
    hand = state.hands[player]

    if action == "passes":
        if state.phase == "FirstUpcard":
            opp = 3 - player
            # Other player gets to decide on FirstUpcard
            state.current_player = opp
            # If both passed, move to Draw phase
            if move["num"] >= 2:  # Both players had a chance
                # Check if previous move was also a pass
                state.phase = "Draw"
                state.current_player = 1  # Non-dealer draws first
        return True

    if action == "picks":
        if from_discard:
            # Pick from discard pile (upcard)
            picked = state.upcard
            if card and card != picked:
                # PBN card might differ from tracked upcard in edge cases
                picked = card
            hand.append(picked)
            state.prev_upcard = picked
            # New upcard from discard pile or XX
            if state.discard_pile:
                state.upcard = state.discard_pile[-1]
            else:
                state.upcard = "XX"
        else:
            # Pick from stock
            if card:
                hand.append(card)
            else:
                # Stock draw — we don't know the card until discard phase
                # This is a problem: PBN doesn't always record stock draws
                return False  # Can't validate
            state.stock_size -= 1
            state.prev_upcard = state.upcard

        state.phase = "Discard"
        return True

    if action == "discards":
        if card is None:
            return False
        if card not in hand:
            return False  # Validation failure
        hand.remove(card)
        state.discard_pile.append(card)
        state.upcard = card
        state.prev_upcard = card
        state.phase = "Draw"
        state.current_player = 3 - player  # Other player's turn
        return True

    if action == "knocks":
        state.phase = "Knock"
        return True

    return False


def convert_pbn_file(filepath: str, stats: ValidationStats) -> dict | None:
    """Read and convert a single PBN file."""
    with open(filepath) as f:
        text = f.read()
    game = parse_pbn(text)
    return convert_game(game, stats)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert Arkadium PBN to SFT format")
    parser.add_argument("--input-dir", default="output/sft_dataset/arkadium_raw/pbn",
                        help="Directory containing .pbn files")
    parser.add_argument("--output", default="output/sft_dataset/arkadium_converted.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of PBN files to process (0 = all)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Randomly sample N winning games for output (0 = all wins)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    pbn_dir = Path(args.input_dir)
    pbn_files = sorted(pbn_dir.glob("*.pbn"))
    if args.limit:
        pbn_files = pbn_files[:args.limit]

    print(f"Processing {len(pbn_files)} PBN files from {pbn_dir}")

    stats = ValidationStats()
    results = []

    for i, filepath in enumerate(pbn_files):
        result = convert_pbn_file(str(filepath), stats)
        if result is not None and result["outcome"] == "win":
            results.append(result)
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(pbn_files)}, wins: {len(results)}")

    print(f"\nTotal wins after conversion: {len(results)}")

    # Sample if requested
    if args.sample and args.sample < len(results):
        import random
        random.seed(args.seed)
        results = random.sample(results, args.sample)
        print(f"Sampled {args.sample} games (seed={args.seed})")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== Conversion Complete ===")
    print(f"Total PBN files: {stats.total_games}")
    print(f"Valid conversions: {stats.valid_games}")
    print(f"Conversion rate: {stats.valid_games/max(stats.total_games,1)*100:.1f}%")
    print(f"Output: {output_path} ({len(results)} examples)")
    print(f"\nSkip reasons:")
    for reason, count in sorted(stats.skipped_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
