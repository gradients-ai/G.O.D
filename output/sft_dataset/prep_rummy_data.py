"""
Prepare gin rummy SFT dataset. Combines converted Arkadium PBN games with
Claude env_training_gradients data. Output format matches the game server
exactly — no system messages, raw server observations as user, bare action
IDs as assistant.

Usage:
    python prep_rummy_data.py --arkadium-sample 500
    python prep_rummy_data.py --arkadium-sample 500 --min-turns 5
"""

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

SUITS_PBN_TO_ENV = {"S": "s", "H": "h", "D": "d", "C": "c"}
SUIT_ORDER = ["s", "c", "d", "h"]
RANK_ORDER = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
RANK_VALUES = {r: i + 1 for i, r in enumerate(RANK_ORDER)}
GRID_COLS = {r: i * 2 for i, r in enumerate(RANK_ORDER)}
GRID_WIDTH = 26

RULES_PREAMBLE = (
    "# Game Rules\n"
    "GIN RUMMY RULES:\n\n"
    "SETUP:\n"
    "- 52-card deck, each player receives 7-10 cards (variant dependent)\n"
    "- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\n"
    "MELDS (Valid Combinations):\n"
    "1. SET: 3+ cards of SAME RANK (e.g., 7\u2660 7\u2665 7\u2663)\n"
    "2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5\u2666 6\u2666 7\u2666)\n"
    "Examples:\n"
    "- Valid runs: A\u2660-2\u2660-3\u2660, 9\u2665-10\u2665-J\u2665-Q\u2665, "
    "10\u2663-J\u2663-Q\u2663-K\u2663\n"
    "- Invalid: K\u2660-A\u2660-2\u2660 (Ace is LOW only, not wraparound)\n\n"
    "CARD NOTATION:\n"
    "- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n"
    "- Suits: s(spades\u2660), h(hearts\u2665), d(diamonds\u2666), c(clubs\u2663)\n"
    "- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades\n\n"
    "GAME PHASES:\n"
    "1. FirstUpcard: Choose to draw first upcard or pass "
    "(action IDs: 52=Draw upcard, 54=Pass)\n"
    "2. Draw: Choose to draw from upcard or stock pile "
    "(action IDs: 52=Draw upcard, 53=Draw stock)\n"
    "3. Discard: Choose which card to discard "
    "(action ID = card's index number, shown in Legal Actions)\n"
    "4. Layoff: After opponent knocks, add cards to their melds or pass "
    "(action IDs: card indices or 54=Pass)\n"
    "5. Knock: Declare end of hand when deadwood \u2264 knock_card value\n\n"
    "EACH TURN:\n"
    "1. DRAW phase: Pick from stock pile (53) OR discard pile upcard (52)\n"
    "2. DISCARD phase: Choose ONE card from hand to discard "
    "(use card's action ID from Legal Actions)\n\n"
    "KNOCKING:\n"
    "- When deadwood \u2264 knock_card value (8-10), you MAY knock to end hand\n"
    "- Gin: ALL cards form melds (0 deadwood) = 25-point bonus\n\n"
    "SCORING: Winner scores difference in deadwood point values.\n"
    "Card Values: A=1, 2-10=face value, J=11, Q=12, K=13\n\n"
    "IMPORTANT: Always respond with the action ID number ONLY, never card names.\n\n"
)


def pbn_to_env(pbn_card: str) -> str:
    rank = pbn_card[1:].replace("10", "T")
    return rank + SUITS_PBN_TO_ENV[pbn_card[0]]


def card_action_id(env_card: str) -> int:
    return SUIT_ORDER.index(env_card[-1]) * 13 + RANK_ORDER.index(env_card[:-1])


def card_value(env_card: str) -> int:
    return RANK_VALUES[env_card[:-1]]


def find_all_melds(hand: list[str]) -> list[list[str]]:
    melds: list[list[str]] = []
    by_rank: dict[str, list[str]] = {}
    for c in hand:
        by_rank.setdefault(c[:-1], []).append(c)
    for cards in by_rank.values():
        if len(cards) >= 3:
            for size in range(3, len(cards) + 1):
                for combo in combinations(cards, size):
                    melds.append(list(combo))
    by_suit: dict[str, list[str]] = {}
    for c in hand:
        by_suit.setdefault(c[-1], []).append(c)
    for cards in by_suit.values():
        indices = sorted(RANK_ORDER.index(c[:-1]) for c in cards)
        idx_to_card = {RANK_ORDER.index(c[:-1]): c for c in cards}
        for start_pos in range(len(indices)):
            run = [indices[start_pos]]
            for j in range(start_pos + 1, len(indices)):
                if indices[j] == run[-1] + 1:
                    run.append(indices[j])
                else:
                    break
            if len(run) >= 3:
                for length in range(3, len(run) + 1):
                    for begin in range(len(run) - length + 1):
                        melds.append([idx_to_card[i] for i in run[begin:begin + length]])
    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for m in melds:
        key = tuple(sorted(m))
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def optimal_deadwood(hand: list[str]) -> int:
    if not hand:
        return 0
    n = len(hand)
    melds = find_all_melds(hand)
    values = [card_value(c) for c in hand]
    total = sum(values)
    masks = []
    for meld in melds:
        mask = 0
        for card in meld:
            mask |= 1 << hand.index(card)
        masks.append(mask)
    best = total

    def search(idx: int, used: int, melded: int):
        nonlocal best
        best = min(best, total - melded)
        for i in range(idx, len(masks)):
            if masks[i] & used == 0:
                mv = sum(values[b] for b in range(n) if masks[i] & (1 << b))
                search(i + 1, used | masks[i], melded + mv)

    search(0, 0, 0)
    return best


def render_box(hand: list[str], deadwood: int | None = None,
               player_id: int = 0, hidden: bool = False) -> str:
    border = "+" + "-" * GRID_WIDTH + "+"
    if hidden:
        rows = [border] + ["|" + " " * GRID_WIDTH + "|"] * 4 + [border]
        return f"Player{player_id}:\n" + "\n".join(rows)
    suit_rows: list[list[str]] = [[] for _ in range(4)]
    for card in hand:
        suit_rows[SUIT_ORDER.index(card[-1])].append(card)
    lines = [border]
    for row_cards in suit_rows:
        chars = [" "] * GRID_WIDTH
        for card in sorted(row_cards, key=lambda c: RANK_ORDER.index(c[:-1])):
            col = GRID_COLS[card[:-1]]
            chars[col] = card[:-1]
            if col + 1 < GRID_WIDTH:
                chars[col + 1] = card[-1]
        lines.append("|" + "".join(chars) + "|")
    lines.append(border)
    header = f"Player{player_id}: Deadwood={deadwood}" if deadwood is not None else f"Player{player_id}:"
    return header + "\n" + "\n".join(lines)


# ---------- PBN parsing ----------

@dataclass
class PBNGame:
    gameplay_id: str = ""
    hand_number: str = ""
    dealer: int = 0
    player1_hand: list[str] = field(default_factory=list)
    player2_hand: list[str] = field(default_factory=list)
    upcard: str = ""
    score: str = ""
    moves: list[dict] = field(default_factory=list)
    knock_card: int = 10


def parse_pbn(text: str) -> PBNGame:
    game = PBNGame()
    for m in re.finditer(r'\[(\w+)\s+"([^"]*)"\]', text):
        key, val = m.group(1), m.group(2)
        if key == "GameplayId":
            game.gameplay_id = val
        elif key == "Hand":
            game.hand_number = val
        elif key == "Dealer":
            game.dealer = int(val)
        elif key == "Deal":
            parts = val.split(", ")
            game.player1_hand = [pbn_to_env(c) for c in parts[0].split()]
            game.player2_hand = [pbn_to_env(c) for c in parts[1].split()]
        elif key == "Upcard":
            game.upcard = pbn_to_env(val)
        elif key == "Score":
            game.score = val
    play_match = re.search(r'\[Play\]\n(.*?)(?:\n\[|\Z)', text, re.DOTALL)
    if play_match:
        for line in play_match.group(1).strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r'(\d+)\.\s+(\d+)\s+(\w+)\s*(.*)', line)
            if not m:
                continue
            move: dict = {"num": int(m.group(1)), "player": int(m.group(2)), "action": m.group(3)}
            card_str = m.group(4).strip()
            if card_str:
                move["from_discard"] = card_str.endswith("x")
                clean = card_str.rstrip("x")
                if clean:
                    move["card"] = pbn_to_env(clean)
            game.moves.append(move)
    return game


# ---------- game state ----------

@dataclass
class GameState:
    hands: dict[int, list[str]]
    discard_pile: list[str]
    upcard: str
    stock_size: int
    phase: str = "FirstUpcard"
    current_player: int = 1
    knock_card: int = 10
    prev_upcard: str = "XX"
    repeated_move: int = 0


def build_state(game: PBNGame) -> GameState:
    hand_size = len(game.player1_hand)
    return GameState(
        hands={1: list(game.player1_hand), 2: list(game.player2_hand)},
        discard_pile=[],
        upcard=game.upcard,
        stock_size=52 - hand_size * 2 - 1,
        knock_card=game.knock_card,
        prev_upcard="XX",
    )


def apply_move(state: GameState, move: dict, player: int) -> bool:
    action = move["action"]
    card = move.get("card")
    from_discard = move.get("from_discard", False)
    hand = state.hands[player]

    if action == "passes":
        if state.phase == "FirstUpcard" and move["num"] >= 2:
            state.phase = "Draw"
            state.current_player = 1
        return True

    if action == "picks":
        if from_discard:
            picked = card if card else state.upcard
            hand.append(picked)
            state.prev_upcard = picked
            state.upcard = state.discard_pile[-1] if state.discard_pile else "XX"
        else:
            if not card:
                return False
            hand.append(card)
            state.stock_size -= 1
            state.prev_upcard = state.upcard
        state.phase = "Discard"
        return True

    if action == "discards":
        if not card or card not in hand:
            return False
        hand.remove(card)
        state.discard_pile.append(card)
        state.upcard = card
        state.prev_upcard = card
        state.phase = "Draw"
        state.current_player = 3 - player
        return True

    if action == "knocks":
        state.phase = "Knock"
        return True

    return False


# ---------- observation rendering (matches server exactly) ----------

def legal_actions(state: GameState, player_id: int) -> list[tuple[int, str]]:
    actions: list[tuple[int, str]] = []
    hand = state.hands[player_id]
    if state.phase == "FirstUpcard":
        actions = [(52, "Draw upcard"), (54, "Pass")]
    elif state.phase == "Draw":
        actions = [(52, "Draw upcard"), (53, "Draw stock")]
    elif state.phase == "Discard":
        for card in sorted(hand, key=card_action_id):
            actions.append((card_action_id(card), card))
        if optimal_deadwood(hand) <= state.knock_card:
            actions.append((55, "Knock"))
    return actions


def render_obs(state: GameState, player_id: int, is_reset: bool = False) -> str:
    hand = state.hands[player_id]
    dw = optimal_deadwood(hand)
    env_pid = player_id - 1
    opp_pid = 1 - env_pid
    upcard_display = state.upcard if state.phase in ("FirstUpcard", "Draw") else "XX"
    dp = "".join(state.discard_pile)

    game_state = "\n".join([
        f"Game: gin_rummy",
        f"You are Player {env_pid}.",
        "",
        "Current State:",
        "",
        f"Knock card: {state.knock_card}",
        f"Prev upcard: {state.prev_upcard}",
        f"Repeated move: {state.repeated_move}",
        f"Current player: {env_pid}",
        f"Phase: {state.phase}",
        "",
        render_box(hand, deadwood=dw, player_id=env_pid),
        "",
        f"Stock size: {state.stock_size}  Upcard: {upcard_display}",
        f"Discard pile: {dp}",
        "",
        render_box([], player_id=opp_pid, hidden=True),
        "",
        "",
        "Legal Actions:",
    ])
    for aid, label in legal_actions(state, player_id):
        game_state += f"\n  {aid} -> Player: {env_pid} Action: {label}"
    game_state += "\n\nYour choice (action ID only):"

    if is_reset:
        return RULES_PREAMBLE + "# Current Game State\n" + game_state
    return game_state


# ---------- PBN move → action ID ----------

def move_to_action_id(move: dict, state: GameState) -> int | None:
    action = move["action"]
    if action == "passes":
        return 54 if state.phase == "FirstUpcard" else None
    if action == "picks":
        return 52 if move.get("from_discard") else 53
    if action == "discards":
        card = move.get("card")
        return card_action_id(card) if card else None
    if action == "knocks":
        return 55
    return None


def _upcard_ratio(game: dict) -> float:
    asst = [m["value"] for m in game["conversations"] if m["from"] == "assistant"]
    draws = sum(1 for v in asst if v == "52")
    return draws / max(len(asst), 1)


# ---------- Arkadium conversion ----------

@dataclass
class ConversionStats:
    total: int = 0
    valid: int = 0
    wins: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str):
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


def knock_deadwood_for_game(game: PBNGame) -> int | None:
    state = build_state(game)
    p1_moves = sum(1 for m in game.moves if m["player"] == 1)
    p2_moves = sum(1 for m in game.moves if m["player"] == 2)
    target = 1 if p1_moves >= p2_moves else 2
    for move in game.moves:
        if move["action"] == "knocks" and move["player"] == target:
            return optimal_deadwood(state.hands[target])
        if not apply_move(state, move, move["player"]):
            return None
    return None


def convert_pbn_game(game: PBNGame, stats: ConversionStats,
                     knock_card_override: int | None = None) -> dict | None:
    stats.total += 1
    if not game.moves or not game.player1_hand or not game.player2_hand:
        stats.skip("no_moves_or_deal")
        return None
    if knock_card_override is not None:
        game.knock_card = knock_card_override
    state = build_state(game)
    conversations: list[dict] = []

    p1_moves = sum(1 for m in game.moves if m["player"] == 1)
    p2_moves = sum(1 for m in game.moves if m["player"] == 2)
    target = 1 if p1_moves >= p2_moves else 2
    num_turns = 0
    is_first = True

    for move in game.moves:
        player = move["player"]
        if player != target:
            if not apply_move(state, move, player):
                stats.skip("state_failed_opponent")
                return None
            continue

        obs = render_obs(state, player, is_reset=is_first)
        is_first = False

        action_id = move_to_action_id(move, state)
        if action_id is None:
            stats.skip("unmappable_action")
            return None

        valid_ids = {a[0] for a in legal_actions(state, player)}
        if action_id not in valid_ids and action_id != 55 and state.phase != "Knock":
            stats.skip(f"illegal_{state.phase}_{move['action']}")
            return None

        conversations.append({"from": "user", "value": obs})
        conversations.append({"from": "assistant", "value": str(action_id)})
        num_turns += 1

        if not apply_move(state, move, player):
            stats.skip("state_failed_self")
            return None

    if num_turns < 2:
        stats.skip("too_few_turns")
        return None

    score_parts = game.score.split(":")
    if len(score_parts) == 2:
        s1, s2 = int(score_parts[0]), int(score_parts[1])
        reward = 1.0 if (s1 > 0 and target == 1) or (s2 > 0 and target == 2) else 0.0
        outcome = "win" if reward > 0 else "loss"
    else:
        return None
    if outcome != "win":
        return None

    stats.valid += 1
    stats.wins += 1
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


def convert_arkadium(pbn_dir: Path, sample: int, min_turns: int,
                     seed: int,
                     claude_kc_counts: dict[int, int] | None = None) -> list[dict]:
    pbn_files = sorted(pbn_dir.glob("*.pbn"))
    if not pbn_files:
        print(f"  No PBN files found in {pbn_dir}")
        return []

    random.seed(seed)
    stats = ConversionStats()
    pool_by_kc: dict[int, list[dict]] = {8: [], 9: [], 10: []}

    for i, fpath in enumerate(pbn_files):
        text = fpath.read_text()
        knock_dw = knock_deadwood_for_game(parse_pbn(text))

        result_10 = convert_pbn_game(parse_pbn(text), stats)
        if result_10 and result_10["num_turns"] >= min_turns:
            pool_by_kc[10].append(result_10)

        if knock_dw is not None and knock_dw <= 8:
            r8 = convert_pbn_game(parse_pbn(text), ConversionStats(), knock_card_override=8)
            if r8 and r8["num_turns"] >= min_turns:
                pool_by_kc[8].append(r8)

        if knock_dw is not None and knock_dw <= 9:
            r9 = convert_pbn_game(parse_pbn(text), ConversionStats(), knock_card_override=9)
            if r9 and r9["num_turns"] >= min_turns:
                pool_by_kc[9].append(r9)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(pbn_files)}, "
                  f"pools: k8={len(pool_by_kc[8])}, k9={len(pool_by_kc[9])}, k10={len(pool_by_kc[10])}")

    print(f"  PBN files: {stats.total}")
    print(f"  Pools: k8={len(pool_by_kc[8])}, k9={len(pool_by_kc[9])}, k10={len(pool_by_kc[10])}")

    claude_kc = claude_kc_counts or {8: 0, 9: 0, 10: 0}
    if sample and sample > 0:
        target_per_kc = (sample + sum(claude_kc.values())) // 3
        max_aggressive_ratio = {8: 0.03, 9: 0.06, 10: 0.15}
        results: list[dict] = []
        for kc in [8, 9, 10]:
            need = max(0, target_per_kc - claude_kc.get(kc, 0))
            pool = pool_by_kc[kc]
            agg_cap = max_aggressive_ratio[kc]
            conservative = [g for g in pool if _upcard_ratio(g) <= 0.3]
            aggressive = [g for g in pool if _upcard_ratio(g) > 0.3]
            max_agg = int(need * agg_cap)
            random.shuffle(conservative)
            random.shuffle(aggressive)
            agg_take = min(max_agg, len(aggressive))
            con_take = min(need - agg_take, len(conservative))
            selected = aggressive[:agg_take] + conservative[:con_take]
            random.shuffle(selected)
            results.extend(selected)
            agg_count = sum(1 for g in selected if _upcard_ratio(g) > 0.3)
            print(f"  k{kc}: target={target_per_kc}, claude={claude_kc.get(kc, 0)}, "
                  f"arkadium={len(selected)} ({agg_count} aggressive), pool={len(pool)}")
    else:
        results = pool_by_kc[10]

    if stats.skip_reasons:
        for reason, count in sorted(stats.skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    skipped {reason}: {count}")

    if sample and 0 < sample < len(results):
        results = stratified_sample(results, sample, seed)

    return results


def stratified_sample(games: list[dict], n: int, seed: int) -> list[dict]:
    random.seed(seed)

    def bucket_key(game: dict) -> str:
        turns = game["num_turns"]
        ratio = _upcard_ratio(game)
        if turns <= 8:
            length_bucket = "short"
        elif turns <= 18:
            length_bucket = "medium"
        else:
            length_bucket = "long"
        style = "aggressive" if ratio > 0.3 else "conservative"
        has_knock = any(m.get("value") == "55" for m in game["conversations"] if m["from"] == "assistant")
        knock = "knock" if has_knock else "no_knock"
        return f"{length_bucket}_{style}_{knock}"

    buckets: dict[str, list[dict]] = {}
    for game in games:
        key = bucket_key(game)
        buckets.setdefault(key, []).append(game)

    min_per_bucket = max(1, n // (len(buckets) * 5))
    sampled: list[dict] = []
    for key in sorted(buckets):
        pool = buckets[key]
        take = min_per_bucket
        if key.startswith("long_"):
            take = int(take * 1.5)
        take = min(take, len(pool))
        sampled.extend(random.sample(pool, take))

    remaining = n - len(sampled)
    if remaining > 0:
        used = set(id(g) for g in sampled)
        leftover = [g for g in games if id(g) not in used]
        sampled.extend(random.sample(leftover, min(remaining, len(leftover))))

    sampled = sampled[:n]
    random.shuffle(sampled)

    bucket_counts = Counter(bucket_key(g) for g in sampled)
    print(f"  Stratified {len(sampled)} games across {len(bucket_counts)} buckets:")
    for key, count in sorted(bucket_counts.items()):
        print(f"    {key}: {count}")

    return sampled


# ---------- Claude data processing ----------

def process_claude(claude_dir: Path, min_turns: int) -> tuple[list[dict], dict[int, int]]:
    parquet_files = list(claude_dir.glob("**/*.parquet"))
    if not parquet_files:
        print(f"  No parquet files found in {claude_dir}")
        return [], {8: 0, 9: 0, 10: 0}

    import pandas as pd
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["game"] == "gin_rummy"]

    results: list[dict] = []
    stripped = 0
    kc_counts: dict[int, int] = {8: 0, 9: 0, 10: 0}

    for _, row in df.iterrows():
        if row.get("reward", 0) <= 0:
            continue
        if row.get("num_turns", 0) < min_turns:
            continue

        convs = row["conversations"]
        clean_convs: list[dict] = []
        valid = True

        for msg in convs:
            if msg["from"] == "system":
                continue
            elif msg["from"] == "assistant":
                text = msg["value"]
                after = text.split("Action:")[-1].strip() if "Action:" in text else text.strip()
                m = re.search(r'-?\d+', after)
                if not m:
                    valid = False
                    break
                action_id = m.group(0)
                if text.strip() != action_id:
                    stripped += 1
                clean_convs.append({"from": "assistant", "value": action_id})
            else:
                clean_convs.append(msg)

        if not valid or not clean_convs:
            continue

        for msg in clean_convs:
            if msg["from"] == "user":
                m = re.search(r"Knock card: (\d+)", msg["value"])
                if m:
                    kc = int(m.group(1))
                    kc_counts[kc] = kc_counts.get(kc, 0) + 1
                break

        results.append({
            "conversations": clean_convs,
            "game": "gin_rummy",
            "reward": float(row["reward"]),
            "outcome": row["outcome"],
            "model": row.get("model", "claude"),
            "seed": int(row.get("seed", 0)),
            "task_id": int(row.get("task_id", 0)),
            "num_turns": int(row["num_turns"]),
        })

    print(f"  Claude gin rummy: {len(df)} total, {len(results)} wins kept, {stripped} responses stripped")
    print(f"  Claude knock card counts: {dict(sorted(kc_counts.items()))}")
    return results, kc_counts


# ---------- main ----------

def prep_rummy_data(
    pbn_dir: Path,
    claude_dir: Path,
    output_path: Path,
    arkadium_sample: int = 500,
    min_turns: int = 2,
    seed: int = 42,
):
    print("Processing Claude data...")
    claude, claude_kc = process_claude(claude_dir, min_turns)

    print("\nConverting Arkadium PBN files...")
    arkadium = convert_arkadium(pbn_dir, arkadium_sample, min_turns, seed,
                                claude_kc_counts=claude_kc)

    merged = claude + arkadium
    random.seed(seed)
    random.shuffle(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in merged:
            f.write(json.dumps(entry) + "\n")

    sources = Counter(d["model"] for d in merged)
    knock_cards: Counter = Counter()
    total_turns = 0
    for d in merged:
        total_turns += d["num_turns"]
        for msg in d["conversations"]:
            if msg["from"] == "user":
                m = re.search(r"Knock card: (\d+)", msg["value"])
                if m:
                    knock_cards[int(m.group(1))] += 1
                break

    print(f"\n{'='*40}")
    print(f"Output: {output_path}")
    print(f"Total games: {len(merged)}")
    print(f"Total training turns: {total_turns}")
    print(f"Sources: {dict(sources)}")
    print(f"Knock card distribution: {dict(sorted(knock_cards.items()))}")


def main():
    parser = argparse.ArgumentParser(description="Prepare gin rummy SFT dataset")
    parser.add_argument("--pbn-dir", type=Path,
                        default=Path("output/sft_dataset/cache/miner_datasets/gradients-io-tournaments--ArkadiumGinrummy/pbn"))
    parser.add_argument("--claude-dir", type=Path,
                        default=Path("output/sft_dataset/cache/miner_datasets/gradients-io-tournaments--env_training_gradients"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/sft_dataset/gin_rummy_sft.jsonl"))
    parser.add_argument("--arkadium-sample", type=int, default=500)
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prep_rummy_data(
        pbn_dir=args.pbn_dir,
        claude_dir=args.claude_dir,
        output_path=args.output,
        arkadium_sample=args.arkadium_sample,
        min_turns=args.min_turns,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
