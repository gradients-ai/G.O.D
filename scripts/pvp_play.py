#!/usr/bin/env python3
"""Manual PvP stepper — play both sides by hand against the real harness.

No GPU/SGLang needed: this renders the *exact* prompt the LLMBot would send each
turn (system = rules + rendered memory, user = state + legal actions, plus the
tool list), lets you apply tool calls (memory writes + the move), and advances
the game. State is persisted to /tmp so you step one turn per invocation.

Usage:
    python scripts/pvp_play.py new --env leduc_poker --seed 42
    python scripts/pvp_play.py show
    python scripts/pvp_play.py act --action 1 --mem long_term_memory_append 1 "opp folds to raises"
    python scripts/pvp_play.py show
    ...
    python scripts/pvp_play.py reflect --seat 0      # after the game ends
"""

import argparse
import pickle
import random
import sys

from core.constants import EnvironmentName
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatResult
from core.models.pvp_models import MemoryArea
from core.pvp import tools as tool_lib
from core.pvp.bot import LLMBot
from core.pvp.game_eval import _AGENT_REGISTRY
from core.pvp.memory import SlotMemory
from core.pvp.memory import WhitespaceTokenCounter
from validator.core import constants as vcst


STATE_PATH = "/tmp/pvp_play.pkl"
_COUNTER = WhitespaceTokenCounter()


def _env(name: str) -> EnvironmentName:
    return EnvironmentName(name)


def _agent_for(env: EnvironmentName):
    return _AGENT_REGISTRY[env]()


def _blank_seat_mem() -> dict:
    return {
        MemoryArea.WORKING.value: {},
        MemoryArea.LONG_TERM.value: {},
    }


def _slotmem(area: MemoryArea, stored: dict) -> SlotMemory:
    if area is MemoryArea.WORKING:
        mem = SlotMemory(vcst.PVP_WORKING_MEM_SLOTS, vcst.PVP_WORKING_SLOT_TOKENS, _COUNTER)
    else:
        mem = SlotMemory(vcst.PVP_LONGTERM_MEM_SLOTS, vcst.PVP_LONGTERM_SLOT_TOKENS, _COUNTER)
    for k, v in stored.items():
        mem.slots[int(k)] = v
    return mem


def _memories_for_seat(bundle: dict, seat: int) -> dict:
    stored = bundle["mem"][seat]
    return {area: _slotmem(area, stored[area.value]) for area in MemoryArea}


def _store_memories(bundle: dict, seat: int, memories: dict) -> None:
    bundle["mem"][seat] = {area.value: mem.to_dict() for area, mem in memories.items()}


def _load() -> dict:
    with open(STATE_PATH, "rb") as f:
        return pickle.load(f)


def _save(bundle: dict) -> None:
    with open(STATE_PATH, "wb") as f:
        pickle.dump(bundle, f)


def _rehydrate(bundle: dict):
    agent = _agent_for(_env(bundle["env"]))
    game = agent.load_game(bundle["params"])
    state = game.deserialize_state(bundle["state_str"])
    rng = random.Random()
    rng.setstate(bundle["rng_state"])
    return game, state, rng


def _resolve_chance(state, rng) -> None:
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        state.apply_action(rng.choices(actions, weights=probs, k=1)[0])


def _bot_for(game, state, bundle, seat: int) -> LLMBot:
    agent = _agent_for(_env(bundle["env"]))
    return LLMBot(
        game=game,
        player_id=seat,
        chat_fn=lambda *a, **k: ChatResult(),  # unused: we only render prompts
        config=ChatCompletionConfig(inference_model="manual", base_url="http://localhost/v1"),
        agent=agent,
        memories=_memories_for_seat(bundle, seat),
    )


def cmd_new(args) -> None:
    env = _env(args.env)
    agent = _agent_for(env)
    params = agent.generate_params(args.seed)
    game = agent.load_game(params)
    state = game.new_initial_state()
    agent.setup_initial_state(state, args.seed)  # seeded opening plies, as in eval
    rng = random.Random(args.seed)
    _resolve_chance(state, rng)
    bundle = {
        "env": env.value,
        "game_name": agent.game_name,
        "params": params,
        "state_str": state.serialize(),
        "rng_state": rng.getstate(),
        "mem": {0: _blank_seat_mem(), 1: _blank_seat_mem()},
        "turn": 0,
    }
    _save(bundle)
    print(f"New {env.value} game (seed {args.seed}). Current player: {state.current_player()}\n")
    cmd_show(args)


def cmd_show(args) -> None:
    bundle = _load()
    game, state, _ = _rehydrate(bundle)
    if state.is_terminal():
        print(f"=== GAME OVER === returns: {state.returns()}")
        print("Run `reflect --seat 0` / `--seat 1` to consolidate long-term memory.")
        return

    seat = state.current_player()
    legal = state.legal_actions(seat)
    bot = _bot_for(game, state, bundle, seat)
    tools = bot._memory_tools + [tool_lib.build_game_action_tool(bot._legal_hint(legal), legal)]

    print(f"################  TURN {bundle['turn']}  —  PLAYER {seat} TO ACT  ################\n")
    print("---------------- SYSTEM MESSAGE (input) ----------------")
    print(bot._system_prompt())
    print("\n---------------- USER MESSAGE (input) ----------------")
    print(bot._user_prompt(state, legal))
    print("\n---------------- TOOLS OFFERED ----------------")
    for t in tools:
        print(f"  - {t.function.name}")
    print("\n(Respond with: act --action <id> [--mem <tool> <slot> <text>] ...)\n")


def cmd_act(args) -> None:
    bundle = _load()
    game, state, rng = _rehydrate(bundle)
    seat = state.current_player()
    legal = state.legal_actions(seat)

    memories = _memories_for_seat(bundle, seat)
    print(f"---------------- PLAYER {seat} OUTPUT (your tool calls) ----------------")
    for tool_name, slot, text in args.mem or []:
        result = tool_lib.execute_memory_tool(memories, tool_name, {"slot": int(slot), "content": text})
        print(f"  {tool_name}(slot={slot}): {result}")
    _store_memories(bundle, seat, memories)

    if args.action is None:
        print("  (no game_action — memory-only step; same player still to act)")
        _save(bundle)
        return

    if args.action not in legal:
        print(f"  game_action({args.action}) -> ERROR: not legal (legal={legal}); not applied")
        _save(bundle)
        sys.exit(1)

    action_str = state.action_to_string(seat, args.action)
    print(f"  game_action({args.action}) -> {action_str}  [committed]")
    state.apply_action(args.action)
    _resolve_chance(state, rng)

    bundle["state_str"] = state.serialize()
    bundle["rng_state"] = rng.getstate()
    bundle["turn"] += 1
    _save(bundle)
    print()
    cmd_show(args)


def cmd_reflect(args) -> None:
    bundle = _load()
    game, state, _ = _rehydrate(bundle)
    bot = _bot_for(game, state, bundle, args.seat)
    print(f"---------------- REFLECTION INPUT (player {args.seat}) ----------------")
    print(bot._reflection_system_prompt())
    print()
    returns = state.returns() if state.is_terminal() else [0, 0]
    from core.models.pvp_models import GameOutcome

    outcome = GameOutcome.DRAW
    if state.is_terminal():
        if returns[args.seat] > 0:
            outcome = GameOutcome.WIN
        elif returns[args.seat] < 0:
            outcome = GameOutcome.LOSS
    print(bot._reflection_user_prompt(state, outcome))
    print("\n(Apply consolidation with: act-reflect --seat N --mem <tool> <slot> <text> ...)")


def cmd_act_reflect(args) -> None:
    bundle = _load()
    memories = _memories_for_seat(bundle, args.seat)
    print(f"---------------- PLAYER {args.seat} REFLECTION OUTPUT ----------------")
    for tool_name, slot, text in args.mem or []:
        result = tool_lib.execute_memory_tool(memories, tool_name, {"slot": int(slot), "content": text})
        print(f"  {tool_name}(slot={slot}): {result}")
    _store_memories(bundle, args.seat, memories)
    _save(bundle)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pn = sub.add_parser("new")
    pn.add_argument("--env", default="leduc_poker")
    pn.add_argument("--seed", type=int, default=42)
    pn.set_defaults(func=cmd_new)

    sub.add_parser("show").set_defaults(func=cmd_show)

    pa = sub.add_parser("act")
    pa.add_argument("--action", type=int, default=None)
    pa.add_argument("--mem", nargs=3, action="append", metavar=("TOOL", "SLOT", "TEXT"))
    pa.set_defaults(func=cmd_act)

    pr = sub.add_parser("reflect")
    pr.add_argument("--seat", type=int, required=True)
    pr.set_defaults(func=cmd_reflect)

    par = sub.add_parser("act-reflect")
    par.add_argument("--seat", type=int, required=True)
    par.add_argument("--mem", nargs=3, action="append", metavar=("TOOL", "SLOT", "TEXT"))
    par.set_defaults(func=cmd_act_reflect)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
