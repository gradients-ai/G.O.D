#!/usr/bin/env python3
"""Manual PvP stepper for the real tool-calling harness."""

import argparse
import pickle
import random
import sys

from core.constants.environments import EnvironmentName
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatResult
from core.models.pvp_models import GameOutcome
from core.models.pvp_models import MemoryArea
from core.pvp import constants as pvp_cst
from core.pvp import tools as tool_lib
from core.pvp.bot import LLMBot
from core.pvp.game_eval import _AGENT_REGISTRY
from core.pvp.memory import SlotMemory
from core.pvp.memory import WhitespaceTokenCounter


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
        mem = SlotMemory(pvp_cst.PVP_WORKING_MEM_SLOTS, pvp_cst.PVP_WORKING_SLOT_TOKENS, _COUNTER)
    else:
        mem = SlotMemory(pvp_cst.PVP_LONGTERM_MEM_SLOTS, pvp_cst.PVP_LONGTERM_SLOT_TOKENS, _COUNTER)
    for key, value in stored.items():
        mem.slots[int(key)] = value
    return mem


def _memories_for_seat(bundle: dict, seat: int) -> dict:
    stored = bundle["mem"][seat]
    return {area: _slotmem(area, stored[area.value]) for area in MemoryArea}


def _store_memories(bundle: dict, seat: int, memories: dict) -> None:
    bundle["mem"][seat] = {area.value: mem.to_dict() for area, mem in memories.items()}


def _load() -> dict:
    with open(STATE_PATH, "rb") as file:
        return pickle.load(file)


def _save(bundle: dict) -> None:
    with open(STATE_PATH, "wb") as file:
        pickle.dump(bundle, file)


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


def _bot_for(game, bundle, seat: int) -> LLMBot:
    agent = _agent_for(_env(bundle["env"]))
    return LLMBot(
        game=game,
        player_id=seat,
        chat_fn=lambda *args, **kwargs: ChatResult(),
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
    agent.setup_initial_state(state, args.seed)
    rng = random.Random(args.seed)
    _resolve_chance(state, rng)
    bundle = {
        "env": env.value,
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
    bot = _bot_for(game, bundle, seat)
    tools = bot._memory_tools + [tool_lib.build_game_action_tool(bot._legal_hint(legal), legal)]

    print(f"################  TURN {bundle['turn']} - PLAYER {seat} TO ACT  ################\n")
    print("---------------- SYSTEM MESSAGE (input) ----------------")
    print(bot._system_prompt())
    print("\n---------------- USER MESSAGE (input) ----------------")
    print(bot._user_prompt(state, legal))
    print("\n---------------- TOOLS OFFERED ----------------")
    for tool in tools:
        print(f"  - {tool.function.name}")
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
        print("  (no game_action - memory-only step; same player still to act)")
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
    if not state.is_terminal():
        print("Game is not terminal yet.")
        sys.exit(1)

    seat = args.seat
    bot = _bot_for(game, bundle, seat)
    outcome = GameOutcome.WIN if state.returns()[seat] > state.returns()[1 - seat] else GameOutcome.LOSS
    if state.returns()[seat] == state.returns()[1 - seat]:
        outcome = GameOutcome.DRAW
    bot.reflect(state, outcome)
    _store_memories(bundle, seat, bot.memories)
    _save(bundle)
    print(f"Reflected for seat {seat}: {outcome.value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    new = sub.add_parser("new")
    new.add_argument("--env", default="leduc_poker")
    new.add_argument("--seed", type=int, default=42)
    new.set_defaults(func=cmd_new)

    show = sub.add_parser("show")
    show.set_defaults(func=cmd_show)

    act = sub.add_parser("act")
    act.add_argument("--action", type=int)
    act.add_argument("--mem", nargs=3, action="append", metavar=("TOOL", "SLOT", "TEXT"))
    act.set_defaults(func=cmd_act)

    reflect = sub.add_parser("reflect")
    reflect.add_argument("--seat", type=int, choices=[0, 1], required=True)
    reflect.set_defaults(func=cmd_reflect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
