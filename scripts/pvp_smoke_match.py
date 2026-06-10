#!/usr/bin/env python3
"""GPU smoke test: two SGLang-served models play the PvP harness head-to-head.

Tier-1 validation for the tool-calling memory harness. It answers the three
questions that decide whether the eval is even viable, none of which a CPU/mock
test can reach:

  1. Does tool calling actually work end to end? (valid game_action rate per turn)
  2. What is the forfeit pressure? (turns that produced no committable move)
  3. What does a turn cost? (completion tokens + wall-clock latency vs the 15s budget)

It starts two real SGLang servers (one per GPU), runs matchups across the
configured environments through the production harness (run_matchup), and prints
a per-environment scoreline plus per-player instrumentation.

  # self-play, the recommended harness-validation run
  python scripts/pvp_smoke_match.py \
      --model-a Qwen/Qwen2.5-7B-Instruct --model-b Qwen/Qwen2.5-7B-Instruct --num-games 5

  # asymmetric / breakage probe (strong vs the tournament weight class)
  python scripts/pvp_smoke_match.py \
      --model-a Qwen/Qwen2.5-7B-Instruct --model-b Qwen/Qwen2.5-1.5B-Instruct --num-games 5

  # point at already-running servers instead of launching them
  python scripts/pvp_smoke_match.py \
      --base-url-a http://localhost:30000/v1 --base-url-b http://localhost:30001/v1

The SGLang tool-call parser is auto-resolved from the model family (see
core/pvp/sglang_parsers.py); pass --tool-call-parser only to override it.
Without a parser SGLang returns tool calls as plain text, the bot sees no
move, and every turn forfeits.
"""

import argparse
import asyncio
import os
import statistics
import time
from dataclasses import dataclass
from dataclasses import field

from core.constants import EnvironmentName
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatMessage
from core.models.pvp_models import ChatResult
from core.models.pvp_models import PreparedModel
from core.models.pvp_models import PvPMatchupConfig
from core.models.pvp_models import ToolSchema
from core.pvp.chat import chat_completion
from core.pvp.chat import create_client
from core.pvp.sglang_parsers import TOOL_CALL_PARSER_ENV
from validator.core import constants as vcst
from core.pvp.game_eval import _AGENT_REGISTRY
from validator.evaluation.pvp.game_runner import Player
from validator.evaluation.pvp.game_runner import run_matchup
from validator.evaluation.pvp.game_runner import warmup_player
from validator.evaluation.pvp.server import start_sglang
from validator.evaluation.pvp.server import wait_for_servers
from validator.evaluation.utils import stop_process


# --- Per-call instrumentation ---------------------------------------------------


@dataclass
class CallStat:
    """One model call, classified as a turn (offers game_action) or a reflection."""

    kind: str  # "turn" | "reflect"
    latency_s: float
    completion_tokens: int | None
    had_game_action: bool
    had_any_tool: bool
    working_writes: int = 0
    longterm_writes: int = 0


def _split_mem_blocks(system: str) -> tuple[str, str]:
    """Return (working_block, long_term_block) from a rendered system prompt."""
    if "LONG_TERM" not in system:
        return "", ""
    working = system.split("WORKING", 1)[-1].split("LONG_TERM", 1)[0]
    longterm = system.split("LONG_TERM", 1)[-1].split("You get ONE response", 1)[0]
    return working, longterm


def _all_slots_empty(block: str) -> bool:
    slot_lines = [l for l in block.splitlines() if l.strip().startswith("[")]
    return bool(slot_lines) and all("(empty)" in l for l in slot_lines)


@dataclass
class Recorder:
    """Wraps chat_completion to time it and record what came back. Acts as a ChatFn."""

    label: str
    client: object
    stats: list[CallStat] = field(default_factory=list)
    mem_samples: list[str] = field(default_factory=list)  # (tool -> content) examples
    last_turn_system: str | None = None  # latest rendered memory the model saw on a turn
    game_start_longterm: list[str] = field(default_factory=list)  # long-term seen at each game's first turn
    turn_reasoning: list[str] = field(default_factory=list)  # model's content/reasoning on turns

    def __call__(
        self,
        config: ChatCompletionConfig,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
    ) -> ChatResult:
        tool_names = {t.function.name for t in tools or []}
        kind = "turn" if "game_action" in tool_names else "reflect"
        if kind == "turn" and messages:
            self.last_turn_system = messages[0].content
            working_block, longterm_block = _split_mem_blocks(messages[0].content or "")
            # Working memory resets each game, so an empty working block marks a game's
            # first turn — snapshot the long-term it carried in.
            if _all_slots_empty(working_block):
                self.game_start_longterm.append(longterm_block.strip())
        start = time.perf_counter()
        try:
            result = chat_completion(self.client, config, messages, tools)
        except Exception:
            # A raised call (timeout/context/transport) is itself a forfeit signal;
            # record it as a turn that produced no move so the rate reflects reality.
            self.stats.append(CallStat(kind, time.perf_counter() - start, None, False, False))
            raise
        if kind == "turn" and result.content and len(self.turn_reasoning) < 8:
            self.turn_reasoning.append(" ".join(result.content.split())[:200])
        calls = result.tool_calls or []
        usage = result.usage or {}
        working = sum(1 for c in calls if c.name.startswith("working_memory"))
        longterm = sum(1 for c in calls if c.name.startswith("long_term_memory"))
        for c in calls:
            if c.name != "game_action" and len(self.mem_samples) < 30:
                self.mem_samples.append(f"[{kind}] {c.name}(slot={c.arguments.get('slot')}): {str(c.arguments.get('content'))[:90]}")
        self.stats.append(
            CallStat(
                kind=kind,
                latency_s=time.perf_counter() - start,
                completion_tokens=usage.get("completion_tokens"),
                had_game_action=any(c.name == "game_action" for c in calls),
                had_any_tool=bool(calls),
                working_writes=working,
                longterm_writes=longterm,
            )
        )
        return result


def _build_player(
    model: str, base_url: str, tokenizer_repo: str | None, label: str, temperature: float, seed: int
) -> tuple[Player, Recorder]:
    config = ChatCompletionConfig(
        inference_model=model,
        tokenizer_repo=tokenizer_repo,
        base_url=base_url,
        temperature=temperature,
        seed=seed,
        read_timeout=vcst.PVP_HTTP_READ_TIMEOUT_SECONDS,
        max_retries=vcst.PVP_HTTP_MAX_RETRIES,
    )
    client = create_client(config)
    recorder = Recorder(label=label, client=client)
    return Player(client=client, config=config, chat_fn=recorder), recorder


# --- Reporting ------------------------------------------------------------------


def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[idx]


def _report_player(label: str, model: str, recorder: Recorder) -> None:
    turns = [s for s in recorder.stats if s.kind == "turn"]
    reflects = [s for s in recorder.stats if s.kind == "reflect"]
    committed = [s for s in turns if s.had_game_action]
    turn_latency = [s.latency_s for s in turns]
    turn_tokens = [s.completion_tokens for s in turns if s.completion_tokens is not None]

    print(f"\n  [{label}] {model}")
    print(f"    turn calls           : {len(turns)}")
    print(f"    committed a move     : {len(committed)} ({_pct(len(committed), len(turns))})  <- tool calling works iff high")
    print(f"    no move (forfeit)    : {len(turns) - len(committed)} ({_pct(len(turns) - len(committed), len(turns))})")
    if turn_latency:
        print(
            f"    turn latency (s)     : p50={_quantile(turn_latency, 0.5):.2f}  "
            f"p95={_quantile(turn_latency, 0.95):.2f}  max={max(turn_latency):.2f}  "
            f"(budget {vcst.PVP_TURN_TIMEOUT_SECONDS}s)"
        )
    if turn_tokens:
        print(
            f"    completion tokens    : mean={statistics.mean(turn_tokens):.0f}  "
            f"max={max(turn_tokens)}  (cap {vcst.PVP_TURN_MAX_TOKENS})"
        )
    print(f"    reflection calls     : {len(reflects)}")
    t_work = sum(s.working_writes for s in turns)
    t_long = sum(s.longterm_writes for s in turns)
    r_work = sum(s.working_writes for s in reflects)
    r_long = sum(s.longterm_writes for s in reflects)
    print(f"    writes during turns  : working={t_work}  long_term={t_long}")
    print(f"    writes in reflection : working={r_work}  long_term={r_long}  <- reflection works iff >0")
    if recorder.mem_samples:
        print("    sample writes        :")
        for s in recorder.mem_samples:
            print(f"      - {s}")
    if recorder.game_start_longterm:
        print("    long-term carried INTO each game's first turn (proves carryover):")
        for i, blk in enumerate(recorder.game_start_longterm, 1):
            filled = [l.strip() for l in blk.splitlines() if l.strip().startswith("[") and "(empty)" not in l]
            summary = "(empty — first game)" if not filled else f"{len(filled)} slot(s) carried: {filled[0][:90]}"
            print(f"      game {i}: {summary}")
    if recorder.turn_reasoning:
        print("    turn reasoning samples (usage = references opponent/notes):")
        for r in recorder.turn_reasoning[:5]:
            print(f"      - {r}")


# --- Main -----------------------------------------------------------------------


_ENV_CHOICES = [e.value for e in _AGENT_REGISTRY]  # every registered PvP env (incl. othello)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-a", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-b", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--tool-call-parser",
        default=None,
        help="Override the SGLang parser; auto-resolved from the model family when omitted",
    )
    parser.add_argument("--envs", nargs="+", default=_ENV_CHOICES, choices=_ENV_CHOICES)
    parser.add_argument("--num-games", type=int, default=5, help="seeds per env; each played twice (position swap)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-a", type=int, default=0)
    parser.add_argument("--gpu-b", type=int, default=1)
    parser.add_argument("--port-a", type=int, default=vcst.PVP_SGLANG_PORT_A)
    parser.add_argument("--port-b", type=int, default=vcst.PVP_SGLANG_PORT_B)
    parser.add_argument("--base-url-a", default=None, help="Skip launch; use a running server (e.g. http://h:30000/v1)")
    parser.add_argument("--base-url-b", default=None)
    args = parser.parse_args()

    if (args.base_url_a is None) != (args.base_url_b is None):
        parser.error("pass both --base-url-a and --base-url-b, or neither")
    launching = args.base_url_a is None
    base_a = args.base_url_a or ""
    base_b = args.base_url_b or ""
    procs: list = []
    try:
        if launching:
            # build_sglang_command auto-resolves the parser from the model path;
            # an explicit --tool-call-parser flows through the env override so it
            # wins (and doesn't duplicate the auto-resolved flag).
            if args.tool_call_parser:
                os.environ[TOOL_CALL_PARSER_ENV] = args.tool_call_parser
            prepared_a = PreparedModel(sglang_model_path=args.model_a, inference_name=args.model_a)
            prepared_b = PreparedModel(sglang_model_path=args.model_b, inference_name=args.model_b)
            print(f"Launching SGLang: A={args.model_a} (gpu {args.gpu_a}), B={args.model_b} (gpu {args.gpu_b})")
            procs = [
                start_sglang(prepared_a, args.gpu_a, args.port_a, args.seed),
                start_sglang(prepared_b, args.gpu_b, args.port_b, args.seed + 1),
            ]
            asyncio.run(wait_for_servers(args.port_a, args.port_b))
            base_a = f"http://{vcst.PVP_SGLANG_HOST}:{args.port_a}{vcst.PVP_SGLANG_API_PATH}"
            base_b = f"http://{vcst.PVP_SGLANG_HOST}:{args.port_b}{vcst.PVP_SGLANG_API_PATH}"

        # tokenizer_repo = served weights so memory slot budgets use real tokens.
        player_a, rec_a = _build_player(args.model_a, base_a, args.model_a, "A", args.temperature, args.seed)
        player_b, rec_b = _build_player(args.model_b, base_b, args.model_b, "B", args.temperature, args.seed)
        warmup_player(player_a)
        warmup_player(player_b)

        print(f"\n=== A={args.model_a}  vs  B={args.model_b} ===")
        wall_start = time.perf_counter()
        for env_value in args.envs:
            env_name = EnvironmentName(env_value)
            result = run_matchup(
                env_name=env_name,
                matchup_config=PvPMatchupConfig(num_games=args.num_games),
                player_a=player_a,
                player_b=player_b,
                base_seed=args.seed,
            )
            print(
                f"\n[{env_value}] A={result.model_a_wins} B={result.model_b_wins} "
                f"draws={result.draws} total={result.total_games}"
            )
        wall_s = time.perf_counter() - wall_start

        print("\n" + "=" * 70)
        print(f"WALL TIME: {wall_s:.1f}s for {len(args.envs)} env(s) x {args.num_games} seeds x2")
        _report_player("A", args.model_a, rec_a)
        _report_player("B", args.model_b, rec_b)
        print("=" * 70)

        player_a.client.close()
        player_b.client.close()
    finally:
        for i, proc in enumerate(procs):
            stop_process(proc, f"sglang-{i}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
