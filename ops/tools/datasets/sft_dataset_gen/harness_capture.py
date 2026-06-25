#!/usr/bin/env python3
"""Generate PvP tool-calling SFT data through the real harness."""

import argparse
import functools
import json
import logging
import sys
import time
from pathlib import Path

import anthropic

import validator.evaluation.constants as eval_cst
from core.constants.environments import ENVIRONMENT_CONFIGS
from core.constants.environments import EnvironmentName
from core.constants.environments import EvalType
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatMessage
from core.models.pvp_models import ChatResult
from core.models.pvp_models import ChatRole
from core.models.pvp_models import PvPMatchupConfig
from core.models.pvp_models import ToolSchema
from core.pvp.tools import GAME_ACTION_TOOL_NAME
from ops.tools.evaluation.pvp_anthropic_match import anthropic_chat
from validator.evaluation.pvp.game_runner import Player
from validator.evaluation.pvp.game_runner import run_matchup


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("harness_capture")

_MOVE_TURN_GUIDANCE = (
    "IMPORTANT: You get exactly one response this turn and there is no follow-up "
    "message. In this one response you must call game_action with a legal action "
    "id. You may also call memory tools in the same response. If you only edit "
    "memory and omit game_action, you forfeit the turn."
)


def pvp_envs() -> list[EnvironmentName]:
    return [name for name, cfg in ENVIRONMENT_CONFIGS.items() if cfg.eval_type == EvalType.PVP]


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as file:
        return sum(1 for _ in file)


def _trim_jsonl(path: Path, n_lines: int) -> None:
    lines = path.read_text().splitlines()
    if len(lines) > n_lines:
        path.write_text("\n".join(lines[:n_lines]) + "\n")


def _stamp_ordering(path: Path) -> None:
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    arc = -1
    prev_matchup = object()
    turn_in_arc = 0
    for index, row in enumerate(rows):
        if row.get("matchup") != prev_matchup:
            arc += 1
            turn_in_arc = 0
            prev_matchup = row.get("matchup")
        row["seq"] = index
        row["arc"] = arc
        row["turn_in_arc"] = turn_in_arc
        turn_in_arc += 1
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


class SampleCollector:
    def __init__(self, env: str, model: str, file):
        self.env = env
        self.model = model
        self.file = file
        self.matchup = 0
        self.kept = 0
        self.skipped = 0
        self.long_term_writes = 0

    def wrap(self, chat_fn):
        @functools.wraps(chat_fn)
        def wrapped(config, messages, tools=None):
            result = chat_fn(config, messages, tools)
            self._record(messages, tools, result)
            return result

        return wrapped

    def _record(self, messages: list[ChatMessage], tools: list[ToolSchema] | None, result: ChatResult) -> None:
        tool_names = {tool.function.name for tool in (tools or [])}
        is_turn = GAME_ACTION_TOOL_NAME in tool_names
        calls = result.tool_calls or []

        if is_turn and not any(call.name == GAME_ACTION_TOOL_NAME for call in calls):
            self.skipped += 1
            return
        if not is_turn and not calls:
            self.skipped += 1
            return

        if any(call.name.startswith("long_term") for call in calls):
            self.long_term_writes += 1

        assistant = ChatMessage(role=ChatRole.ASSISTANT, content=result.content, tool_calls=result.tool_calls)
        sample = {
            "messages": [message.to_openai() for message in messages] + [assistant.to_openai()],
            "tools": [tool.to_openai() for tool in (tools or [])],
            "env": self.env,
            "model": self.model,
            "matchup": self.matchup,
            "turn_type": "turn" if is_turn else "reflection",
        }
        self.file.write(json.dumps(sample) + "\n")
        self.file.flush()
        self.kept += 1


def _inject_move_guidance(chat_fn):
    @functools.wraps(chat_fn)
    def wrapped(config, messages, tools=None):
        if any(tool.function.name == GAME_ACTION_TOOL_NAME for tool in (tools or [])):
            messages = list(messages)
            for index, message in enumerate(messages):
                if message.role == ChatRole.SYSTEM:
                    messages[index] = message.model_copy(update={"content": f"{message.content or ''}\n\n{_MOVE_TURN_GUIDANCE}"})
                    break
        return chat_fn(config, messages, tools)

    return wrapped


def _player(client: anthropic.Anthropic, model: str, collector: SampleCollector) -> Player:
    config = ChatCompletionConfig(inference_model=model, base_url="http://anthropic/v1")
    chat = collector.wrap(functools.partial(anthropic_chat, client))
    return Player(client=client, config=config, chat_fn=_inject_move_guidance(chat))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--model-b", default=None)
    parser.add_argument("--target-samples-per-env", type=int, default=None)
    parser.add_argument("--trim-to-target", action="store_true")
    parser.add_argument("--max-matchups-per-env", type=int, default=40)
    parser.add_argument("--matchups-per-env", type=int, default=2)
    parser.add_argument("--time-budget-seconds-per-matchup", type=float, default=900.0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--keep-forfeit-truncation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--output-dir", default="output/pvp_sft")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_a = args.model
    model_b = args.model_b or args.model
    envs = [EnvironmentName(env) for env in args.envs] if args.envs else pvp_envs()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    if not args.keep_forfeit_truncation:
        eval_cst.PVP_EPISODE_FORFEIT_THRESHOLD = 10**9
        eval_cst.PVP_CONSECUTIVE_LOSS_FORFEIT = 10**9
        logger.info("Matchup forfeit truncation disabled for data generation")

    target = args.target_samples_per_env
    started = time.time()
    totals: dict[str, int] = {}
    for env in envs:
        env_path = out_dir / f"{env.value}.jsonl"
        existing = _count_lines(env_path) if args.append else 0
        if args.append and target and existing >= target:
            logger.info("[%s] already has %d >= target %d", env.value, existing, target)
            totals[env.value] = existing
            continue

        mode = "a" if args.append else "w"
        with open(env_path, mode) as file:
            collector_a = SampleCollector(env.value, model_a, file)
            collector_b = SampleCollector(env.value, model_b, file)
            player_a = _player(client, model_a, collector_a)
            player_b = _player(client, model_b, collector_b)

            if target is not None:
                remaining_matchups = args.max_matchups_per_env
            else:
                remaining_matchups = args.matchups_per_env

            for matchup in range(remaining_matchups):
                current_count = _count_lines(env_path)
                if target is not None and current_count >= target:
                    break
                collector_a.matchup = existing + matchup
                collector_b.matchup = existing + matchup
                run_matchup(
                    env_name=env,
                    matchup_config=PvPMatchupConfig(time_budget_seconds=args.time_budget_seconds_per_matchup),
                    player_a=player_a,
                    player_b=player_b,
                    base_seed=args.seed + matchup,
                )

        if target is not None and args.trim_to_target:
            _trim_jsonl(env_path, target)
        _stamp_ordering(env_path)
        totals[env.value] = _count_lines(env_path)
        logger.info("[%s] wrote %d samples", env.value, totals[env.value])

    if args.upload:
        if not args.repo_id:
            raise ValueError("--upload requires --repo-id")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(out_dir))

    logger.info("Done in %.1fs: %s", time.time() - started, totals)


if __name__ == "__main__":
    main()
