#!/usr/bin/env python3
"""Generate SFT cold-start data by playing Claude vs Claude through the *real*
PvP tool-calling harness, capturing every model turn as an OpenAI messages+tools
sample.

Why this and not the old plain-text generator: games are now played as a
tool-calling memory loop (the model commits its move via the game_action tool
and edits memory slots via memory tools), so cold-start data must be in that
exact format. We reuse run_matchup + the Anthropic chat adapter, wrap each
player's chat_fn, and dump every (system+user -> assistant tool_calls) exchange.
Per-turn context is rebuilt fresh by the harness, so each captured turn is a
self-contained training example.

Long-term memory: run_matchup carries one long-term SlotMemory per player across
all games of a matchup (consolidated by a reflection turn after each game), so we
run several independent matchups per env — each a fresh long-term arc — and tag
samples with the matchup index.

  ANTHROPIC_API_KEY must be set.
  python -m scripts.sft_dataset_gen.harness_capture \
      --model claude-sonnet-4-6 --model-b claude-haiku-4-5 \
      --matchups-per-env 2 --games-per-matchup 25 --output-dir output/pvp_sft \
      --repo-id <namespace>/<dataset> --upload
"""

import argparse
import functools
import json
import logging
import sys
import time
from pathlib import Path

import anthropic

from core.constants import ENVIRONMENT_CONFIGS
from core.constants import EnvironmentName
from core.constants import EvalType
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatMessage
from core.models.pvp_models import ChatResult
from core.models.pvp_models import ChatRole
from core.models.pvp_models import PvPMatchupConfig
from core.models.pvp_models import ToolSchema
from core.pvp.tools import GAME_ACTION_TOOL_NAME
from validator.core import constants as vcst
from validator.evaluation.pvp.game_runner import Player
from validator.evaluation.pvp.game_runner import run_matchup
from scripts.pvp_anthropic_match import anthropic_chat


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger("harness_capture")

# Data-gen-only guidance appended to the system prompt on move turns. Claude does
# sequential tool use (edits memory, then waits for a result) so under the
# harness's single-call contract it omits game_action and forfeits. Spelling out
# "one response, no follow-up, game_action required now, parallel calls allowed"
# plus a worked example makes it co-emit reliably. Only added when a game_action
# tool is offered (i.e. a move turn, not a reflection turn).
_MOVE_TURN_GUIDANCE = (
    "IMPORTANT — single-response turn: You get EXACTLY ONE response this turn and "
    "there is NO follow-up message, so you cannot act in a later step. In this one "
    "response you MUST call game_action with a legal action id to commit your move. "
    "You MAY also call memory tools in the SAME response — parallel tool calls are "
    "supported and encouraged. If you only edit memory and omit game_action, you "
    "forfeit the turn.\n"
    "Example of one correct response (two tool calls together): "
    "working_memory_rewrite(slot=1, content=\"opponent opened aggressively; conserve high cards\") "
    "AND game_action(action_id=3)."
)


def pvp_envs() -> list[EnvironmentName]:
    """Every model-vs-model env (excludes intercode, which is not PvP)."""
    return [name for name, cfg in ENVIRONMENT_CONFIGS.items() if cfg.eval_type == EvalType.PVP]


def _count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def _trim_jsonl(path: Path, n: int) -> None:
    """Keep only the first n lines, so every env ends with the same count."""
    lines = path.read_text().splitlines()
    if len(lines) > n:
        path.write_text("\n".join(lines[:n]) + "\n")


def _stamp_ordering(path: Path) -> None:
    """Add seq / arc / turn_in_arc so order survives shuffling and reload.

    File line order is the chronological play order (each turn is written as it
    happens). seq is that order; arc is a unique long-term-memory arc id (it
    increments whenever `matchup` changes, so append runs that restart matchup at
    0 get distinct arcs); turn_in_arc orders turns within an arc.
    """
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    arc, prev_matchup, turn_in_arc = -1, object(), 0
    for i, r in enumerate(rows):
        if r.get("matchup") != prev_matchup:
            arc += 1
            turn_in_arc = 0
            prev_matchup = r.get("matchup")
        r["seq"], r["arc"], r["turn_in_arc"] = i, arc, turn_in_arc
        turn_in_arc += 1
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class SampleCollector:
    """Wraps a ChatFn to dump each call as an SFT sample.

    A sample is the exact prompt the harness built (system + user messages) plus
    the assistant response (text + tool_calls) and the tool schemas — i.e. what a
    tokenizer.apply_chat_template(messages, tools=tools) would consume. Quality
    gate: turn samples are kept only when a legal-looking game_action was emitted
    (no forfeits as training targets); reflection samples need >=1 memory call.
    """

    def __init__(self, env: str, model: str, fh):
        self.env = env
        self.model = model
        self.fh = fh
        self.matchup = 0  # set per matchup so long-term arcs are distinguishable
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
        tool_names = {t.function.name for t in (tools or [])}
        is_turn = GAME_ACTION_TOOL_NAME in tool_names
        calls = result.tool_calls or []

        if is_turn and not any(c.name == GAME_ACTION_TOOL_NAME for c in calls):
            self.skipped += 1  # forfeit turn — bad training target
            return
        if not is_turn and not calls:
            self.skipped += 1  # empty reflection — nothing learned
            return

        if any(c.name.startswith("long_term") for c in calls):
            self.long_term_writes += 1

        assistant = ChatMessage(role=ChatRole.ASSISTANT, content=result.content, tool_calls=result.tool_calls)
        sample = {
            "messages": [m.to_openai() for m in messages] + [assistant.to_openai()],
            "tools": [t.to_openai() for t in (tools or [])],
            "env": self.env,
            "model": self.model,
            "matchup": self.matchup,
            "turn_type": "turn" if is_turn else "reflection",
        }
        self.fh.write(json.dumps(sample) + "\n")
        self.fh.flush()
        self.kept += 1


def _inject_move_guidance(chat_fn):
    """On move turns (game_action offered), append the strong single-response
    guidance to the system message before recording/sending. Applied outside the
    collector so the guidance is part of the captured sample (data-gen only)."""
    @functools.wraps(chat_fn)
    def wrapped(config, messages, tools=None):
        if any(t.function.name == GAME_ACTION_TOOL_NAME for t in (tools or [])):
            messages = list(messages)
            for i, m in enumerate(messages):
                if m.role == ChatRole.SYSTEM:
                    messages[i] = m.model_copy(update={"content": f"{m.content or ''}\n\n{_MOVE_TURN_GUIDANCE}"})
                    break
        return chat_fn(config, messages, tools)
    return wrapped


def _player(client: anthropic.Anthropic, model: str, collector: SampleCollector) -> Player:
    config = ChatCompletionConfig(inference_model=model, base_url="http://anthropic/v1")
    chat = collector.wrap(functools.partial(anthropic_chat, client))
    chat = _inject_move_guidance(chat)
    return Player(client=client, config=config, chat_fn=chat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Claude model for both players (self-play)")
    parser.add_argument("--model-b", default=None, help="Optional different model for player B (defaults to --model)")
    parser.add_argument("--target-samples-per-env", type=int, default=None,
                        help="Generate matchups until each env reaches AT LEAST this many kept samples (existing + new), "
                             "then stop. Overshoot is kept (never discarded). Overrides --matchups-per-env.")
    parser.add_argument("--trim-to-target", action="store_true",
                        help="Also trim each env down to exactly --target-samples-per-env. OFF by default — trimming "
                             "throws away already-generated (paid) samples.")
    parser.add_argument("--max-matchups-per-env", type=int, default=40,
                        help="Safety cap on matchups when chasing --target-samples-per-env")
    parser.add_argument("--matchups-per-env", type=int, default=2,
                        help="Independent matchups per env; each carries its own long-term memory arc")
    parser.add_argument("--games-per-matchup", type=int, default=25,
                        help="Seeds per matchup; each played twice (position swap). long-term memory continues across them")
    parser.add_argument("--append", action="store_true",
                        help="Keep existing per-env files and only ADD samples (top up to --target-samples-per-env); "
                             "skip envs already at target. Default overwrites each env file.")
    parser.add_argument("--keep-forfeit-truncation", action="store_true",
                        help="Keep the eval-time matchup forfeit/early-stop truncation (off by default for data-gen, "
                             "so all games play out and slow envs still yield data)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--envs", nargs="+", default=None, help="Env subset (default: all PvP envs)")
    parser.add_argument("--output-dir", default="output/pvp_sft")
    parser.add_argument("--repo-id", default=None, help="HF dataset repo id to upload to, e.g. ns/name")
    parser.add_argument("--private", action="store_true", help="Create the HF dataset as private")
    parser.add_argument("--upload", action="store_true", help="Upload the output dir to --repo-id after generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_a = args.model
    model_b = args.model_b or args.model
    envs = [EnvironmentName(e) for e in args.envs] if args.envs else pvp_envs()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    # Data-gen wants every game played: the eval-time matchup truncation (award
    # remaining games once a model forfeits enough / loses a streak) starves slow,
    # forfeit-prone envs like othello. Disable it unless explicitly kept.
    if not args.keep_forfeit_truncation:
        vcst.PVP_EPISODE_FORFEIT_THRESHOLD = 10**9
        vcst.PVP_CONSECUTIVE_LOSS_FORFEIT = 10**9
        logger.info("Matchup forfeit-truncation disabled for data-gen (all games play out)")

    target = args.target_samples_per_env
    start = time.time()
    totals: dict[str, int] = {}
    for env in envs:
        env_path = out_dir / f"{env.value}.jsonl"
        existing = _count_lines(env_path) if (args.append and env_path.exists()) else 0
        if args.append and target and existing >= target:
            logger.info("[%s] already has %d >= target %d — leaving untouched", env.value, existing, target)
            totals[env.value] = existing
            continue

        # Append keeps what's already there and only adds the deficit; otherwise
        # the file is rewritten from scratch.
        with open(env_path, "a" if args.append else "w") as fh:
            # One collector per player so each sample records the model that
            # authored it (and we see per-model yield). existing samples count
            # toward the target so append only generates the shortfall.
            coll_a = SampleCollector(env.value, model_a, fh)
            coll_b = SampleCollector(env.value, model_b, fh)
            player_a = _player(client, model_a, coll_a)
            player_b = _player(client, model_b, coll_b)
            # Each matchup is a fresh long-term-memory arc (run_matchup makes new
            # long-term SlotMemory per call); a distinct seed makes the arcs differ.
            # In target mode, keep adding matchups until the env reaches the target.
            # In append mode, offset the seed so new arcs differ from the originals.
            max_matchups = args.max_matchups_per_env if target else args.matchups_per_env
            seed_base = args.seed + (5_000_000 if args.append else 0)
            for m in range(max_matchups):
                coll_a.matchup = coll_b.matchup = m
                total_so_far = existing + coll_a.kept + coll_b.kept
                logger.info("[%s] matchup %d — %s vs %s, %d games (x2 swap) | have %d (existing %d + new %d)",
                            env.value, m + 1, model_a, model_b, args.games_per_matchup,
                            total_so_far, existing, coll_a.kept + coll_b.kept)
                result = run_matchup(
                    env_name=env,
                    matchup_config=PvPMatchupConfig(num_games=args.games_per_matchup),
                    player_a=player_a,
                    player_b=player_b,
                    base_seed=seed_base + m * 100_000,
                )
                logger.info("[%s] matchup %d done | games a=%d b=%d draws=%d",
                            env.value, m + 1, result.model_a_wins, result.model_b_wins, result.draws)
                if target and existing + coll_a.kept + coll_b.kept >= target:
                    break
            total = existing + coll_a.kept + coll_b.kept
            logger.info("[%s] total=%d (existing %d + new %d: %s=%d, %s=%d) skipped=%d long_term_writes=%d",
                        env.value, total, existing, coll_a.kept + coll_b.kept,
                        model_a, coll_a.kept, model_b, coll_b.kept,
                        coll_a.skipped + coll_b.skipped, coll_a.long_term_writes + coll_b.long_term_writes)
            if target and total < target:
                logger.warning("[%s] only %d/%d after %d matchups (cap hit)", env.value, total, target, max_matchups)

        # Only trim when explicitly asked — trimming discards already-paid samples.
        if target and args.trim_to_target:
            _trim_jsonl(env_path, target)
        _stamp_ordering(env_path)  # seq / arc / turn_in_arc, robust to later shuffling
        totals[env.value] = _count_lines(env_path)

    grand = sum(totals.values())
    logger.info("=== Done in %.1f min — %d samples across %d envs: %s ===",
                (time.time() - start) / 60, grand, len(envs), totals)

    if args.upload:
        if not args.repo_id:
            raise SystemExit("--upload requires --repo-id")
        _upload(out_dir, args.repo_id, args.private, model_a, model_b)


def _write_readme(out_dir: Path, model_a: str, model_b: str) -> None:
    """Describe the dataset from the actual files on disk (every env, not just the
    envs touched by the last run — a partial run must not shrink the description)."""
    counts = {p.stem: _count_lines(p) for p in sorted(out_dir.glob("*.jsonl"))}
    total = sum(counts.values())
    (out_dir / "README.md").write_text(
        "# PvP tool-calling SFT cold-start data\n\n"
        "Claude-vs-Claude games played through the G.O.D PvP tool-calling harness. "
        "Each row is one model turn (or post-game reflection): the system+user prompt "
        "the harness built, the assistant response (`content` + `tool_calls`), and the "
        "`tools` schemas — i.e. the OpenAI messages+tools format consumed by "
        "`tokenizer.apply_chat_template(messages, tools=tools)`. On a move turn the "
        "assistant co-emits any memory-tool edits and a `game_action` committing a legal "
        "move; reflection rows consolidate long-term memory after a game.\n\n"
        f"Players: {model_a} vs {model_b} (positions swapped each seed, so both orderings appear).\n\n"
        f"Total: {total} samples across {len(counts)} environments.\n\n"
        "## Per-env sample counts\n\n"
        + "\n".join(f"- `{env}`: {n}" for env, n in counts.items())
        + "\n\n## Fields\n"
        "- `messages`, `tools`: the training example (OpenAI chat + tools format)\n"
        "- `env`: game name\n"
        "- `model`: the Claude model that authored the sample\n"
        "- `turn_type`: `turn` (commits `game_action`) or `reflection` (consolidates long-term memory)\n"
        "- `seq`: chronological play order within the env file (stable across shuffling)\n"
        "- `arc`: unique long-term-memory arc id — memory persists across the games within one arc\n"
        "- `turn_in_arc`: order of the sample within its arc\n"
        "- `matchup`: per-generation-run arc index (legacy; use `arc` for a globally-unique id)\n\n"
        "To reconstruct a long-term-memory trajectory, take one `(env, arc)` group and sort by "
        "`turn_in_arc` (or `seq`). For plain SFT this is unnecessary — each row is self-contained, "
        "since the memory state at that turn is already rendered inside its prompt.\n"
    )


def _upload(out_dir: Path, repo_id: str, private: bool, model_a: str, model_b: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    _write_readme(out_dir, model_a, model_b)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset")
    logger.info("Uploaded %s -> https://huggingface.co/datasets/%s", out_dir, repo_id)


if __name__ == "__main__":
    main()
