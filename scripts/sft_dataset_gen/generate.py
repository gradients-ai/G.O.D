import asyncio
import logging
import random
import re
import sys
import time
from pathlib import Path

from .claude_client import ClaudePlayer
from .config import FORMAT_INSTRUCTIONS, GAME_CONFIGS, MAX_TURNS, CONTEXT_WINDOW, GameConfig, parse_args
from .dataset_formatter import append_episode, count_completed, print_sample_episode
from .env_client import EnvClient
from .game_prompts import SYSTEM_PROMPTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


def parse_action(text: str) -> str:
    if "Action:" in text:
        action_part = text.split("Action:")[-1].strip()
        match = re.search(r"\d+", action_part)
        if match:
            return match.group(0)
        return action_part
    match = re.search(r"\d+", text)
    if match:
        return match.group(0)
    return text.strip()


async def run_episode(
    env: EnvClient,
    player: ClaudePlayer,
    game_name: str,
    game_config: GameConfig,
    episode_idx: int,
) -> dict | None:
    task_id = random.randint(*game_config.task_id_range)
    seed = task_id
    mcts_sims = random.randint(*game_config.mcts_sim_range)
    system_prompt = SYSTEM_PROMPTS[game_name]
    model = player.pick_model()

    try:
        episode_id, observation = await env.reset(
            task_id=task_id,
            seed=seed,
            opponent=game_config.opponent,
            mcts_max_simulations=mcts_sims,
            mcts_num_rollouts=game_config.mcts_num_rollouts,
        )
    except Exception as exc:
        logger.error("[%s #%d] Reset failed: %s", game_name, episode_idx, exc)
        return None

    first_observation = observation + "\n\n" + FORMAT_INSTRUCTIONS
    messages: list[dict[str, str]] = [{"role": "user", "content": first_observation}]
    conversations: list[dict[str, str]] = [
        {"from": "system", "value": system_prompt},
        {"from": "user", "value": first_observation},
    ]

    done = False
    reward = 0.0
    turn = 0

    while not done and turn < MAX_TURNS:
        try:
            # Send only recent context to Claude — observations contain full game state
            windowed = messages[-CONTEXT_WINDOW:] if len(messages) > CONTEXT_WINDOW else messages
            # Ensure first message is always a user message
            if windowed and windowed[0]["role"] == "assistant":
                windowed = windowed[1:]
            response_text, _ = await player.get_action(
                system_prompt, windowed, model_override=model,
            )
        except Exception as exc:
            logger.error("[%s #%d] Claude call failed turn %d: %s", game_name, episode_idx, turn, exc)
            return None

        action = parse_action(response_text)
        conversations.append({"from": "assistant", "value": response_text})
        messages.append({"role": "assistant", "content": response_text})

        try:
            observation, step_reward, done = await env.step(action, episode_id)
        except Exception as exc:
            logger.error("[%s #%d] Step failed turn %d: %s", game_name, episode_idx, turn, exc)
            return None

        if done:
            reward = step_reward
        else:
            messages.append({"role": "user", "content": observation})
            conversations.append({"from": "user", "value": observation})

        turn += 1

    outcome = "win" if reward > 0 else ("loss" if reward < 0 else "draw")

    return {
        "conversations": conversations,
        "game": game_name,
        "reward": reward,
        "outcome": outcome,
        "model": model,
        "mcts_sims": mcts_sims,
        "seed": seed,
        "task_id": task_id,
        "num_turns": turn,
    }


async def run_game_episodes(
    env: EnvClient,
    player: ClaudePlayer,
    game_name: str,
    game_config: GameConfig,
    num_episodes: int,
    semaphore: asyncio.Semaphore,
    checkpoint_path: Path,
    start_idx: int = 0,
) -> dict[str, int]:
    stats = {"completed": 0, "failed": 0, "wins": 0}

    async def run_one(idx: int) -> None:
        async with semaphore:
            episode = await run_episode(env, player, game_name, game_config, idx)
            if episode is None:
                stats["failed"] += 1
                logger.warning("[%s #%d] Episode failed", game_name, idx)
                return
            append_episode(checkpoint_path, episode)
            stats["completed"] += 1
            if episode["reward"] > 0:
                stats["wins"] += 1
            total = stats["completed"] + stats["failed"]
            outcome = episode["outcome"]
            logger.info(
                "[%s #%d] mcts=%d model=%s %s reward=%.3f turns=%d | %d/%d done, cost=$%.2f",
                game_name, idx, episode["mcts_sims"],
                episode["model"].split("-")[1],  # haiku/sonnet
                outcome, episode["reward"], episode["num_turns"],
                total, num_episodes, player.usage.cost_usd,
            )

    tasks = [run_one(start_idx + i) for i in range(num_episodes)]
    await asyncio.gather(*tasks)
    return stats


async def main() -> None:
    args = parse_args()

    if args.smoke_test:
        for gc in GAME_CONFIGS.values():
            object.__setattr__(gc, "num_episodes", 1)
        logger.info("Smoke test mode: 1 episode per game")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.jsonl"

    completed = count_completed(checkpoint_path)
    if any(completed[g] > 0 for g in args.games):
        logger.info("Resuming from checkpoint: %s", dict(completed))

    env = EnvClient(args.env_url)
    player = ClaudePlayer(
        sonnet_model=args.model,
        haiku_ratio=args.haiku_ratio,
        temperature=args.temperature,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    start_time = time.time()
    all_stats: dict[str, dict[str, int]] = {}

    for game_name in args.games:
        if game_name not in GAME_CONFIGS:
            logger.error("Unknown game: %s", game_name)
            continue

        already_done = completed[game_name]
        remaining = GAME_CONFIGS[game_name].num_episodes - already_done
        if remaining <= 0:
            logger.info("[%s] Already completed %d episodes, skipping", game_name, already_done)
            continue

        logger.info(
            "[%s] Generating %d episodes (already done: %d)",
            game_name, remaining, already_done,
        )
        stats = await run_game_episodes(
            env=env,
            player=player,
            game_name=game_name,
            game_config=GAME_CONFIGS[game_name],
            num_episodes=remaining,
            semaphore=semaphore,
            checkpoint_path=checkpoint_path,
            start_idx=already_done,
        )
        all_stats[game_name] = stats

    await env.close()

    elapsed = time.time() - start_time
    logger.info("=== Generation Complete ===")
    logger.info("Time: %.1f minutes", elapsed / 60)
    logger.info("Total cost: $%.2f", player.usage.cost_usd)
    logger.info(
        "Tokens — input: %d, output: %d",
        player.usage.input_tokens, player.usage.output_tokens,
    )
    for game_name, stats in all_stats.items():
        total = stats["completed"] + stats["failed"]
        logger.info(
            "  %s: %d/%d completed, %d wins (%.1f%%)",
            game_name, stats["completed"], total,
            stats["wins"],
            (stats["wins"] / stats["completed"] * 100) if stats["completed"] else 0,
        )

    for game_name in args.games:
        print_sample_episode(checkpoint_path, game_name)


def cli_entry() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli_entry()
