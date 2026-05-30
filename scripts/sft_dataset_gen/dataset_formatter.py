import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path) -> list[dict]:
    episodes = []
    if not checkpoint_path.exists():
        return episodes
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def count_completed(checkpoint_path: Path) -> Counter:
    counts: Counter = Counter()
    if not checkpoint_path.exists():
        return counts
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ep = json.loads(line)
                counts[ep["game"]] += 1
    return counts


def append_episode(checkpoint_path: Path, episode: dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(episode, ensure_ascii=False) + "\n")


def _strip_system_messages(episodes: list[dict]) -> list[dict]:
    for ep in episodes:
        ep["conversations"] = [
            msg for msg in ep["conversations"] if msg.get("from") != "system"
        ]
    return episodes


def format_and_validate(checkpoint_path: Path, output_path: Path) -> None:
    episodes = load_checkpoint(checkpoint_path)
    if not episodes:
        logger.error("No episodes found in checkpoint")
        return

    episodes = _strip_system_messages(episodes)

    game_counts: Counter = Counter()
    game_rewards: dict[str, list[float]] = {}
    game_turns: dict[str, list[int]] = {}

    for ep in episodes:
        game = ep["game"]
        game_counts[game] += 1
        game_rewards.setdefault(game, []).append(ep["reward"])
        game_turns.setdefault(game, []).append(ep["num_turns"])

    logger.info("=== Dataset Summary ===")
    logger.info("Total episodes: %d", len(episodes))
    for game in sorted(game_counts.keys()):
        rewards = game_rewards[game]
        turns = game_turns[game]
        win_rate = sum(1 for r in rewards if r > 0) / len(rewards)
        avg_reward = sum(rewards) / len(rewards)
        avg_turns = sum(turns) / len(turns)
        logger.info(
            "  %s: %d episodes, win_rate=%.1f%%, avg_reward=%.3f, avg_turns=%.1f",
            game, game_counts[game], win_rate * 100, avg_reward, avg_turns,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    logger.info("Dataset written to %s", output_path)


def print_sample_episode(checkpoint_path: Path, game: str | None = None) -> None:
    episodes = load_checkpoint(checkpoint_path)
    if game:
        episodes = [e for e in episodes if e["game"] == game]
    if not episodes:
        logger.info("No episodes to sample")
        return

    ep = episodes[0]
    print(f"\n{'='*60}")
    print(f"Game: {ep['game']} | Reward: {ep['reward']} | Turns: {ep['num_turns']}")
    print(f"Task ID: {ep['task_id']} | Seed: {ep['seed']}")
    print(f"{'='*60}")
    for msg in ep["conversations"]:
        role = msg["from"]
        value = msg["value"]
        if role == "system":
            print(f"\n[SYSTEM] (length={len(value)} chars)")
        elif role == "user":
            preview = value[:300] + "..." if len(value) > 300 else value
            print(f"\n[USER]\n{preview}")
        else:
            print(f"\n[ASSISTANT]\n{value}")
    print(f"\n{'='*60}\n")
