import httpx

from core.logging import get_logger


logger = get_logger(__name__)


async def send_to_discord(webhook: str, message: str):
    async with httpx.AsyncClient() as client:
        payload = {"content": message}
        response = await client.post(webhook, json=payload)
        return response

async def notify_tournament_started(tournament_id: str, tournament_type: str, participants: int, discord_url: str):
    try:
        message = (
            f"Tournament Started!\nTournament ID: {tournament_id}\nType: {tournament_type}\n"
            f"Participants: {participants}\nStatus: ACTIVE"
        )
        await send_to_discord(discord_url, message)
    except Exception as e:
        logger.error(f"Failed to send Discord notification for tournament start: {e}")

async def notify_tournament_completed(
    tournament_id: str, tournament_type: str, winner: str, discord_url: str, diff_report: str | None = None
):
    try:
        message = (
            f"Tournament Completed!\nTournament ID: {tournament_id}\nType: {tournament_type}\nWinner: {winner}\nStatus: COMPLETED"
        )
        if diff_report:
            message += f"\nDiff Report: {diff_report}"
        await send_to_discord(discord_url, message)
    except Exception as e:
        logger.error(f"Failed to send Discord notification for tournament completion: {e}")


async def notify_tournament_dedup_autoremoved(
    tournament_id: str, tournament_type: str, clusters, flagged: list[str], discord_url: str
):
    """R1 info ping: deterministic hash duplicates auto-removed before training."""
    try:
        lines = [
            "Dedup (R1, pre-training): auto-removed exact/normalized duplicate submissions.",
            f"Tournament: {tournament_id} ({tournament_type})",
            f"Removed {len(flagged)} duplicate(s) across {len(clusters)} cluster(s); boss protected.",
        ]
        for index, cluster in enumerate(clusters, 1):
            lines.append(f"  Cluster {index} [{cluster.basis}]: {', '.join(cluster.members)}")
        await send_to_discord(discord_url, "\n".join(lines))
    except Exception as e:
        logger.error(f"Failed to send Discord notification for R1 dedup: {e}")


async def notify_tournament_dedup_review(
    tournament_id: str,
    tournament_type: str,
    round_id: str,
    clusters,
    flagged: list[str],
    report_url: str | None,
    discord_url: str,
):
    """R2 ping: duplicates flagged by pairwise review; tournament halted pending approval."""
    try:
        lines = [
            "Dedup (R2): functional duplicates flagged; tournament halted pending manual review.",
            f"Tournament: {tournament_id} ({tournament_type})",
            f"Guarded round: {round_id}",
            f"Flagged for removal ({len(flagged)}): {', '.join(flagged)}",
        ]
        for index, cluster in enumerate(clusters, 1):
            lines.append(f"  Cluster {index} [{cluster.basis}]: {', '.join(cluster.members)} - {cluster.reason}")
        if report_url:
            lines.append(f"Full report: {report_url}")
        lines.append("")
        lines.append("Review the report and code, then in the DB:")
        lines.append(
            "  approve: UPDATE tournament_dedup_reviews "
            f"SET status='approved', reviewed_at=now() WHERE round_id='{round_id}';"
        )
        lines.append("  edit first if needed: SET approved_eliminations='[\"hk\", ...]'::jsonb")
        lines.append(
            "  skip: UPDATE tournament_dedup_reviews "
            f"SET status='skipped', reviewed_at=now() WHERE round_id='{round_id}';"
        )
        await send_to_discord(discord_url, "\n".join(lines))
    except Exception as e:
        logger.error(f"Failed to send Discord notification for R2 dedup review: {e}")


async def notify_tournament_dedup_error(
    tournament_id: str, tournament_type: str, round_id: str, error: str, discord_url: str
):
    """R2 ping: the dedup gate failed to evaluate; tournament halted."""
    try:
        lines = [
            "Dedup (R2): gate failed to evaluate; tournament halted with no eliminations applied.",
            f"Tournament: {tournament_id} ({tournament_type})",
            f"Guarded round: {round_id}",
            f"Error: {error}",
            "",
            "Held without re-running until you act.",
            "Fix the cause and restart the validator, or insert a skip row to advance without eliminations:",
            (
                f"  INSERT INTO tournament_dedup_reviews (round_id, tournament_id, tournament_type, status) "
                f"VALUES ('{round_id}', '{tournament_id}', '{tournament_type}', 'skipped');"
            ),
        ]
        await send_to_discord(discord_url, "\n".join(lines))
    except Exception as e:
        logger.error(f"Failed to send Discord notification for R2 dedup gate error: {e}")


async def notify_tournament_dedup_resolved(
    tournament_id: str, tournament_type: str, eliminated: list[str], published, discord_url: str
):
    """R2 ping: review approved; duplicates eliminated and published."""
    try:
        lines = [
            "Dedup (R2): review approved; duplicates knocked out and tournament resuming.",
            f"Tournament: {tournament_id} ({tournament_type})",
            f"Eliminated ({len(eliminated)}): {', '.join(eliminated) if eliminated else 'none'}",
        ]
        for published_repo in published:
            lines.append(f"  Published: {published_repo.public_repo_url}")
        await send_to_discord(discord_url, "\n".join(lines))
    except Exception as e:
        logger.error(f"Failed to send Discord notification for R2 dedup resolution: {e}")

async def notify_organic_task_created(task_id: str, task_type: str, discord_url: str, is_benchmark: bool = False):
    try:
        if is_benchmark:
            message = f"New Benchmark Task Created!\nTask ID: {task_id}\nType: {task_type}"
        else:
            message = f"New Organic Task Created!\nTask ID: {task_id}\nType: {task_type}"
        await send_to_discord(discord_url, message)
    except Exception as e:
        logger.error(f"Failed to send Discord notification for task creation: {e}")
