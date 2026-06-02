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

async def notify_organic_task_created(task_id: str, task_type: str, discord_url: str, is_benchmark: bool = False):
    try:
        if is_benchmark:
            message = f"New Benchmark Task Created!\nTask ID: {task_id}\nType: {task_type}"
        else:
            message = f"New Organic Task Created!\nTask ID: {task_id}\nType: {task_type}"
        await send_to_discord(discord_url, message)
    except Exception as e:
        logger.error(f"Failed to send Discord notification for task creation: {e}")
