import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


class EnvClient:
    def __init__(self, base_url: str, timeout: float = 120.0, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def reset(
        self,
        task_id: int,
        seed: int,
        opponent: str,
        mcts_max_simulations: int,
        mcts_num_rollouts: int,
    ) -> tuple[str, str]:
        payload = {
            "task_id": task_id,
            "seed": seed,
            "opponent": opponent,
            "mcts_max_simulations": mcts_max_simulations,
            "mcts_num_rollouts": mcts_num_rollouts,
        }
        data = await self._post("/reset", payload)
        result = data["result"]
        return result["episode_id"], result.get("observation", "")

    async def step(self, action: str, episode_id: str) -> tuple[str, float, bool]:
        payload = {"action": action, "episode_id": episode_id}
        data = await self._post("/step", payload)
        result = data["result"]
        return (
            result.get("observation", ""),
            result.get("reward", 0.0),
            result.get("done", False),
        )

    async def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self.client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Env %s attempt %d/%d failed: %s — retrying in %ds",
                        path, attempt, self.max_retries, exc, wait,
                    )
                    await asyncio.sleep(wait)
        raise RuntimeError(f"Env {path} failed after {self.max_retries} attempts") from last_exc

    async def close(self) -> None:
        await self.client.aclose()
