"""Individual OpenSpiel evaluation container entry point.

Runs one model against a single-player OpenSpiel environment and writes the
standard {repo: {"is_finetune": True, "eval_loss": avg_score}} result payload.
"""

import functools
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from core import constants as core_cst
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import PvPModelSpec
from core.pvp.chat import chat_completion
from core.pvp.chat import create_client
from core.pvp.individual import run_individual_open_spiel_eval
from validator.core import constants as vcst
from validator.evaluation.eval_environment import _wait_for_health
from validator.evaluation.pvp.__main__ import _prepare_model
from validator.evaluation.pvp.server import start_sglang
from validator.evaluation.utils import configure_eval_logging
from validator.evaluation.utils import stop_process


logger = logging.getLogger(__name__)


def main() -> int:
    configure_eval_logging()
    try:
        _run()
        return 0
    except Exception as exc:
        logger.exception("Individual OpenSpiel evaluation failed: %s", exc)
        return 1


def _single_model_repo() -> str:
    models_raw = os.getenv("MODELS", "")
    repos = [repo.strip() for repo in models_raw.split(",") if repo.strip()]
    if len(repos) != 1:
        raise ValueError("MODELS is required and must contain a single repo")
    return repos[0]


def _base_chain() -> list[str]:
    raw = os.getenv("BASE_CHAIN")
    if not raw:
        return []
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("BASE_CHAIN must be a JSON list of repo strings")
    return value


def _chat_config(prepared_model_path: str, inference_name: str, seed: int, temperature: float) -> ChatCompletionConfig:
    return ChatCompletionConfig(
        inference_model=inference_name,
        tokenizer_repo=prepared_model_path,
        base_url=f"http://{vcst.PVP_SGLANG_HOST}:{vcst.PVP_SGLANG_PORT_A}{vcst.PVP_SGLANG_API_PATH}",
        temperature=temperature,
        seed=seed,
        read_timeout=vcst.PVP_HTTP_READ_TIMEOUT_SECONDS,
        max_retries=vcst.PVP_HTTP_MAX_RETRIES,
    )


def _run() -> None:
    model_repo = _single_model_repo()
    original_model = os.environ["ORIGINAL_MODEL"]
    env_name = core_cst.EnvironmentName(os.environ["ENVIRONMENT_NAME"])
    env_config = core_cst.ENVIRONMENT_CONFIGS[env_name]
    if env_config.eval_type != core_cst.EvalType.INDIVIDUAL:
        raise ValueError(f"{env_name.value} is not configured as an INDIVIDUAL environment")

    seed = int(os.getenv("EVAL_SEED", str(vcst.ENV_EVAL_DEFAULT_SEED)))
    temperature = float(os.getenv("ENV_EVAL_TEMPERATURE", str(vcst.ENV_EVAL_TEMPERATURE)))
    num_games = int(
        os.getenv("INDIVIDUAL_OPEN_SPIEL_NUM_GAMES_PER_ENV", str(vcst.INDIVIDUAL_OPEN_SPIEL_NUM_GAMES_PER_ENV))
    )
    eval_timeout_seconds = float(
        os.getenv("INDIVIDUAL_OPEN_SPIEL_EVAL_TIMEOUT_SECONDS", str(vcst.INDIVIDUAL_OPEN_SPIEL_EVAL_TIMEOUT_SECONDS))
    )
    episode_timeout_seconds = float(
        os.getenv(
            "INDIVIDUAL_OPEN_SPIEL_EPISODE_TIMEOUT_SECONDS",
            str(vcst.INDIVIDUAL_OPEN_SPIEL_EPISODE_TIMEOUT_SECONDS),
        )
    )
    max_player_actions = int(
        os.getenv(
            "INDIVIDUAL_OPEN_SPIEL_MAX_PLAYER_ACTIONS_PER_EPISODE",
            str(vcst.INDIVIDUAL_OPEN_SPIEL_MAX_PLAYER_ACTIONS_PER_EPISODE),
        )
    )

    prepared = _prepare_model(
        PvPModelSpec(repo=model_repo, original_model=original_model, base_chain=_base_chain()),
        label="individual",
        gpu_id=0,
    )

    sglang_proc: subprocess.Popen | None = None
    client = None
    started = time.time()

    try:
        sglang_proc = start_sglang(prepared, gpu_id=0, port=vcst.PVP_SGLANG_PORT_A, seed=seed)
        import asyncio

        asyncio.run(
            _wait_for_health(
                f"http://{vcst.PVP_SGLANG_HOST}:{vcst.PVP_SGLANG_PORT_A}",
                vcst.PVP_SGLANG_HEALTH_PATH,
                vcst.PVP_SGLANG_HEALTH_TIMEOUT,
                service_name="sglang-individual",
            )
        )

        config = _chat_config(prepared.sglang_model_path, prepared.inference_name, seed, temperature)
        client = create_client(config)
        chat_fn = functools.partial(chat_completion, client)
        result = run_individual_open_spiel_eval(
            env_name=env_name,
            chat_fn=chat_fn,
            config=config,
            num_games=num_games,
            base_seed=seed,
            time_budget_seconds=eval_timeout_seconds,
            episode_timeout_seconds=episode_timeout_seconds,
            max_player_actions_per_episode=max_player_actions,
        )

        output = {model_repo: {"is_finetune": True, core_cst.CONTAINER_EVAL_SCORE_KEY: result.mean_score}}
        result_path = Path(core_cst.CONTAINER_EVAL_RESULTS_PATH)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(output), encoding="utf-8")
        logger.info(
            "%s individual eval wrote %s games=%d avg=%.6f elapsed=%.1fs",
            env_name.value, result_path, result.num_games, result.mean_score, time.time() - started,
        )
    finally:
        if client is not None:
            client.close()
        stop_process(sglang_proc, "sglang-individual")


if __name__ == "__main__":
    sys.exit(main())
