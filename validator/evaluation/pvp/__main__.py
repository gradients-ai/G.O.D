"""PvP evaluation container entry point.

Loads config, starts two SGLang instances (one per GPU),
runs all matchups, writes results JSON.

Usage: python -m validator.evaluation.pvp
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from core.constants import EnvironmentName
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import PreparedModel
from core.models.pvp_models import PvPEnvironmentResult
from core.models.pvp_models import PvPEvalConfig
from core.models.pvp_models import PvPEvalMetadata
from core.models.pvp_models import PvPEvalResults
from core.models.pvp_models import PvPModelSpec
from core.pvp.sglang_parsers import tool_call_parser_for
from validator.core import constants as vcst
from validator.evaluation.pvp.game_runner import Player
from validator.evaluation.pvp.game_runner import create_player
from validator.evaluation.pvp.game_runner import run_matchup
from validator.evaluation.pvp.game_runner import warmup_player
from validator.evaluation.pvp.server import start_sglang
from validator.evaluation.pvp.server import wait_for_servers
from validator.evaluation.utils import check_for_lora
from validator.evaluation.utils import configure_eval_logging
from validator.evaluation.utils import stop_process


logger = logging.getLogger(__name__)


def main() -> int:
    configure_eval_logging()
    try:
        config = _load_config()
        results = _run_evaluation(config)
        _write_results(results)
        return 0
    except Exception as exc:
        logger.exception("PvP evaluation failed: %s", exc)
        return 1


def _load_config() -> PvPEvalConfig:
    """Load config from env var or mounted file."""
    config_raw = os.getenv(vcst.PVP_CONFIG_ENV_VAR)
    if not config_raw:
        config_path = Path(vcst.PVP_CONFIG_PATH)
        if config_path.exists():
            config_raw = config_path.read_text()

    if not config_raw:
        raise ValueError(
            f"No config found. Set {vcst.PVP_CONFIG_ENV_VAR} env var or mount {vcst.PVP_CONFIG_PATH}"
        )

    return PvPEvalConfig.model_validate_json(config_raw)


def _resolve_spec(spec: PvPModelSpec, default_gpu: int, default_port: int) -> tuple[int, int]:
    """Apply defaults to GPU and port if not explicitly set."""
    gpu = spec.gpu_id if spec.gpu_id is not None else default_gpu
    port = spec.port if spec.port is not None else default_port
    return gpu, port


def _prepare_model(spec: PvPModelSpec, label: str) -> PreparedModel:
    """Detect LoRA and build SGLang flags.

    Passes HF repo IDs to SGLang which handles downloads internally.
    """
    is_lora = check_for_lora(spec.repo, local_files_only=False)
    logger.info("Model %s: repo=%s is_lora=%s", label, spec.repo, is_lora)

    if is_lora:
        lora_name = f"{label}_trained_lora"
        return PreparedModel(
            sglang_model_path=spec.original_model,
            inference_name=f"{spec.original_model}:{lora_name}",
            extra_sglang_args=f"--enable-lora --lora-paths {lora_name}={spec.repo} --lora-backend triton",
        )

    # A full-weight miner repo id is often opaque (no family substring), and the
    # config.json fallback can't run here — SGLang downloads the repo itself, so
    # there is no local weights dir at command-build time. Resolve from the base
    # model instead; without a parser every turn forfeits.
    parser = tool_call_parser_for(spec.repo, log_unmapped=False) or tool_call_parser_for(spec.original_model)
    return PreparedModel(
        sglang_model_path=spec.repo,
        inference_name=spec.repo,
        tool_call_parser=parser,
    )


def _build_chat_config(port: int, eval_config: PvPEvalConfig, prepared: PreparedModel) -> ChatCompletionConfig:
    """Construct ChatCompletionConfig from resolved port and eval settings.

    tokenizer_repo points at the served weights (base repo for LoRA, repo for full
    weights) — never the ':lora'-suffixed inference name — so memory slot budgets
    use real tokens. read_timeout/max_retries are kept under the turn wall-clock.
    """
    return ChatCompletionConfig(
        inference_model=prepared.inference_name,
        tokenizer_repo=prepared.sglang_model_path,
        base_url=f"http://{vcst.PVP_SGLANG_HOST}:{port}{vcst.PVP_SGLANG_API_PATH}",
        temperature=eval_config.temperature,
        seed=eval_config.seed,
        read_timeout=vcst.PVP_HTTP_READ_TIMEOUT_SECONDS,
        max_retries=vcst.PVP_HTTP_MAX_RETRIES,
    )


def _run_evaluation(config: PvPEvalConfig) -> PvPEvalResults:
    """Start servers, run pair matchups, return results."""
    if config.model_a is None or config.model_b is None:
        raise ValueError("Pair mode requires model_a and model_b")

    start_time = time.time()
    model_a = config.model_a
    model_b = config.model_b

    gpu_a, port_a = _resolve_spec(model_a, default_gpu=0, default_port=vcst.PVP_SGLANG_PORT_A)
    gpu_b, port_b = _resolve_spec(model_b, default_gpu=1, default_port=vcst.PVP_SGLANG_PORT_B)

    prepared_a = _prepare_model(model_a, "a")
    prepared_b = _prepare_model(model_b, "b")

    sglang_a: subprocess.Popen | None = None
    sglang_b: subprocess.Popen | None = None
    player_a: Player | None = None
    player_b: Player | None = None

    try:
        sglang_a = start_sglang(prepared_a, gpu_a, port_a, config.seed)
        sglang_b = start_sglang(prepared_b, gpu_b, port_b, config.seed + 1)
        asyncio.run(wait_for_servers(port_a, port_b))

        config_a = _build_chat_config(port_a, config, prepared_a)
        config_b = _build_chat_config(port_b, config, prepared_b)

        player_a = create_player(config_a)
        player_b = create_player(config_b)
        warmup_player(player_a)
        warmup_player(player_b)

        env_results: dict[EnvironmentName, PvPEnvironmentResult] = {}
        for env_name, matchup_config in config.matchups.items():
            logger.info("Starting matchup: %s (%d seeds)", env_name.value, matchup_config.num_games)
            env_results[env_name] = run_matchup(
                env_name=env_name,
                matchup_config=matchup_config,
                player_a=player_a,
                player_b=player_b,
                base_seed=config.seed,
            )

        return PvPEvalResults(
            model_a=model_a.repo,
            model_b=model_b.repo,
            results=env_results,
            metadata=PvPEvalMetadata(
                seed=config.seed,
                temperature=config.temperature,
                wall_time_seconds=time.time() - start_time,
            ),
        )
    finally:
        if player_a:
            player_a.client.close()
        if player_b:
            player_b.client.close()
        stop_process(sglang_a, "sglang-a")
        stop_process(sglang_b, "sglang-b")


def _write_results(results: PvPEvalResults) -> None:
    results_path = Path(os.getenv("PVP_RESULTS_PATH", vcst.PVP_RESULTS_PATH))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(results.model_dump_json(indent=2))
    logger.info("Results written to %s", results_path)


if __name__ == "__main__":
    sys.exit(main())
