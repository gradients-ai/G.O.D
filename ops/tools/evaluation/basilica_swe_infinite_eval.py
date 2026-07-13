#!/usr/bin/env python3
"""Live Basilica smoke test for SWE Infinite individual evaluation.

This exercises the tournament individual-eval path without requiring validator
DB access. It deploys one model to Basilica, exposes the public SGLang proxy,
and calls the external Affinetes SWE Infinite server configured by URL. The
SWE agent is fixed to MiniSWE by the evaluator.

Example:
    BASILICA_API_TOKEN=... SWE_INFINITE_SERVER_BASE_URL=https://affinetes.example \
        uv run --extra dev python -m ops.tools.evaluation.basilica_swe_infinite_eval \
        --model Qwen/Qwen2.5-7B-Instruct \
        --num-seeds 1 \
        --task-id 7 83 45
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import Sequence

from dotenv import load_dotenv

import core.constants.environments as env_cst
from validator.evaluation.docker_evaluation import run_evaluation_individual
from validator.evaluation.swe_infinite_config import DEFAULT_SWE_INFINITE_EVAL_CONFIG
from validator.evaluation.swe_infinite_config import SWE_INFINITE_SERVER_BASE_URL_ENV
from validator.evaluation.swe_infinite_config import SweInfiniteEvalConfig
from validator.evaluation.swe_infinite_config import SweInfiniteTaskSelectionOverride
from validator.scoring.models import MinerRepos


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_HOTKEY = "swe-infinite-smoke-hotkey"
FIXED_SWE_AGENT = "miniswe"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a live Basilica smoke test for the SWE Infinite individual environment evaluator."
    )
    parser.add_argument("--env-file", default=".vali.env", help="Dotenv file to load before reading env vars.")
    parser.add_argument("--model", default=None, help="HF model or LoRA repo to evaluate. Defaults to --base-model.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Original/base model repo.")
    parser.add_argument("--hotkey", default=DEFAULT_HOTKEY, help="Synthetic hotkey used for the single miner result.")
    parser.add_argument(
        "--swe-server-url",
        default=None,
        help="External Affinetes SWE Infinite server URL. Defaults to SWE_INFINITE_SERVER_BASE_URL.",
    )
    parser.add_argument(
        "--image",
        default=env_cst.ENVIRONMENT_CONFIGS[env_cst.EnvironmentName.SWE_INFINITE].tournament_eval_image,
        help="Basilica image to deploy.",
    )
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs to request from Basilica.")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
    parser.add_argument("--num-seeds", type=int, default=None, help="Override the default number of SWE tasks.")
    parser.add_argument(
        "--task-id",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate exactly these SWE task IDs. Example: --task-id 7 83 45.",
    )
    parser.add_argument("--task-id-min", type=int, default=None, help="Override the default SWE task ID minimum.")
    parser.add_argument("--task-id-max", type=int, default=None, help="Override the default SWE task ID maximum.")
    parser.add_argument("--metadata-url", default=None, help="Override the default SWE metadata URL.")
    parser.add_argument("--task-timeout-seconds", type=int, default=None, help="Override per-SWE-task timeout.")
    parser.add_argument("--session-timeout-seconds", type=int, default=None, help="Override total SWE session timeout.")
    parser.add_argument("--max-concurrent-requests", type=int, default=None, help="Override Affinetes request concurrency.")
    parser.add_argument("--affinetes-call-path", default=None, help="Affinetes call path, usually /call or /evaluate.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Agent iteration budget.")
    parser.add_argument("--collect-logprobs", action="store_true", help="Ask Affinetes to collect logprobs when supported.")
    parser.add_argument("--model-api-key", default=None, help="Static API key for the public SGLang proxy.")
    parser.add_argument(
        "--model-base-url",
        default=None,
        help="Override public model base URL instead of inferring from Basilica.",
    )
    parser.add_argument("--base-chain-json", default=None, help="JSON list of prior base repos for continuation LoRA evals.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved config without deploying.")
    return parser.parse_args(argv)


def build_swe_eval_config(args: argparse.Namespace) -> SweInfiniteEvalConfig:
    overrides = {
        "metadata_url": args.metadata_url,
        "task_timeout_seconds": args.task_timeout_seconds,
        "session_timeout_seconds": args.session_timeout_seconds,
        "max_concurrent_requests": args.max_concurrent_requests,
        "affinetes_call_path": args.affinetes_call_path,
        "max_iterations": args.max_iterations,
        "model_api_key": args.model_api_key,
        "model_base_url": args.model_base_url,
    }
    if args.collect_logprobs:
        overrides["collect_logprobs"] = True
    return DEFAULT_SWE_INFINITE_EVAL_CONFIG.with_overrides(**overrides)


def build_task_selection_override(args: argparse.Namespace) -> SweInfiniteTaskSelectionOverride:
    return SweInfiniteTaskSelectionOverride(
        task_id_min=args.task_id_min,
        task_id_max=args.task_id_max,
        num_seeds=args.num_seeds,
        task_ids=tuple(args.task_id) if args.task_id else (),
    )


def _config_for_json(config: SweInfiniteEvalConfig) -> dict:
    payload = json.loads(config.to_json())
    if payload.get("model_api_key"):
        payload["model_api_key"] = "***"
    return payload


def _parse_base_chain(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("--base-chain-json must be a JSON list of strings")
    return parsed


async def run(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file, override=False)

    swe_server_url = args.swe_server_url or os.getenv(SWE_INFINITE_SERVER_BASE_URL_ENV)
    if not swe_server_url:
        raise SystemExit(f"{SWE_INFINITE_SERVER_BASE_URL_ENV} is required. Pass --swe-server-url or set it in the environment.")
    if not os.getenv("BASILICA_API_TOKEN"):
        raise SystemExit("BASILICA_API_TOKEN is required for this live Basilica smoke test.")

    model_repo = args.model or args.base_model
    base_chain = _parse_base_chain(args.base_chain_json)
    base_chains = {args.hotkey: base_chain} if base_chain else None
    os.environ[SWE_INFINITE_SERVER_BASE_URL_ENV] = swe_server_url
    swe_eval_config = build_swe_eval_config(args)
    task_selection_override = build_task_selection_override(args)

    config = {
        "environment": env_cst.EnvironmentName.SWE_INFINITE.value,
        "model": model_repo,
        "base_model": args.base_model,
        "image": args.image,
        "gpu_count": args.gpu_count,
        "seed": args.seed,
        "task_ids": args.task_id or [],
        "agent": FIXED_SWE_AGENT,
        "hotkey": args.hotkey,
        "base_chain": base_chain or [],
        "swe_server_url": swe_server_url,
        "swe_eval_config": _config_for_json(swe_eval_config),
        "task_selection_override": json.loads(task_selection_override.to_json()),
    }
    print("Resolved SWE Infinite Basilica smoke-test config:")
    print(json.dumps(config, indent=2, sort_keys=True))

    if args.dry_run:
        print("Dry run requested; not deploying.")
        return

    start = time.perf_counter()
    result = await run_evaluation_individual(
        miners=MinerRepos(by_hotkey={args.hotkey: model_repo}),
        base_model=args.base_model,
        environment_name=env_cst.EnvironmentName.SWE_INFINITE,
        seed=args.seed,
        image=args.image,
        gpu_count=args.gpu_count,
        task_id=None,
        psql_db=None,
        base_chains=base_chains,
        swe_eval_config=swe_eval_config,
        swe_task_selection_override=task_selection_override,
    )
    elapsed = time.perf_counter() - start

    print("\nSWE Infinite evaluation complete.")
    print(result.model_dump_json(indent=2))
    print(f"Elapsed seconds: {elapsed:.2f}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
