#!/usr/bin/env python3
"""Live Basilica smoke test for SWE Infinite individual evaluation.

This exercises the tournament individual-eval path without requiring validator
DB access. It deploys one model to Basilica, exposes the public SGLang proxy,
and calls the external Affinetes SWE Infinite server configured by URL.

Example:
    BASILICA_API_KEY=... SWE_INFINITE_SERVER_BASE_URL=https://affinetes.example \
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
from validator.scoring.models import MinerRepos


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_HOTKEY = "swe-infinite-smoke-hotkey"


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
    parser.add_argument("--num-seeds", type=int, default=None, help="Override SWE_INFINITE_NUM_SEEDS.")
    parser.add_argument(
        "--task-id",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate exactly these SWE task IDs. Example: --task-id 7 83 45.",
    )
    parser.add_argument("--task-id-min", type=int, default=None, help="Override SWE_INFINITE_TASK_ID_MIN.")
    parser.add_argument("--task-id-max", type=int, default=None, help="Override SWE_INFINITE_TASK_ID_MAX.")
    parser.add_argument("--metadata-url", default=None, help="Override SWE_INFINITE_METADATA_URL.")
    parser.add_argument("--task-timeout-seconds", type=int, default=None, help="Override per-SWE-task timeout.")
    parser.add_argument("--session-timeout-seconds", type=int, default=None, help="Override total SWE session timeout.")
    parser.add_argument("--max-concurrent-requests", type=int, default=None, help="Override Affinetes request concurrency.")
    parser.add_argument("--affinetes-call-path", default=None, help="Affinetes call path, usually /call or /evaluate.")
    parser.add_argument("--agent", default=None, help="SWE agent override, for example miniswe, affent, or codex.")
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


def build_swe_env_overrides(args: argparse.Namespace, swe_server_url: str) -> dict[str, str]:
    overrides = {"SWE_INFINITE_SERVER_BASE_URL": swe_server_url}

    optional_values = {
        "SWE_INFINITE_TASK_IDS": ",".join(str(task_id) for task_id in args.task_id) if args.task_id else None,
        "SWE_INFINITE_METADATA_URL": args.metadata_url,
        "SWE_INFINITE_TASK_ID_MIN": args.task_id_min,
        "SWE_INFINITE_TASK_ID_MAX": args.task_id_max,
        "SWE_INFINITE_NUM_SEEDS": args.num_seeds,
        "SWE_INFINITE_TASK_TIMEOUT_SECONDS": args.task_timeout_seconds,
        "SWE_INFINITE_SESSION_TIMEOUT": args.session_timeout_seconds,
        "SWE_INFINITE_MAX_CONCURRENT_REQUESTS": args.max_concurrent_requests,
        "SWE_INFINITE_AFFINETES_CALL_PATH": args.affinetes_call_path,
        "SWE_INFINITE_AGENT": args.agent,
        "SWE_INFINITE_MAX_ITERATIONS": args.max_iterations,
        "SWE_INFINITE_MODEL_API_KEY": args.model_api_key,
        "SWE_INFINITE_MODEL_BASE_URL": args.model_base_url,
    }
    for key, value in optional_values.items():
        if value is not None and value != "":
            overrides[key] = str(value)
    if args.collect_logprobs:
        overrides["SWE_INFINITE_COLLECT_LOGPROBS"] = "true"
    return overrides


def apply_env_overrides(overrides: dict[str, str]) -> None:
    for key, value in overrides.items():
        os.environ[key] = value


def _masked_overrides(overrides: dict[str, str]) -> dict[str, str]:
    masked = dict(overrides)
    if "SWE_INFINITE_MODEL_API_KEY" in masked:
        masked["SWE_INFINITE_MODEL_API_KEY"] = "***"
    return masked


def _parse_base_chain(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("--base-chain-json must be a JSON list of strings")
    return parsed


async def run(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file, override=False)

    swe_server_url = args.swe_server_url or os.getenv("SWE_INFINITE_SERVER_BASE_URL")
    if not swe_server_url:
        raise SystemExit("SWE_INFINITE_SERVER_BASE_URL is required. Pass --swe-server-url or set it in the environment.")
    if not os.getenv("BASILICA_API_KEY"):
        raise SystemExit("BASILICA_API_KEY is required for this live Basilica smoke test.")

    model_repo = args.model or args.base_model
    base_chain = _parse_base_chain(args.base_chain_json)
    base_chains = {args.hotkey: base_chain} if base_chain else None
    overrides = build_swe_env_overrides(args, swe_server_url)
    apply_env_overrides(overrides)

    config = {
        "environment": env_cst.EnvironmentName.SWE_INFINITE.value,
        "model": model_repo,
        "base_model": args.base_model,
        "image": args.image,
        "gpu_count": args.gpu_count,
        "seed": args.seed,
        "task_ids": args.task_id or [],
        "hotkey": args.hotkey,
        "base_chain": base_chain or [],
        "env_overrides": _masked_overrides(overrides),
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
