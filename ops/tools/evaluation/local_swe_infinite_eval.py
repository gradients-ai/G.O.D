#!/usr/bin/env python3
"""Local SWE Infinite smoke test for miners.

This runs the same SWE Infinite evaluation request path as the tournament
individual evaluator, but starts the dependencies locally:

* SGLang runs as a host process.
* The Affinetes SWE Infinite server runs as a Docker container.

The default model URL passed to the Dockerized SWE server uses
``host.docker.internal`` so the container can reach the host SGLang process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv

import core.constants.environments as env_cst
import validator.evaluation.constants as vcst
from validator.evaluation.evaluation_logging import configure_eval_logging
from validator.evaluation.evaluators import swe
from validator.evaluation.runtime import stop_process
from validator.evaluation.swe_infinite_config import DEFAULT_SWE_INFINITE_EVAL_CONFIG
from validator.evaluation.swe_infinite_config import SweInfiniteEvalConfig
from validator.evaluation.swe_infinite_config import SweInfiniteTaskSelectionOverride


logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SWE_IMAGE = "gradientsio/swe-infinite:v1"
DEFAULT_SWE_HOST = "127.0.0.1"
DEFAULT_SWE_PORT = 8000
DEFAULT_SWE_CONTAINER_PORT = 8000
DEFAULT_SWE_HEALTH_PATH = "/health"
DEFAULT_SGLANG_HOST = "127.0.0.1"
DEFAULT_SGLANG_PORT = vcst.LOCAL_ENV_SGLANG_PORT
DEFAULT_CONTAINER_MODEL_HOST = "host.docker.internal"


@dataclass(frozen=True)
class EvalSelection:
    eval_list: list[tuple[int, int]]
    task_id_min: int
    task_id_max: int
    num_seeds: int
    explicit_task_ids: list[int]


@dataclass(frozen=True)
class SweServerHandle:
    url: str
    container: object | None = None
    docker_client: object | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local SWE Infinite evaluation by starting host SGLang and a Dockerized "
            "Affinetes SWE Infinite server."
        )
    )
    parser.add_argument("--env-file", default=".vali.env", help="Dotenv file to load before reading env vars.")
    parser.add_argument("--model", default=None, help="HF model or LoRA repo to evaluate. Defaults to --base-model.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Original/base model repo.")
    parser.add_argument("--seed", type=int, default=vcst.ENV_EVAL_DEFAULT_SEED, help="Evaluation seed.")
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
    parser.add_argument("--max-iterations", type=int, default=None, help="MiniSWE iteration budget.")
    parser.add_argument("--collect-logprobs", action="store_true", help="Ask Affinetes to collect logprobs when supported.")
    parser.add_argument("--model-api-key", default=None, help="API key passed to the OpenAI-compatible model server.")
    parser.add_argument(
        "--model-base-url",
        default=None,
        help="Override the model base URL passed to the SWE server. /v1 is appended when missing.",
    )
    parser.add_argument("--base-chain-json", default=None, help="JSON list of prior base repos for continuation LoRA evals.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override ENV_EVAL_TEMPERATURE. Defaults to production SWE evaluator behavior.",
    )

    parser.add_argument("--sglang-host", default=DEFAULT_SGLANG_HOST, help="Host used for local SGLang health checks.")
    parser.add_argument("--sglang-port", type=int, default=DEFAULT_SGLANG_PORT, help="Local SGLang port.")
    parser.add_argument(
        "--sglang-base-url",
        default=None,
        help="Local SGLang base URL for health checks. Defaults to http://<sglang-host>:<sglang-port>.",
    )
    parser.add_argument("--sglang-health-path", default="/v1/models", help="SGLang health path.")
    parser.add_argument("--sglang-health-timeout", type=int, default=1800, help="Seconds to wait for SGLang.")
    parser.add_argument("--sglang-start-cmd", default=None, help="Custom SGLang command. Passed as SGLANG_START_CMD.")
    parser.add_argument(
        "--use-existing-sglang",
        action="store_true",
        help="Do not launch SGLang; use --sglang-base-url or the default local SGLang URL.",
    )
    parser.add_argument(
        "--skip-sglang-health-check",
        action="store_true",
        help="Skip the SGLang /v1/models health check before evaluation.",
    )
    parser.add_argument(
        "--inference-model-name",
        default=None,
        help="OpenAI model name exposed by SGLang. Defaults to the repo or native LoRA name.",
    )

    parser.add_argument("--swe-image", default=DEFAULT_SWE_IMAGE, help="Docker image for the SWE Infinite server.")
    parser.add_argument(
        "--swe-server-url",
        default=None,
        help="Use an already-running SWE server URL instead of starting the Docker image.",
    )
    parser.add_argument("--swe-host", default=DEFAULT_SWE_HOST, help="Host IP for the local Docker port binding.")
    parser.add_argument("--swe-port", type=int, default=DEFAULT_SWE_PORT, help="Host port for the SWE server.")
    parser.add_argument(
        "--swe-container-port",
        type=int,
        default=DEFAULT_SWE_CONTAINER_PORT,
        help="Port exposed by the SWE server inside the Docker container.",
    )
    parser.add_argument("--swe-container-name", default=None, help="Optional Docker container name.")
    parser.add_argument("--swe-health-path", default=DEFAULT_SWE_HEALTH_PATH, help="SWE server health path.")
    parser.add_argument("--swe-health-timeout", type=int, default=600, help="Seconds to wait for the SWE server.")
    parser.add_argument("--skip-swe-health-check", action="store_true", help="Skip the SWE server health check.")
    parser.add_argument("--keep-swe-server", action="store_true", help="Leave the SWE server container running.")
    parser.add_argument(
        "--container-model-host",
        default=DEFAULT_CONTAINER_MODEL_HOST,
        help="Hostname the Dockerized SWE server should use to reach host SGLang.",
    )
    parser.add_argument(
        "--no-host-gateway",
        action="store_true",
        help="Do not add host.docker.internal:host-gateway to the SWE server container.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write a JSON summary.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved config without launching services.")
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


def build_sglang_env_overrides(args: argparse.Namespace, sglang_base_url: str) -> dict[str, str]:
    overrides = {
        "SGLANG_PORT": str(args.sglang_port),
        "SGLANG_BASE_URL": sglang_base_url,
        "SGLANG_HEALTH_PATH": args.sglang_health_path,
    }
    if args.sglang_start_cmd:
        overrides["SGLANG_START_CMD"] = args.sglang_start_cmd
    return overrides


@contextmanager
def temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    old_values = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _masked_overrides(overrides: dict[str, str]) -> dict[str, str]:
    masked = dict(overrides)
    if "SWE_INFINITE_MODEL_API_KEY" in masked:
        masked["SWE_INFINITE_MODEL_API_KEY"] = "***"
    return masked


def _config_for_json(config: SweInfiniteEvalConfig) -> dict:
    payload = json.loads(config.to_json())
    if payload.get("model_api_key"):
        payload["model_api_key"] = "***"
    return payload


def _parse_base_chain(raw: str | None) -> list[str]:
    if not raw:
        return []
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("--base-chain-json must be a JSON list of strings")
    return parsed


def _local_sglang_base_url(args: argparse.Namespace) -> str:
    return args.sglang_base_url or f"http://{args.sglang_host}:{args.sglang_port}"


def _local_swe_server_url(args: argparse.Namespace) -> str:
    return f"http://{args.swe_host}:{args.swe_port}"


def _is_loopback_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def resolve_model_base_url(args: argparse.Namespace, *, swe_runs_in_docker: bool) -> str:
    if args.model_base_url:
        return swe._with_v1(args.model_base_url)

    sglang_base_url = _local_sglang_base_url(args)
    if swe_runs_in_docker and _is_loopback_url(sglang_base_url):
        return swe._with_v1(f"http://{args.container_model_host}:{args.sglang_port}")
    return swe._with_v1(sglang_base_url)


def _resolve_explicit_task_ids(task_selection_override: SweInfiniteTaskSelectionOverride) -> list[int]:
    invalid_task_ids = [task_id for task_id in task_selection_override.task_ids if task_id <= 0]
    if invalid_task_ids:
        raise ValueError(f"Invalid SWE task ids {invalid_task_ids}; expected positive integers")
    return list(task_selection_override.task_ids)


async def resolve_eval_selection(
    args: argparse.Namespace,
    eval_config: SweInfiniteEvalConfig | None = None,
    task_selection_override: SweInfiniteTaskSelectionOverride | None = None,
) -> EvalSelection:
    eval_config = eval_config or build_swe_eval_config(args)
    task_selection_override = task_selection_override or build_task_selection_override(args)
    env_config = env_cst.ENVIRONMENT_CONFIGS[env_cst.EnvironmentName.SWE_INFINITE]
    explicit_task_ids = _resolve_explicit_task_ids(task_selection_override)

    if explicit_task_ids:
        return EvalSelection(
            eval_list=swe._build_eval_list_for_task_ids(args.seed, explicit_task_ids),
            task_id_min=min(explicit_task_ids),
            task_id_max=max(explicit_task_ids),
            num_seeds=len(explicit_task_ids),
            explicit_task_ids=explicit_task_ids,
        )

    task_id_min, task_id_max = await swe._resolve_task_range(env_config, eval_config, task_selection_override)
    num_seeds = (
        task_selection_override.num_seeds
        if task_selection_override.num_seeds is not None
        else env_config.num_seeds
    )
    return EvalSelection(
        eval_list=swe._build_eval_list(args.seed, num_seeds, task_id_min, task_id_max),
        task_id_min=task_id_min,
        task_id_max=task_id_max,
        num_seeds=num_seeds,
        explicit_task_ids=[],
    )


def build_swe_docker_command(args: argparse.Namespace, container_name: str) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{args.swe_host}:{args.swe_port}:{args.swe_container_port}",
    ]
    if not args.no_host_gateway:
        command.extend(["--add-host", f"{args.container_model_host}:host-gateway"])
    command.append(args.swe_image)
    return command


async def start_swe_server(args: argparse.Namespace) -> SweServerHandle:
    if args.swe_server_url:
        if not args.skip_swe_health_check:
            await swe._wait_for_health(
                args.swe_server_url.rstrip("/"),
                args.swe_health_path,
                args.swe_health_timeout,
                service_name="SWE Infinite",
            )
        return SweServerHandle(url=args.swe_server_url.rstrip("/"))

    import docker

    container_name = args.swe_container_name or f"swe-infinite-local-{uuid.uuid4().hex[:8]}"
    docker_client = docker.from_env()
    extra_hosts = None if args.no_host_gateway else {args.container_model_host: "host-gateway"}

    container = None
    try:
        logger.info("Starting SWE Infinite container %s from image %s", container_name, args.swe_image)
        container = await asyncio.to_thread(
            docker_client.containers.run,
            args.swe_image,
            name=container_name,
            detach=True,
            ports={f"{args.swe_container_port}/tcp": (args.swe_host, args.swe_port)},
            extra_hosts=extra_hosts,
            remove=False,
        )
        swe_server_url = _local_swe_server_url(args)
        if not args.skip_swe_health_check:
            await swe._wait_for_health(
                swe_server_url,
                args.swe_health_path,
                args.swe_health_timeout,
                service_name="SWE Infinite",
            )
        return SweServerHandle(url=swe_server_url, container=container, docker_client=docker_client)
    except Exception:
        if container is not None:
            await asyncio.to_thread(container.remove, force=True)
        await asyncio.to_thread(docker_client.close)
        raise


async def cleanup_swe_server(handle: SweServerHandle | None, *, keep: bool) -> None:
    if handle is None:
        return
    try:
        if handle.container is not None:
            if keep:
                name = getattr(handle.container, "name", "<unknown>")
                logger.info("Leaving SWE Infinite container running: %s", name)
            else:
                await asyncio.to_thread(handle.container.remove, force=True)
                logger.info("Removed SWE Infinite container")
    finally:
        if handle.docker_client is not None:
            await asyncio.to_thread(handle.docker_client.close)


def _ensure_flashinfer_workspace() -> None:
    min_workspace = vcst.SGLANG_FLASHINFER_WORKSPACE_MIN_BYTES
    try:
        current_workspace = int(os.environ.get("SGLANG_FLASHINFER_WORKSPACE_SIZE", "0") or "0")
    except ValueError:
        current_workspace = 0
    if current_workspace < min_workspace:
        os.environ["SGLANG_FLASHINFER_WORKSPACE_SIZE"] = str(min_workspace)


async def _start_or_reuse_sglang(
    args: argparse.Namespace,
    *,
    model_repo: str,
    base_model: str,
    base_chain: list[str],
) -> tuple[str, object | None, asyncio.Task | None]:
    if args.use_existing_sglang:
        inference_model_name = args.inference_model_name or model_repo
        if not args.skip_sglang_health_check:
            await swe._wait_for_health(
                _local_sglang_base_url(args),
                args.sglang_health_path,
                args.sglang_health_timeout,
                service_name="SGLang",
            )
        return inference_model_name, None, None

    inference_model_name, model_path_for_sglang, sglang_command = await swe._prepare_sglang_command(
        model_repo=model_repo,
        original_model=base_model,
        base_chain=base_chain,
        base_seed=args.seed,
    )
    if args.inference_model_name:
        inference_model_name = args.inference_model_name

    _ensure_flashinfer_workspace()
    logger.info(
        "Launching SGLang: model_path=%s inference_model_name=%s local_base_url=%s",
        model_path_for_sglang,
        inference_model_name,
        _local_sglang_base_url(args),
    )
    logger.info("SGLang command: %s", sglang_command)
    sglang_proc = swe._start_process(sglang_command, "sglang")
    sglang_log_task = asyncio.create_task(swe._stream_logs(sglang_proc, "sglang"))
    if not args.skip_sglang_health_check:
        await swe._wait_for_health(
            _local_sglang_base_url(args),
            args.sglang_health_path,
            args.sglang_health_timeout,
            service_name="SGLang",
        )
    return inference_model_name, sglang_proc, sglang_log_task


def _evals_for_json(selection: EvalSelection) -> list[dict[str, int]]:
    return [
        {"seed": seed, "task_id": task_id}
        for seed, task_id in selection.eval_list
    ]


async def run(args: argparse.Namespace) -> dict:
    load_dotenv(args.env_file, override=False)

    model_repo = args.model or args.base_model
    base_chain = _parse_base_chain(args.base_chain_json)
    swe_runs_in_docker = args.swe_server_url is None
    swe_server_url = args.swe_server_url.rstrip("/") if args.swe_server_url else _local_swe_server_url(args)
    sglang_base_url = _local_sglang_base_url(args).rstrip("/")
    model_base_url = resolve_model_base_url(args, swe_runs_in_docker=swe_runs_in_docker)
    swe_eval_config = build_swe_eval_config(args)
    task_selection_override = build_task_selection_override(args)
    temperature = args.temperature
    if temperature is None:
        temperature = vcst.ENV_EVAL_TEMPERATURE
    task_timeout = swe_eval_config.task_timeout_seconds

    env_overrides = build_sglang_env_overrides(args, sglang_base_url)

    with temporary_env(env_overrides):
        selection = await resolve_eval_selection(args, swe_eval_config, task_selection_override)
        container_name = args.swe_container_name or "swe-infinite-local-<generated>"
        config = {
            "environment": env_cst.EnvironmentName.SWE_INFINITE.value,
            "model": model_repo,
            "base_model": args.base_model,
            "seed": args.seed,
            "temperature": temperature,
            "agent": swe.MINISWE_AGENT_NAME,
            "base_chain": base_chain,
            "swe_server_url": swe_server_url,
            "swe_runs_in_docker": swe_runs_in_docker,
            "swe_docker_command": build_swe_docker_command(args, container_name) if swe_runs_in_docker else [],
            "sglang_base_url": sglang_base_url,
            "model_base_url_sent_to_swe": model_base_url,
            "task_id_range": [selection.task_id_min, selection.task_id_max],
            "num_seeds": selection.num_seeds,
            "explicit_task_ids": selection.explicit_task_ids,
            "evaluations": _evals_for_json(selection),
            "swe_eval_config": _config_for_json(swe_eval_config),
            "task_selection_override": json.loads(task_selection_override.to_json()),
            "env_overrides": _masked_overrides(env_overrides),
        }
        print("Resolved local SWE Infinite evaluation config:")
        print(json.dumps(config, indent=2, sort_keys=True))

        if args.dry_run:
            print("Dry run requested; not launching services.")
            return {**config, "avg_score": None, "elapsed_seconds": 0.0}

        sglang_proc = None
        sglang_log_task = None
        swe_handle = None
        start = time.perf_counter()
        try:
            inference_model_name, sglang_proc, sglang_log_task = await _start_or_reuse_sglang(
                args,
                model_repo=model_repo,
                base_model=args.base_model,
                base_chain=base_chain,
            )
            swe_handle = await start_swe_server(args)
            avg_score = await swe._run_swe_evaluation(
                swe_server_url=swe_handle.url,
                model_base_url=model_base_url,
                inference_model_name=inference_model_name,
                eval_list=selection.eval_list,
                temperature=temperature,
                task_timeout=task_timeout,
                eval_config=swe_eval_config,
            )
            elapsed = time.perf_counter() - start
            summary = {
                **config,
                "inference_model_name": inference_model_name,
                "avg_score": avg_score,
                "elapsed_seconds": elapsed,
            }
            print("\nSWE Infinite local evaluation complete.")
            print(json.dumps(summary, indent=2, sort_keys=True))
            if args.output_json:
                from pathlib import Path

                Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            return summary
        finally:
            await cleanup_swe_server(swe_handle, keep=args.keep_swe_server)
            stop_process(sglang_proc, "sglang")
            if sglang_log_task:
                sglang_log_task.cancel()


def main(argv: Sequence[str] | None = None) -> int:
    configure_eval_logging()
    args = parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
