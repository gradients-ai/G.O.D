#!/usr/bin/env python3
"""Run process_miners_pool through InterCode and Liar's Dice tournament evals.

This is a live Basilica smoke test. It keeps the production evaluation path but
replaces task/submission DB reads and tournament-result persistence with a small
in-memory store so it can be run from a checkout without seeded validator rows.

Example:
    BASILICA_API_KEY=... uv run --extra dev \
        --with Pillow==11.1.0 --with transformers==4.46.2 --with cryptography \
        python -m scripts.process_miners_pool_mixed_env_eval \
        --miner gradients-io-tournaments/example-miner-a \
        --miner gradients-io-tournaments/example-miner-b
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
import types
import warnings
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID
from uuid import uuid4


warnings.filterwarnings("ignore", message=r'Field ".*" in .* has conflict with protected namespace')

from core.constants import EnvironmentName
from core.models.pvp_models import PvPEnvironmentResult
from core.models.pvp_models import PvPIndividualScoreDbRow
from core.models.pvp_models import PvPPairDbRow
from core.models.pvp_models import PvPPairResult
from core.models.pvp_models import PvPStatus
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.models import EnvRawTask
from validator.evaluation import basilica as basilica_eval
from validator.evaluation import docker_evaluation


def preload_tournament_gpu_module() -> None:
    """Load validator.tournament.gpu without executing validator.tournament.__init__."""
    module_name = "validator.tournament.gpu"
    if module_name in sys.modules:
        return

    repo_root = Path(__file__).resolve().parents[1]
    package_name = "validator.tournament"
    package = types.ModuleType(package_name)
    package.__path__ = [str(repo_root / "validator" / "tournament")]
    sys.modules.setdefault(package_name, package)

    module_path = repo_root / "validator" / "tournament" / "gpu.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


preload_tournament_gpu_module()

from validator.evaluation import scoring


DEFAULT_BASE_MODEL = "Qwen/Qwen2-7B-Instruct"
DEFAULT_MODEL_PARAMS_COUNT = 7_000_000_000
DEFAULT_SEED = 42


@dataclass(frozen=True)
class MinerSpec:
    hotkey: str
    expected_repo_name: str


@dataclass
class TimingWindow:
    label: str
    start: float
    end: float
    source: str

    @property
    def seconds(self) -> float:
        return max(0.0, self.end - self.start)


class EvaluationTimingTracker:
    def __init__(self) -> None:
        self.script_start = time.perf_counter()
        self.windows: list[TimingWindow] = []

    def record_window(self, label: str, start: float, end: float, source: str) -> None:
        self.windows.append(TimingWindow(label=label, start=start, end=end, source=source))

    def record_duration_ending_now(self, label: str, seconds: float, source: str) -> None:
        end = time.perf_counter()
        self.record_window(label=label, start=end - max(0.0, seconds), end=end, source=source)

    def total_eval_seconds(self) -> float:
        if not self.windows:
            return 0.0

        intervals = sorted((window.start, window.end) for window in self.windows)
        total = 0.0
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
                continue
            total += current_end - current_start
            current_start, current_end = start, end
        total += current_end - current_start
        return total

    def summary(self) -> dict[str, Any]:
        script_wall_seconds = time.perf_counter() - self.script_start
        eval_seconds = self.total_eval_seconds()
        return {
            "evaluation_wall_seconds_excluding_basilica_startup": round(eval_seconds, 3),
            "full_script_wall_seconds": round(script_wall_seconds, 3),
            "non_evaluation_wall_seconds": round(max(0.0, script_wall_seconds - eval_seconds), 3),
            "windows": [
                {
                    "label": window.label,
                    "seconds": round(window.seconds, 3),
                    "source": window.source,
                }
                for window in self.windows
            ],
        }


class InMemoryTournamentStore:
    """Mimic the tournament SQL helpers used by scoring.py."""

    def __init__(self) -> None:
        self.pvp_rows: dict[tuple[str, str, str, str], PvPPairDbRow] = {}
        self.individual_rows: dict[tuple[str, str, str], PvPIndividualScoreDbRow] = {}

    async def get_pvp_pair_results(self, task_id: str, psql_db: Any = None) -> list[PvPPairDbRow]:
        return [row for key, row in self.pvp_rows.items() if key[0] == task_id]

    async def ensure_pvp_pairs_exist(
        self,
        task_id: str,
        pairs: list[PvPPairResult],
        environment_names: list[str],
        psql_db: Any = None,
    ) -> None:
        for pair in pairs:
            hotkey_a, hotkey_b = sorted([pair.hotkey_a, pair.hotkey_b])
            for environment_name in environment_names:
                key = (task_id, hotkey_a, hotkey_b, environment_name)
                self.pvp_rows.setdefault(
                    key,
                    PvPPairDbRow(
                        task_id=task_id,
                        hotkey_a=hotkey_a,
                        hotkey_b=hotkey_b,
                        environment_name=environment_name,
                        status=PvPStatus.PENDING,
                    ),
                )

    async def save_pvp_pair_result(
        self,
        task_id: str,
        result: PvPPairResult,
        environment_name: str,
        env_result: PvPEnvironmentResult,
        psql_db: Any = None,
    ) -> None:
        hotkey_a, hotkey_b = sorted([result.hotkey_a, result.hotkey_b])
        swapped = hotkey_a != result.hotkey_a
        model_a_wins = env_result.model_b_wins if swapped else env_result.model_a_wins
        model_b_wins = env_result.model_a_wins if swapped else env_result.model_b_wins
        key = (task_id, hotkey_a, hotkey_b, environment_name)
        attempts = self.pvp_rows[key].n_attempts if key in self.pvp_rows else 0
        deployment_id = self.pvp_rows[key].deployment_id if key in self.pvp_rows else None
        self.pvp_rows[key] = PvPPairDbRow(
            task_id=task_id,
            hotkey_a=hotkey_a,
            hotkey_b=hotkey_b,
            environment_name=environment_name,
            model_a_wins=model_a_wins,
            model_b_wins=model_b_wins,
            draws=env_result.draws,
            total_games=env_result.total_games,
            n_attempts=attempts,
            deployment_id=deployment_id,
            status=PvPStatus.COMPLETE,
        )

    async def set_pvp_pair_deployment_id(
        self,
        task_id: str,
        hotkey_a: str,
        hotkey_b: str,
        deployment_id: str | None,
        psql_db: Any = None,
    ) -> None:
        sorted_a, sorted_b = sorted([hotkey_a, hotkey_b])
        for key, row in list(self.pvp_rows.items()):
            if key[0] == task_id and key[1] == sorted_a and key[2] == sorted_b:
                self.pvp_rows[key] = row.model_copy(update={"deployment_id": deployment_id})

    async def increment_pvp_pair_attempts(
        self,
        task_id: str,
        hotkey_a: str,
        hotkey_b: str,
        psql_db: Any = None,
    ) -> None:
        sorted_a, sorted_b = sorted([hotkey_a, hotkey_b])
        for key, row in list(self.pvp_rows.items()):
            if key[0] == task_id and key[1] == sorted_a and key[2] == sorted_b and not row.is_complete:
                self.pvp_rows[key] = row.model_copy(update={"n_attempts": row.n_attempts + 1})

    async def ensure_individual_scores_exist(
        self,
        task_id: str,
        hotkeys: list[str],
        environment_names: list[str],
        psql_db: Any = None,
    ) -> None:
        for hotkey in hotkeys:
            for environment_name in environment_names:
                key = (task_id, hotkey, environment_name)
                self.individual_rows.setdefault(
                    key,
                    PvPIndividualScoreDbRow(
                        task_id=task_id,
                        hotkey=hotkey,
                        environment_name=environment_name,
                        status=PvPStatus.PENDING,
                    ),
                )

    async def get_individual_scores(self, task_id: str, psql_db: Any = None) -> list[PvPIndividualScoreDbRow]:
        return [row for key, row in self.individual_rows.items() if key[0] == task_id]

    async def save_individual_score(
        self,
        task_id: str,
        hotkey: str,
        environment_name: str,
        score: float,
        psql_db: Any = None,
    ) -> None:
        key = (task_id, hotkey, environment_name)
        attempts = self.individual_rows[key].n_attempts if key in self.individual_rows else 0
        self.individual_rows[key] = PvPIndividualScoreDbRow(
            task_id=task_id,
            hotkey=hotkey,
            environment_name=environment_name,
            score=score,
            n_attempts=attempts,
            status=PvPStatus.COMPLETE,
        )

    async def increment_individual_score_attempts(
        self,
        task_id: str,
        hotkey: str,
        environment_name: str,
        psql_db: Any = None,
    ) -> None:
        key = (task_id, hotkey, environment_name)
        row = self.individual_rows.get(
            key,
            PvPIndividualScoreDbRow(
                task_id=task_id,
                hotkey=hotkey,
                environment_name=environment_name,
                status=PvPStatus.PENDING,
            ),
        )
        if not row.is_complete:
            self.individual_rows[key] = row.model_copy(update={"n_attempts": row.n_attempts + 1})


class FakeHfApi:
    def repo_info(self, repo: str, timeout: int = 30) -> SimpleNamespace:
        print(f"[hf-check] skipped repo_info({repo!r}, timeout={timeout})")
        return SimpleNamespace(id=repo)


def parse_miner_spec(raw_value: str) -> tuple[str, str | None, str]:
    """Return (hotkey, namespace, expected_repo_name)."""
    hotkey: str | None = None
    repo_value = raw_value
    if "=" in raw_value:
        hotkey, repo_value = raw_value.split("=", 1)
        if not hotkey:
            raise ValueError(f"Invalid miner spec {raw_value!r}: hotkey is empty")

    namespace = None
    expected_repo_name = repo_value
    if "/" in repo_value:
        namespace, expected_repo_name = repo_value.split("/", 1)
        if not namespace or not expected_repo_name:
            raise ValueError(f"Invalid repo spec {raw_value!r}: expected namespace/repo")

    if not hotkey:
        hotkey = expected_repo_name
    return hotkey, namespace, expected_repo_name


def resolve_miners(raw_values: list[str], hf_namespace: str | None) -> tuple[str, list[MinerSpec]]:
    namespaces: list[str] = []
    parsed: list[tuple[str, str | None, str]] = []
    for raw_value in raw_values:
        hotkey, namespace, expected_repo_name = parse_miner_spec(raw_value)
        parsed.append((hotkey, namespace, expected_repo_name))
        if namespace:
            namespaces.append(namespace)

    resolved_namespace = hf_namespace or (namespaces[0] if namespaces else scoring.cts.RAYONLABS_HF_USERNAME)
    mismatches = sorted({namespace for namespace in namespaces if namespace != resolved_namespace})
    if mismatches:
        raise ValueError(
            "All full repo specs must use the same namespace as --hf-namespace. "
            f"resolved namespace={resolved_namespace!r}, mismatches={mismatches!r}"
        )

    miners = [MinerSpec(hotkey=hotkey, expected_repo_name=expected_repo_name) for hotkey, _, expected_repo_name in parsed]
    duplicate_hotkeys = sorted({miner.hotkey for miner in miners if [m.hotkey for m in miners].count(miner.hotkey) > 1})
    if duplicate_hotkeys:
        raise ValueError(f"Duplicate miner hotkeys are not allowed: {duplicate_hotkeys}")
    return resolved_namespace, miners


def build_task(args: argparse.Namespace) -> EnvRawTask:
    task_id = UUID(args.task_id) if args.task_id else uuid4()
    return EnvRawTask(
        is_organic=False,
        task_id=task_id,
        status=TaskStatus.EVALUATING,
        model_id=args.base_model,
        ds="process-miners-pool-mixed-env-live-test",
        account_id=uuid4(),
        hours_to_complete=1.0,
        created_at=datetime.now(timezone.utc),
        task_type=TaskType.ENVIRONMENTTASK,
        model_params_count=args.model_params_count,
        environment_names=[EnvironmentName.INTERCODE, EnvironmentName.LIARS_DICE],
        eval_seed=args.seed,
    )


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    return value


def model_dump_pretty(value: Any) -> str:
    return json.dumps(jsonable(value), indent=2, sort_keys=True, default=str)


def install_in_memory_patches(
    *,
    store: InMemoryTournamentStore,
    miners: list[MinerSpec],
    seed: int,
    skip_hf_repo_check: bool,
    poll_interval_seconds: int | None,
    timing_tracker: EvaluationTimingTracker,
) -> None:
    repo_by_hotkey = {miner.hotkey: miner.expected_repo_name for miner in miners}

    async def get_expected_repo_name(_task_id: UUID, hotkey: str, _psql_db: Any) -> str | None:
        return repo_by_hotkey.get(hotkey)

    async def get_env_task_eval_seed(_task_id: UUID, _psql_db: Any) -> int:
        return seed

    async def get_training_status_for_task(_task_id: str, _psql_db: Any) -> dict[str, str]:
        return {}

    async def load_eval_pair_state_for_models(
        _task_id: UUID | None,
        _psql_db: Any,
        _models: list[str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        return {}, {}

    async def load_shared_eval_deployment_id(
        _task_id: UUID | None,
        _psql_db: Any,
        _hotkeys: list[str],
    ) -> None:
        return None

    async def persist_deployment_ids_for_repo(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def persist_shared_eval_deployment_id(*_args: Any, **_kwargs: Any) -> None:
        return None

    scoring.get_expected_repo_name = get_expected_repo_name
    scoring.get_env_task_eval_seed = get_env_task_eval_seed
    scoring.tournament_sql.get_training_status_for_task = get_training_status_for_task
    scoring.tournament_sql.get_pvp_pair_results = store.get_pvp_pair_results
    scoring.tournament_sql.ensure_pvp_pairs_exist = store.ensure_pvp_pairs_exist
    scoring.tournament_sql.save_pvp_pair_result = store.save_pvp_pair_result
    scoring.tournament_sql.set_pvp_pair_deployment_id = store.set_pvp_pair_deployment_id
    scoring.tournament_sql.increment_pvp_pair_attempts = store.increment_pvp_pair_attempts
    scoring.tournament_sql.ensure_individual_scores_exist = store.ensure_individual_scores_exist
    scoring.tournament_sql.get_individual_scores = store.get_individual_scores
    scoring.tournament_sql.save_individual_score = store.save_individual_score
    scoring.tournament_sql.increment_individual_score_attempts = store.increment_individual_score_attempts
    docker_evaluation.load_eval_pair_state_for_models = load_eval_pair_state_for_models
    docker_evaluation.load_shared_eval_deployment_id = load_shared_eval_deployment_id
    docker_evaluation.persist_shared_eval_deployment_id = persist_shared_eval_deployment_id
    basilica_eval.persist_deployment_ids_for_repo = persist_deployment_ids_for_repo

    if skip_hf_repo_check:
        scoring.HfApi = FakeHfApi

    original_poll = basilica_eval._poll_basilica_result

    async def patched_poll(
        deployment: Any,
        repo: str,
        eval_logger: Any,
        *args: Any,
        **kwargs: Any,
    ) -> dict | str:
        if poll_interval_seconds is not None:
            kwargs.setdefault("poll_interval_seconds", poll_interval_seconds)

        start = time.perf_counter()
        try:
            return await original_poll(deployment, repo, *args, eval_logger=eval_logger, **kwargs)
        finally:
            end = time.perf_counter()
            if not repo.startswith("pvp-"):
                timing_tracker.record_window(
                    label=f"individual:{repo}",
                    start=start,
                    end=end,
                    source="post-deploy Basilica polling window",
                )

    basilica_eval._poll_basilica_result = patched_poll
    docker_evaluation._poll_basilica_result = patched_poll


def install_result_logging(timing_tracker: EvaluationTimingTracker) -> None:
    original_pair_eval = scoring.run_evaluation_pvp_pair
    original_individual_eval = scoring.run_evaluation_individual

    async def logged_pair_eval(*args: Any, **kwargs: Any) -> Any:
        environment_names = kwargs.get("environment_names", [])
        hotkey_a = kwargs.get("hotkey_a", args[2] if len(args) > 2 else None)
        hotkey_b = kwargs.get("hotkey_b", args[3] if len(args) > 3 else None)
        gpu_count = kwargs.get("gpu_count", args[8] if len(args) > 8 else None)
        if gpu_count != scoring.cts.PVP_BASILICA_GPU_COUNT:
            raise RuntimeError(
                f"PvP pair eval requested gpu_count={gpu_count}; "
                f"expected {scoring.cts.PVP_BASILICA_GPU_COUNT}"
            )
        print("\n[pvp] Starting pair evaluation")
        print(model_dump_pretty({
            "hotkey_a": hotkey_a,
            "hotkey_b": hotkey_b,
            "environment_names": environment_names,
            "gpu_count": gpu_count,
        }))
        result = await original_pair_eval(*args, **kwargs)
        print("\n[pvp] Raw pair evaluation result")
        print(model_dump_pretty(result))
        timing_tracker.record_duration_ending_now(
            label="pvp_pair",
            seconds=float(result.metadata.wall_time_seconds),
            source="PvP evaluator metadata.wall_time_seconds",
        )
        return result

    async def logged_individual_eval(*args: Any, **kwargs: Any) -> Any:
        miners = kwargs.get("miners")
        environment_name = kwargs.get("environment_name")
        gpu_count = kwargs.get("gpu_count", args[5] if len(args) > 5 else None)
        if gpu_count != scoring.cts.INDIVIDUAL_BASILICA_GPU_COUNT:
            raise RuntimeError(
                f"Individual eval requested gpu_count={gpu_count}; "
                f"expected {scoring.cts.INDIVIDUAL_BASILICA_GPU_COUNT}"
            )
        print("\n[individual] Starting evaluation")
        print(model_dump_pretty({
            "miners": miners,
            "environment_name": environment_name,
            "gpu_count": gpu_count,
        }))
        result = await original_individual_eval(*args, **kwargs)
        print("\n[individual] Raw evaluation result")
        print(model_dump_pretty(result))
        return result

    scoring.run_evaluation_pvp_pair = logged_pair_eval
    scoring.run_evaluation_individual = logged_individual_eval


async def run(args: argparse.Namespace) -> None:
    if not os.getenv("BASILICA_API_KEY"):
        raise SystemExit("BASILICA_API_KEY is not set. Export it before running this live Basilica test.")

    namespace, miners = resolve_miners(args.miner, args.hf_namespace)
    if len(miners) < 2:
        raise SystemExit("At least two --miner entries are required because PvP uses pairwise evaluation.")

    scoring.cts.RAYONLABS_HF_USERNAME = namespace

    timing_tracker = EvaluationTimingTracker()
    store = InMemoryTournamentStore()
    install_in_memory_patches(
        store=store,
        miners=miners,
        seed=args.seed,
        skip_hf_repo_check=args.skip_hf_repo_check,
        poll_interval_seconds=args.poll_interval_seconds,
        timing_tracker=timing_tracker,
    )
    install_result_logging(timing_tracker)

    task = build_task(args)
    fake_config = SimpleNamespace(psql_db=object())
    fake_miners = [SimpleNamespace(hotkey=miner.hotkey) for miner in miners]

    print("\n[config] Starting process_miners_pool mixed environment evaluation")
    print(
        model_dump_pretty(
            {
                "task_id": task.task_id,
                "base_model": task.model_id,
                "model_params_count": task.model_params_count,
                "hf_namespace": namespace,
                "miners": [
                    {
                        "hotkey": miner.hotkey,
                        "repo": f"{namespace}/{miner.expected_repo_name}",
                    }
                    for miner in miners
                ],
                "environments": [env.value for env in task.environment_names],
                "expected_gpu_counts": {
                    "pvp_pair": scoring.cts.PVP_BASILICA_GPU_COUNT,
                    "individual": scoring.cts.INDIVIDUAL_BASILICA_GPU_COUNT,
                },
                "seed": args.seed,
            }
        )
    )

    results = await scoring.process_miners_pool(
        miners=fake_miners,
        task=task,
        config=fake_config,
        num_gpus=args.num_gpus,
    )

    print("\n[results] Final process_miners_pool results")
    print(model_dump_pretty(results))

    print("\n[store] Persisted in-memory PvP rows")
    print(model_dump_pretty(list(store.pvp_rows.values())))
    print("\n[store] Persisted in-memory individual rows")
    print(model_dump_pretty(list(store.individual_rows.values())))

    print("\n[timing] Total evaluation time excluding Basilica startup")
    print(model_dump_pretty(timing_tracker.summary()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live Basilica test for process_miners_pool with both intercode "
            "and liars_dice environment evaluation."
        )
    )
    parser.add_argument(
        "--miner",
        action="append",
        required=True,
        help=(
            "Miner repo to evaluate. Accepts namespace/repo, repo, HOTKEY=repo, "
            "or HOTKEY=namespace/repo. Pass at least two."
        ),
    )
    parser.add_argument("--hf-namespace", help="HF namespace used by process_miners_pool when constructing repos.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model used as the original model.")
    parser.add_argument(
        "--model-params-count",
        type=int,
        default=DEFAULT_MODEL_PARAMS_COUNT,
        help="Parameter count recorded on the synthetic task; eval deploy GPU counts are fixed by validator constants.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Environment evaluation seed.")
    parser.add_argument("--task-id", help="Optional UUID to use for the synthetic task.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="process_miners_pool num_gpus argument. Tournament env evals ignore this and use fixed deploy counts.",
    )
    parser.add_argument(
        "--skip-hf-repo-check",
        action="store_true",
        help="Bypass the Hugging Face repo_info check inside process_miners_pool.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        help="Override Basilica result polling interval for this script run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
