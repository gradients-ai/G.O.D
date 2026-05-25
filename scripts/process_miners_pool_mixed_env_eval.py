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
            status=PvPStatus.COMPLETE,
        )

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
) -> None:
    repo_by_hotkey = {miner.hotkey: miner.expected_repo_name for miner in miners}

    async def get_expected_repo_name(_task_id: UUID, hotkey: str, _psql_db: Any) -> str | None:
        return repo_by_hotkey.get(hotkey)

    async def get_env_task_eval_seed(_task_id: UUID, _psql_db: Any) -> int:
        return seed

    async def load_eval_pair_state_for_models(
        _task_id: UUID | None,
        _psql_db: Any,
        _models: list[str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        return {}, {}

    async def persist_deployment_ids_for_repo(*_args: Any, **_kwargs: Any) -> None:
        return None

    scoring.get_expected_repo_name = get_expected_repo_name
    scoring.get_env_task_eval_seed = get_env_task_eval_seed
    scoring.tournament_sql.get_pvp_pair_results = store.get_pvp_pair_results
    scoring.tournament_sql.ensure_pvp_pairs_exist = store.ensure_pvp_pairs_exist
    scoring.tournament_sql.save_pvp_pair_result = store.save_pvp_pair_result
    scoring.tournament_sql.increment_pvp_pair_attempts = store.increment_pvp_pair_attempts
    scoring.tournament_sql.ensure_individual_scores_exist = store.ensure_individual_scores_exist
    scoring.tournament_sql.get_individual_scores = store.get_individual_scores
    scoring.tournament_sql.save_individual_score = store.save_individual_score
    scoring.tournament_sql.increment_individual_score_attempts = store.increment_individual_score_attempts
    docker_evaluation.load_eval_pair_state_for_models = load_eval_pair_state_for_models
    basilica_eval.persist_deployment_ids_for_repo = persist_deployment_ids_for_repo

    if skip_hf_repo_check:
        scoring.HfApi = FakeHfApi

    if poll_interval_seconds is not None:
        original_poll = basilica_eval._poll_basilica_result

        async def patched_poll(
            deployment: Any,
            repo: str,
            eval_logger: Any,
            poll_interval_seconds: int = poll_interval_seconds,
            max_poll_seconds: int | None = None,
        ) -> dict | str:
            kwargs: dict[str, Any] = {"poll_interval_seconds": poll_interval_seconds}
            if max_poll_seconds is not None:
                kwargs["max_poll_seconds"] = max_poll_seconds
            return await original_poll(deployment, repo, eval_logger=eval_logger, **kwargs)

        basilica_eval._poll_basilica_result = patched_poll
        docker_evaluation._poll_basilica_result = patched_poll


def install_result_logging() -> None:
    original_group_eval = scoring.run_evaluation_pvp_group
    original_individual_eval = scoring.run_evaluation_individual

    async def logged_group_eval(*args: Any, **kwargs: Any) -> Any:
        participants = kwargs.get("participants", args[0] if args else [])
        environment_names = kwargs.get("environment_names", [])
        print("\n[pvp] Starting group evaluation")
        print(model_dump_pretty({"participants": participants, "environment_names": environment_names}))
        result = await original_group_eval(*args, **kwargs)
        print("\n[pvp] Raw group evaluation result")
        print(model_dump_pretty(result))
        return result

    async def logged_individual_eval(*args: Any, **kwargs: Any) -> Any:
        miners = kwargs.get("miners")
        environment_name = kwargs.get("environment_name")
        print("\n[individual] Starting evaluation")
        print(model_dump_pretty({"miners": miners, "environment_name": environment_name}))
        result = await original_individual_eval(*args, **kwargs)
        print("\n[individual] Raw evaluation result")
        print(model_dump_pretty(result))
        return result

    scoring.run_evaluation_pvp_group = logged_group_eval
    scoring.run_evaluation_individual = logged_individual_eval


async def run(args: argparse.Namespace) -> None:
    if not os.getenv("BASILICA_API_KEY"):
        raise SystemExit("BASILICA_API_KEY is not set. Export it before running this live Basilica test.")

    namespace, miners = resolve_miners(args.miner, args.hf_namespace)
    if len(miners) < 2:
        raise SystemExit("At least two --miner entries are required because liars_dice uses PvP group evaluation.")

    scoring.cts.RAYONLABS_HF_USERNAME = namespace

    store = InMemoryTournamentStore()
    install_in_memory_patches(
        store=store,
        miners=miners,
        seed=args.seed,
        skip_hf_repo_check=args.skip_hf_repo_check,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    install_result_logging()

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
        help="Parameter count used by tournament GPU sizing.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Environment evaluation seed.")
    parser.add_argument("--task-id", help="Optional UUID to use for the synthetic task.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="process_miners_pool num_gpus argument. Tournament envs size GPUs from model params.",
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
