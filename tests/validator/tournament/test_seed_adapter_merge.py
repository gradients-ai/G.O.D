"""A boss seed training that uploads a LoRA adapter: the adapter is merged into its base and the merged full model is the
seed's output. The seed is never replaced, charged or halted for the adapter.

Mock-based: the hub is a dict of repo -> file list, the trainer is the dispatch mock.
"""

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import UUID
from uuid import uuid4

import pytest

import validator.tournament.constants as t_cst
from core.constants.quant import QUANT_ADAPTER_CONFIG_FILENAME
from core.constants.quant import QUANT_MODEL_CONFIG_FILENAME
from core.merged_repo import merged_repo_name
from core.models.payload_models import ModelPrepResponse
from core.models.task_models import TaskStatus
from tests.validator.tournament.test_chained_release import _chained_fixed
from tests.validator.tournament.test_chained_release import _chained_freeform
from tests.validator.tournament.test_seed_tasks import FakeSeedStore
from tests.validator.tournament.test_seed_tasks import _instruct
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.db.sql import tournaments as tournament_sql
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tasks import quant_input_repo
from validator.tournament import chaining
from validator.tournament import orchestrator
from validator.tournament import seed_tasks
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus
from validator.tournament.orchestrator import SeedMergeLookup


TID = "tourn_adapter_merge"
CHALLENGER = "5Challenger"
MERGED = "gradients-io/merged-0123456789abcdef"
FULL_MODEL_FILES = [QUANT_MODEL_CONFIG_FILENAME, "model.safetensors"]
ADAPTER_FILES = [QUANT_ADAPTER_CONFIG_FILENAME, "adapter_model.safetensors"]
CONFIG = SimpleNamespace(psql_db=None, keypair=None, discord_url="http://discord")


def _lookup(job=None, unreachable: int = 0) -> SeedMergeLookup:
    return SeedMergeLookup.model_construct(job=job, unreachable_trainers=unreachable)


class MergeStore(FakeSeedStore):
    async def set_seed_merged_output_repo(self, task_id, repo, _db):
        for row in self.rows:
            if row.task_id == UUID(str(task_id)):
                row.merged_output_repo = repo


class World:
    """One quant tournament with one live instruct seed (the boss's training done) and the hub, trainers and DB around it."""

    def __init__(self, monkeypatch, role: SeedRole = SeedRole.BOSS_INSTRUCT, tournament_type=TournamentType.QUANT) -> None:
        self.task = _instruct(status=TaskStatus.SUCCESS, training_data="s3://bucket/train.json")
        self.store = MergeStore()
        self.store.rows.append(
            TournamentSeedTask(tournament_id=TID, role=role, task_id=self.task.task_id, status=SeedStatus.TRAINING)
        )
        self.raw_repo = f"{chaining.RAYONLABS_HF_USERNAME}/tournament-{TID}-{self.task.task_id}-{EMISSION_BURN_HOTKEY[:8]}"
        self.hub: dict[str, list[str]] = {self.raw_repo: ADAPTER_FILES, MERGED: FULL_MODEL_FILES}
        self.trained = TrainingStatus.SUCCESS.value
        self.dispatch = AsyncMock(return_value=ModelPrepResponse(augmented_model_id=MERGED))
        self.found_job = AsyncMock(return_value=SeedMergeLookup())
        self.gpus = AsyncMock(return_value=("10.0.0.1", [0, 1], "nvlink"))
        self.availability = AsyncMock()
        self.replace = AsyncMock()
        self.discord = AsyncMock()
        mp = monkeypatch
        for name in (
            "get_latest_seed_tasks",
            "get_active_seed_tasks",
            "get_failed_seed_tasks",
            "update_seed_status",
            "get_seed_task_history",
            "get_seed_task_by_task_id",
            "set_seed_merged_output_repo",
            "role_ever_attached",
            "is_task_attached_to_round",
        ):
            mp.setattr(seeds_sql, name, getattr(self.store, name))
        mp.setattr(quant_input_repo, "list_repo_files", MagicMock(side_effect=self._list_repo_files))
        mp.setattr(task_sql, "get_task", AsyncMock(side_effect=lambda task_id, _db, **_: self.task))
        mp.setattr(
            task_sql,
            "get_expected_repo_name",
            AsyncMock(
                side_effect=lambda task_id, hotkey, _db: (
                    self.raw_repo.split("/", 1)[1] if hotkey == EMISSION_BURN_HOTKEY else None
                )
            ),
        )
        mp.setattr(chaining, "shared_input_candidates", AsyncMock(return_value=[EMISSION_BURN_HOTKEY]))
        mp.setattr(
            tournament_sql,
            "get_training_status_for_task_and_hotkeys",
            AsyncMock(side_effect=lambda task_id, hotkeys, _db: {EMISSION_BURN_HOTKEY: self.trained}),
        )
        mp.setattr(
            tournament_sql,
            "get_tournaments_with_status",
            AsyncMock(
                return_value=[TournamentData(tournament_id=TID, tournament_type=tournament_type, status=TournamentStatus.ACTIVE)]
            ),
        )
        mp.setattr(seed_tasks, "_retry_missing_boss_round_seeds", AsyncMock())
        mp.setattr(seed_tasks.seeds_sql, "tournament_is_decided", AsyncMock(return_value=False))
        mp.setattr(seed_tasks, "replace_seed", self.replace)
        mp.setattr(seed_tasks, "send_to_discord", self.discord)
        mp.setattr(seed_tasks, "_SEED_ISSUE_ALERTED", set())
        mp.setattr(seed_tasks, "_SEED_HALT_ALERTED", set())
        mp.setattr(orchestrator, "_seed_merge_in_progress", set())
        mp.setattr(orchestrator, "_seed_merge_failures", {})
        mp.setattr(orchestrator, "_seed_merge_retry_after", {})
        mp.setattr(orchestrator, "_seed_merge_parked", set())
        mp.setattr(orchestrator, "_seed_merge_unknown_cycles", {})
        mp.setattr(orchestrator, "_seed_output_known_full", set())
        self.clock = [1000.0]
        mp.setattr(orchestrator, "monotonic", lambda: self.clock[0])
        self.delete = AsyncMock()
        mp.setattr(orchestrator, "delete_merged_repo", self.delete)
        mp.setattr(orchestrator, "_find_seed_merge_job", self.found_job)
        mp.setattr(orchestrator, "_check_suitable_gpus", self.gpus)
        mp.setattr(orchestrator, "get_task_gpu_requirement", MagicMock())
        mp.setattr(orchestrator, "dispatch_augmentation_and_stats", self.dispatch)
        mp.setattr(orchestrator, "_start_prep_cost_run", AsyncMock())
        mp.setattr(orchestrator, "_finish_prep_cost_run", AsyncMock())
        mp.setattr(orchestrator.tournament_sql, "update_gpu_availability", self.availability)

    def _list_repo_files(self, repo, token=None):
        if repo not in self.hub:
            raise seed_tasks.RepositoryNotFoundError(f"{repo} not found")
        return self.hub[repo]

    def wait(self, seconds: float) -> None:
        self.clock[0] += seconds

    @property
    def seed(self) -> TournamentSeedTask:
        return self.store.rows[0]

    async def merge_cycle(self) -> None:
        """One merge pass, then every merge job it started to completion."""
        await orchestrator._seed_adapter_merge_cycle(CONFIG)
        pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        await asyncio.gather(*pending)

    async def seed_cycle(self) -> None:
        await seed_tasks.process_tournament_seed_tasks(TID, CONFIG)


@pytest.fixture
def world(monkeypatch) -> World:
    return World(monkeypatch)


class TestAnAdapterOutputWaits:
    async def test_the_seed_waits_and_is_never_replaced_or_charged(self, world):
        assert await seed_tasks._evaluate_seed(world.seed, world.task, CONFIG) == (None, None)

        for _ in range(10):
            await world.seed_cycle()

        world.replace.assert_not_awaited()
        assert [(row.attempt, row.status) for row in world.store.rows] == [(1, SeedStatus.TRAINING)]
        assert await seed_tasks.seed_halt_reason(TID, None) is None
        world.discord.assert_not_awaited()

    async def test_an_input_seed_waits_the_same_way(self, monkeypatch):
        world = World(monkeypatch, role=SeedRole.INPUT_LARGE)

        assert await seed_tasks._evaluate_seed(world.seed, world.task, CONFIG) == (None, None)

    async def test_a_full_model_output_is_ready_and_is_never_merged(self, world):
        world.hub[world.raw_repo] = FULL_MODEL_FILES

        await world.seed_cycle()
        await world.merge_cycle()

        assert world.seed.status == SeedStatus.READY
        world.dispatch.assert_not_awaited()
        world.found_job.assert_not_awaited()

    async def test_a_missing_boss_repo_is_still_replaced(self, world):
        del world.hub[world.raw_repo]

        assert await seed_tasks._evaluate_seed(world.seed, world.task, CONFIG) == (
            None,
            f"{world.raw_repo} not found",
        )

    async def test_an_output_without_a_config_is_still_replaced(self, world):
        world.hub[world.raw_repo] = ["model.safetensors"]

        status, reason = await seed_tasks._evaluate_seed(world.seed, world.task, CONFIG)

        assert status is None and "config.json" in reason

    async def test_an_output_with_missing_shards_is_still_replaced(self, world, monkeypatch, tmp_path):
        index = tmp_path / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"w": "model-00001-of-00002.safetensors"}}))
        monkeypatch.setattr(quant_input_repo, "hf_hub_download", MagicMock(return_value=str(index)))
        world.hub[world.raw_repo] = [QUANT_MODEL_CONFIG_FILENAME, "model.safetensors.index.json"]

        status, reason = await seed_tasks._evaluate_seed(world.seed, world.task, CONFIG)

        assert status is None and "shard" in reason

    async def test_a_defective_output_is_replaced_by_the_cycle(self, world):
        world.hub[world.raw_repo] = ["model.safetensors"]

        await world.seed_cycle()

        world.replace.assert_awaited_once()

    async def test_scoring_no_fine_tune_is_still_replaced(self, world, monkeypatch):
        monkeypatch.setattr(chaining, "shared_input_candidates", AsyncMock(return_value=[]))

        assert await seed_tasks._evaluate_seed(world.seed, world.task, CONFIG) == (None, "the boss's output scored no fine-tune")


class TestTheMergeJob:
    async def test_dispatched_once_with_the_raw_repo_and_no_augmentation(self, world):
        release = asyncio.Event()

        async def blocked_dispatch(**_kwargs):
            await release.wait()
            return ModelPrepResponse(augmented_model_id=MERGED)

        world.dispatch.side_effect = blocked_dispatch

        await orchestrator._seed_adapter_merge_cycle(CONFIG)
        await orchestrator._seed_adapter_merge_cycle(CONFIG)  # in progress: not dispatched twice
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(*[task for task in asyncio.all_tasks() if task is not asyncio.current_task()])

        world.dispatch.assert_called_once()
        kwargs = world.dispatch.call_args.kwargs
        assert kwargs["task_id"] == f"{world.seed.task_id}-merge"
        assert kwargs["model_id"] == world.raw_repo
        assert kwargs["augmentation_config"] is None
        assert kwargs["training_data_url"] == world.task.training_data
        assert kwargs["task_type"] == world.task.task_type
        assert (kwargs["trainer_ip"], kwargs["gpu_ids"]) == ("10.0.0.1", [0, 1])

    async def test_the_gpus_are_reserved_and_released(self, world):
        await world.merge_cycle()

        assert [call.args[2] for call in world.availability.await_args_list] == [t_cst.MODEL_PREP_GPU_RESERVE_HOURS, 0]
        assert orchestrator._seed_merge_in_progress == set()

    async def test_the_gpus_are_released_when_the_dispatch_errors(self, world):
        world.dispatch.side_effect = RuntimeError("trainer down")

        await world.merge_cycle()

        assert world.availability.await_args_list[-1].args[2] == 0
        assert orchestrator._seed_merge_in_progress == set()
        assert world.seed.merged_output_repo is None

    async def test_the_trainers_merged_repo_is_recorded_as_is(self, world):
        await world.merge_cycle()

        assert world.seed.merged_output_repo == MERGED

    async def test_a_recovered_finished_job_is_used_without_a_dispatch(self, world):
        world.found_job.return_value = _lookup(
            SimpleNamespace(status=TaskStatus.SUCCESS, result=ModelPrepResponse(augmented_model_id=MERGED))
        )

        await world.merge_cycle()

        world.dispatch.assert_not_awaited()
        assert world.seed.merged_output_repo == MERGED
        world.found_job.assert_awaited_once()
        assert world.found_job.await_args.args[0] == f"{world.seed.task_id}-merge"

    async def test_a_job_still_running_on_a_trainer_is_not_dispatched_again(self, world):
        world.found_job.return_value = _lookup(SimpleNamespace(status=TaskStatus.TRAINING, result=None))

        await world.merge_cycle()

        world.dispatch.assert_not_awaited()
        world.gpus.assert_not_awaited()

    async def test_nothing_runs_before_the_boss_training_succeeded(self, world):
        world.trained = TrainingStatus.TRAINING.value

        await world.merge_cycle()

        world.dispatch.assert_not_awaited()

    async def test_without_free_gpus_it_waits_without_a_failure(self, world):
        world.gpus.return_value = None

        await world.merge_cycle()

        world.dispatch.assert_not_awaited()
        assert orchestrator._seed_merge_failures == {}

    @pytest.mark.parametrize("tournament_type", [TournamentType.TEXT, TournamentType.IMAGE])
    async def test_text_and_image_tournaments_never_merge(self, monkeypatch, tournament_type):
        world = World(monkeypatch, tournament_type=tournament_type)

        await world.merge_cycle()

        world.dispatch.assert_not_awaited()
        world.found_job.assert_not_awaited()

    async def test_a_seed_with_a_merge_is_not_merged_again(self, world):
        world.seed.merged_output_repo = MERGED

        await world.merge_cycle()

        world.dispatch.assert_not_awaited()


class TestAFailedMerge:
    @pytest.mark.parametrize(
        "merged_files",
        [ADAPTER_FILES, ["model.safetensors"], None],
        ids=["still an adapter", "no config", "repo missing"],
    )
    async def test_a_merged_repo_that_is_not_a_full_model_is_not_persisted(self, world, merged_files):
        if merged_files is None:
            del world.hub[MERGED]
        else:
            world.hub[MERGED] = merged_files

        await world.merge_cycle()

        assert world.seed.merged_output_repo is None
        assert orchestrator._seed_merge_failures == {str(world.seed.task_id): 1}

    async def test_no_merged_repo_in_the_response_is_a_failure(self, world):
        world.dispatch.return_value = ModelPrepResponse(augmented_model_id=None)

        await world.merge_cycle()

        assert world.seed.merged_output_repo is None
        assert orchestrator._seed_merge_failures == {str(world.seed.task_id): 1}

    async def test_a_failed_job_is_a_failure(self, world):
        world.dispatch.return_value = None

        await world.merge_cycle()

        assert orchestrator._seed_merge_failures == {str(world.seed.task_id): 1}

    async def test_a_recovered_result_that_does_not_verify_is_re_run_after_the_cooldown(self, world):
        world.hub[MERGED] = ["model.safetensors"]
        world.found_job.return_value = _lookup(
            SimpleNamespace(status=TaskStatus.SUCCESS, result=ModelPrepResponse(augmented_model_id=MERGED))
        )

        await world.merge_cycle()

        world.dispatch.assert_not_awaited()
        assert orchestrator._seed_merge_failures[str(world.seed.task_id)] == 1

        world.wait(seed_tasks.SEED_ADAPTER_MERGE_COOLDOWN_BASE_SECONDS + 1)
        world.found_job.return_value = _lookup()
        await world.merge_cycle()

        world.dispatch.assert_awaited_once()
        assert orchestrator._seed_merge_failures[str(world.seed.task_id)] == 2

    async def test_the_alert_goes_out_once_after_the_threshold_and_the_merge_keeps_retrying(self, world):
        world.hub[MERGED] = ["model.safetensors"]
        threshold = seed_tasks.SEED_ADAPTER_MERGE_FAILURES_BEFORE_ALERT
        assert threshold + 2 < seed_tasks.SEED_ADAPTER_MERGE_MAX_FAILURES

        for cycle in range(threshold + 2):
            await world.merge_cycle()
            world.wait(seed_tasks.SEED_ADAPTER_MERGE_COOLDOWN_MAX_SECONDS + 1)
            await world.seed_cycle()
            assert (world.discord.await_count == 0) == (cycle + 1 < threshold)

        world.discord.assert_awaited_once()
        assert str(world.seed.task_id) in world.discord.await_args.kwargs["message"]
        assert world.dispatch.await_count == threshold + 2
        world.replace.assert_not_awaited()
        assert world.seed.status == SeedStatus.TRAINING
        assert await seed_tasks.seed_halt_reason(TID, None) is None

    async def test_a_later_success_clears_the_failure_count(self, world):
        world.dispatch.return_value = None
        await world.merge_cycle()
        world.wait(seed_tasks.SEED_ADAPTER_MERGE_COOLDOWN_BASE_SECONDS + 1)
        world.dispatch.return_value = ModelPrepResponse(augmented_model_id=MERGED)

        await world.merge_cycle()

        assert world.seed.merged_output_repo == MERGED
        assert orchestrator._seed_merge_failures == {}


class TestTheCooldownAndTheCap:
    @pytest.fixture
    def failing(self, world):
        world.dispatch.return_value = None
        return world

    async def test_no_dispatch_inside_the_cooldown_and_a_dispatch_after_it(self, failing):
        base = seed_tasks.SEED_ADAPTER_MERGE_COOLDOWN_BASE_SECONDS
        await failing.merge_cycle()
        failing.wait(base - 1)
        await failing.merge_cycle()
        assert failing.dispatch.await_count == 1

        failing.wait(2)
        await failing.merge_cycle()
        assert failing.dispatch.await_count == 2

    async def test_the_cooldown_doubles_per_failure_up_to_the_max(self, failing):
        base = seed_tasks.SEED_ADAPTER_MERGE_COOLDOWN_BASE_SECONDS
        cap = seed_tasks.SEED_ADAPTER_MERGE_COOLDOWN_MAX_SECONDS
        key = str(failing.seed.task_id)
        for failures in range(1, 5):
            await failing.merge_cycle()
            assert orchestrator._seed_merge_retry_after[key] - failing.clock[0] == min(base * 2 ** (failures - 1), cap)
            failing.wait(cap + 1)

    async def test_the_seed_is_parked_after_the_cap_with_one_alert(self, failing):
        cap = seed_tasks.SEED_ADAPTER_MERGE_MAX_FAILURES
        for _ in range(cap + 3):
            await failing.merge_cycle()
            failing.wait(seed_tasks.SEED_ADAPTER_MERGE_COOLDOWN_MAX_SECONDS + 1)

        assert failing.dispatch.await_count == cap
        assert str(failing.seed.task_id) in orchestrator._seed_merge_parked
        messages = [call.kwargs["message"] for call in failing.discord.await_args_list]
        assert len([m for m in messages if "PARKED" in m]) == 1
        assert "restart" in [m for m in messages if "PARKED" in m][0]
        failing.replace.assert_not_awaited()
        assert failing.seed.status == SeedStatus.TRAINING
        assert await seed_tasks.seed_halt_reason(TID, None) is None


class TestARejectedMergedRepo:
    async def _run_one(self, world, files):
        if files is None:
            del world.hub[MERGED]
        else:
            world.hub[MERGED] = files
        await world.merge_cycle()

    @pytest.mark.parametrize("files", [ADAPTER_FILES, ["model.safetensors"]], ids=["adapter", "no config"])
    async def test_an_existing_repo_that_is_not_a_full_model_is_deleted(self, world, files):
        await self._run_one(world, files)

        world.delete.assert_awaited_once_with(MERGED, world.raw_repo, world.task.model_id)
        assert orchestrator._seed_merge_failures == {str(world.seed.task_id): 1}

    async def test_a_missing_repo_is_not_deleted(self, world):
        await self._run_one(world, None)

        world.delete.assert_not_awaited()

    async def test_a_hub_error_is_not_deleted(self, world, monkeypatch):
        monkeypatch.setattr(orchestrator, "full_model_rejection", AsyncMock(side_effect=RuntimeError("hub 503")))

        await world.merge_cycle()

        world.delete.assert_not_awaited()
        assert orchestrator._seed_merge_failures == {str(world.seed.task_id): 1}

    async def test_a_failed_delete_is_counted_not_raised(self, world):
        world.delete.side_effect = RuntimeError("403")

        await self._run_one(world, ADAPTER_FILES)

        assert orchestrator._seed_merge_failures == {str(world.seed.task_id): 1}
        assert world.seed.merged_output_repo is None

    @pytest.mark.parametrize("which", ["raw", "base", "foreign name"])
    async def test_the_real_helper_never_deletes_the_base_the_adapter_or_a_foreign_name(self, monkeypatch, which):
        api = MagicMock()
        monkeypatch.setattr(quant_input_repo, "HfApi", MagicMock(return_value=api))
        raw, base = "gradients-io-tournaments/tournament-x", "org/base-model"
        target = {"raw": raw, "base": base, "foreign name": "gradients-io/my-model"}[which]

        with pytest.raises(ValueError):
            await quant_input_repo.delete_merged_repo(target, raw, base)

        api.delete_repo.assert_not_called()

    async def test_the_real_helper_deletes_the_repo_the_merge_would_publish(self, monkeypatch):
        api = MagicMock()
        monkeypatch.setattr(quant_input_repo, "HfApi", MagicMock(return_value=api))
        raw = "gradients-io-tournaments/tournament-x"

        await quant_input_repo.delete_merged_repo(merged_repo_name(raw, "gradients-io"), raw, "org/base-model")

        api.delete_repo.assert_called_once_with(merged_repo_name(raw, "gradients-io"), repo_type="model")


class TestAnUnreachableTrainer:
    async def test_no_dispatch_while_a_trainer_could_not_be_asked_then_dispatch_after_n_cycles(self, world):
        world.found_job.return_value = _lookup(None, unreachable=1)
        unknown = seed_tasks.SEED_ADAPTER_MERGE_UNKNOWN_CYCLES_BEFORE_DISPATCH

        for _ in range(unknown - 1):
            await world.merge_cycle()
        world.dispatch.assert_not_awaited()
        world.gpus.assert_not_awaited()

        await world.merge_cycle()
        world.dispatch.assert_awaited_once()

    async def test_a_job_found_resets_the_unknown_count(self, world):
        world.found_job.return_value = _lookup(None, unreachable=1)
        await world.merge_cycle()
        assert orchestrator._seed_merge_unknown_cycles == {str(world.seed.task_id): 1}

        world.found_job.return_value = _lookup(SimpleNamespace(status=TaskStatus.TRAINING, result=None), unreachable=1)
        await world.merge_cycle()

        assert orchestrator._seed_merge_unknown_cycles == {}

    async def test_the_lookup_counts_trainers_that_error_but_not_a_404(self, monkeypatch):
        trainers = [
            SimpleNamespace(trainer_ip="1.1.1.1"),
            SimpleNamespace(trainer_ip="2.2.2.2"),
            SimpleNamespace(trainer_ip="3.3.3.3"),
        ]
        monkeypatch.setattr(orchestrator.tournament_sql, "get_trainers", AsyncMock(return_value=trainers))

        class Client:
            def __init__(self, *_, **__): ...

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_): ...

            async def get(self, url):
                if "1.1.1.1" in url:
                    raise RuntimeError("timeout")
                if "2.2.2.2" in url:
                    return SimpleNamespace(status_code=404)
                return SimpleNamespace(status_code=500, raise_for_status=MagicMock(side_effect=RuntimeError("500")))

        monkeypatch.setattr(orchestrator.httpx, "AsyncClient", Client)

        lookup = await orchestrator._find_seed_merge_job("job", CONFIG)

        assert lookup.job is None
        assert lookup.unreachable_trainers == 2


class TestAKnownFullSeedIsNotPolled:
    async def test_the_hub_is_asked_once_for_a_full_model_output(self, world):
        world.hub[world.raw_repo] = FULL_MODEL_FILES
        listing = quant_input_repo.list_repo_files

        await orchestrator._seed_adapter_merge_cycle(CONFIG)
        calls_after_first = listing.call_count
        assert calls_after_first > 0
        assert str(world.seed.task_id) in orchestrator._seed_output_known_full

        for _ in range(3):
            await orchestrator._seed_adapter_merge_cycle(CONFIG)

        assert listing.call_count == calls_after_first

    async def test_a_missing_repo_is_not_remembered_as_full(self, world):
        del world.hub[world.raw_repo]

        await orchestrator._seed_adapter_merge_cycle(CONFIG)

        assert orchestrator._seed_output_known_full == set()

    async def test_an_adapter_output_is_not_remembered_as_full(self, world):
        await world.merge_cycle()

        assert orchestrator._seed_output_known_full == set()


class TestAfterTheMerge:
    @pytest.mark.parametrize("role", [SeedRole.INPUT_SMALL, SeedRole.BOSS_INSTRUCT])
    async def test_the_seed_becomes_ready_on_the_merged_repo(self, monkeypatch, role):
        world = World(monkeypatch, role=role)
        await world.seed_cycle()
        assert world.seed.status == SeedStatus.TRAINING

        await world.merge_cycle()
        await world.seed_cycle()

        assert world.seed.status == SeedStatus.READY
        world.replace.assert_not_awaited()

    async def test_the_boss_output_is_the_merged_repo_for_every_consumer(self, world):
        assert await chaining.chained_input_repo(world.task.task_id, EMISSION_BURN_HOTKEY, None) == world.raw_repo

        await world.merge_cycle()

        assert await chaining.chained_input_repo(world.task.task_id, EMISSION_BURN_HOTKEY, None) == MERGED

    async def test_only_the_boss_reads_the_merged_repo(self, world):
        await world.merge_cycle()

        assert await chaining.chained_input_repo(world.task.task_id, CHALLENGER, None) is None

    async def test_a_task_that_is_not_a_seed_reads_its_own_expected_repo(self, world):
        other = uuid4()

        repo = await chaining.chained_input_repo(other, EMISSION_BURN_HOTKEY, None)

        assert repo == world.raw_repo

    @pytest.mark.parametrize(
        ("role", "make_tasks"),
        [
            (SeedRole.INPUT_SMALL, lambda upstream: [_chained_fixed(upstream)]),
            (SeedRole.BOSS_INSTRUCT, lambda upstream: [_chained_freeform(upstream), _chained_fixed(upstream)]),
        ],
        ids=["round task on an input seed", "the two boss-round compression seeds"],
    )
    async def test_chained_tasks_are_held_then_released_on_the_merged_repo(self, monkeypatch, role, make_tasks):
        world = World(monkeypatch, role=role)
        monkeypatch.setattr(orchestrator.task_sql, "get_task", AsyncMock(return_value=world.task))
        monkeypatch.setattr(
            orchestrator.task_sql,
            "get_nodes_assigned_to_task",
            AsyncMock(return_value=[SimpleNamespace(hotkey=EMISSION_BURN_HOTKEY)]),
        )
        monkeypatch.setattr(orchestrator.task_sql, "update_task", AsyncMock())
        monkeypatch.setattr(orchestrator, "_is_non_boss_tournament_task", AsyncMock(return_value=False))
        tasks = make_tasks(world.task)

        for task in tasks:
            await orchestrator._release_chained_task(task, CONFIG)
            assert task.status == TaskStatus.AWAITING_UPSTREAM  # the raw adapter is not a full model: held, not failed

        await world.merge_cycle()

        for task in tasks:
            await orchestrator._release_chained_task(task, CONFIG)
            assert task.status == TaskStatus.PENDING
            assert task.model_id == MERGED


class TestMergedRepoName:
    @pytest.mark.parametrize(
        ("model_id", "username"),
        [("miner/adapter-xyz", "gradients-io"), ("org/lora", "my-org"), ("a/b", "x")],
    )
    def test_matches_the_trainers_original_formula(self, model_id, username):
        digest = hashlib.sha256(f"{model_id}:lora-merge".encode()).hexdigest()[:16]

        assert merged_repo_name(model_id, username) == f"{username}/merged-{digest}"

    def test_is_stable_and_keyed_on_the_source_repo(self):
        assert merged_repo_name("m/a", "u") == merged_repo_name("m/a", "u")
        assert merged_repo_name("m/a", "u") != merged_repo_name("m/b", "u")

    def test_the_trainer_entrypoint_delegates_to_it(self, monkeypatch):
        entrypoint = pytest.importorskip("trainer.model_prep.entrypoint")
        monkeypatch.setenv("HUGGINGFACE_USERNAME", "my-org")

        assert entrypoint.generate_merged_repo_name("m/a") == merged_repo_name("m/a", "my-org")
        monkeypatch.delenv("HUGGINGFACE_USERNAME")
        assert entrypoint.generate_merged_repo_name("m/a") == merged_repo_name("m/a", "gradients-io")
