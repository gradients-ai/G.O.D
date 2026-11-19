"""The quant boss round: four tasks, the text boss's round-2 model freeform, its round-3 model fixed and its continuous model
freeform and fixed (no instruct, chat or DPO).

Also the continuous-chunk machinery for own-upstream compressions of a chat task on a continuous chat chunk: the creation,
hours, row-share and URL-redaction behaviour of that code.
"""

import logging
import random
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from datasets import Dataset
from tenacity import wait_none

from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import FileFormat
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.tasks import requests
from validator.tasks.datasets import constants as data_cst
from validator.tasks.datasets import preparation
from validator.tasks.models import ChatRawTask
from validator.tasks.models import InstructTextRawTask
from validator.tasks.quant_eval_set import build_quant_eval_set
from validator.tasks.synthetics import scheduler
from validator.tournament import constants as t_cst
from validator.tournament import orchestrator
from validator.tournament import task_creator
from validator.tournament.models import ANY_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import BOSS_ROUND_SEED_ROLES
from validator.tournament.models import RETIRED_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import UNCREATED_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentType
from validator.tournament.seed_tasks import SEED_CREATION_ORDER


BASE = "org/random-base"
TID = "tourn"

X_INDEX = 7


class World:
    """The creation seams: a stand-in for the DB and the content service."""

    def __init__(self, monkeypatch, params: int = 3_000_000_000):
        self.content_calls: list[dict] = []
        self.config = MagicMock()

        async def add_task(task, psql_db):
            task.task_id = uuid4()
            return task

        async def content_service(endpoint, keypair, call_params):
            self.content_calls.append(call_params)
            n = len(self.content_calls)
            return {
                "train_s3_url": f"https://minio/train-{n}?X-Amz-Signature=secret{n}",
                "test_s3_url": f"https://minio/test-{n}?X-Amz-Signature=secret{n}",
                "ds": f"chunk-{call_params['train_index']}",
            }

        monkeypatch.setattr(scheduler, "add_task", add_task)
        monkeypatch.setattr(scheduler, "get_model_num_params", lambda model_id: params)
        monkeypatch.setattr(scheduler, "call_content_service", content_service)
        monkeypatch.setattr(scheduler.create_own_upstream_quant_task.retry, "wait", wait_none())

    async def c_task(self) -> ChatRawTask:
        """A chat upstream on the base, with its own fetch of the chunk."""
        return ChatRawTask(
            task_id=uuid4(), model_id=BASE, ds="org/chat", status=TaskStatus.SUCCESS.value, is_organic=False,
            account_id=uuid4(), hours_to_complete=3.0, created_at=datetime(2026, 1, 1), file_format=FileFormat.S3,
            training_data="https://minio/c-train", test_data="https://minio/c-test", chat_template="tokenizer_default",
        )

    async def compression(self, c_task, task_type=TaskType.QUANTFREEFORMTASK, quant_format=None, train_index=X_INDEX):
        c_task.model_params_count = 3_000_000_000
        return await scheduler.create_own_upstream_quant_task(
            self.config, task_type, c_task, quant_format=quant_format, train_index=train_index, tournament_id=TID
        )


@pytest.fixture
async def world(monkeypatch):
    return World(monkeypatch)


async def test_the_quant_round_is_four_compression_tasks():
    assert t_cst.FINAL_ROUND_QUANT_TASKS == 4
    assert t_cst.FINAL_ROUND_QUANT_TASK_DISTRIBUTION == {TaskType.QUANTFREEFORMTASK: 2, TaskType.QUANTFIXEDTASK: 2}
    assert t_cst.expected_boss_round_task_count(TournamentType.QUANT) == 4
    assert not hasattr(t_cst, "CHAINED_BOSS_SLOTS")


async def test_the_continuous_compressions_are_own_upstream_on_the_chunk_with_the_upstream_s_base(world):
    c = await world.c_task()
    for task_type in QUANT_TASK_TYPES:
        fmt = None
        if task_type == TaskType.QUANTFIXEDTASK:
            fmt = scheduler.draw_quant_format(random.Random(1))
        task = await world.compression(c, task_type, fmt)

        assert task.training_start_point == TrainingStartPoint.OWN_UPSTREAM_MODEL
        assert task.upstream_task_id == c.task_id and task.status == TaskStatus.AWAITING_UPSTREAM
        assert task.model_id == c.model_id == BASE
        assert t_cst.continuous_quant_train_index(task.ds) == X_INDEX
        assert t_cst.continuous_quant_tournament(task.ds) == TID
        assert task.training_data and task.test_data and task.training_data != c.training_data  # their own fetch of the chunk
        assert task.file_format == FileFormat.S3 and isinstance(task.dataset_type, ChatTemplateDatasetType)
        assert task.carve_calibration is True
        assert t_cst.continuous_sft_lineage_from_ds(task.ds) is None


async def test_the_continuous_compression_hours_are_the_normal_quant_formula_with_no_instruct_term(world):
    c = await world.c_task()
    freeform = await world.compression(c, TaskType.QUANTFREEFORMTASK)
    fixed = await world.compression(c, TaskType.QUANTFIXEDTASK, scheduler.draw_quant_format(random.Random(1)))

    assert freeform.hours_to_complete == data_cst.QUANT_FREEFORM_EARLY_HOURS
    assert fixed.hours_to_complete == scheduler.quant_fixed_hours(3.0)  # no fine-tune hours added
    assert data_cst.QUANT_BOSS_HOURS_MULTIPLIER == {TaskType.QUANTFREEFORMTASK: 1.0, TaskType.QUANTFIXEDTASK: 1.0}
    for gone in ("is_random_base_fixed_quant_task", "_random_base_fixed_hours", "_random_base_fixed_hours_after_prep"):
        assert not hasattr(scheduler, gone)


async def test_a_replacement_keeps_the_chunk_with_fresh_urls_and_stays_own_upstream(world, monkeypatch):
    c = await world.c_task()
    original = await world.compression(c)

    async def relabelled(endpoint, keypair, params):
        return {"train_s3_url": "https://minio/t?sig=1", "test_s3_url": "https://minio/e?sig=2", "ds": "a-different-label"}

    monkeypatch.setattr(scheduler, "call_content_service", relabelled)
    fresh = await task_creator.create_new_task_of_same_type(original, world.config)

    assert fresh.ds == original.ds and t_cst.continuous_quant_train_index(fresh.ds) == X_INDEX
    assert fresh.training_start_point == TrainingStartPoint.OWN_UPSTREAM_MODEL and fresh.upstream_task_id == c.task_id
    assert fresh.training_data == "https://minio/t?sig=1" and fresh.carve_calibration is True


def _wire_release(monkeypatch, upstream) -> None:
    monkeypatch.setattr(orchestrator.task_sql, "get_task", AsyncMock(return_value=upstream))
    monkeypatch.setattr(orchestrator.task_sql, "update_task", AsyncMock())
    monkeypatch.setattr(orchestrator.task_sql, "set_starting_model_repo", AsyncMock())
    monkeypatch.setattr(orchestrator, "_training_status_of", AsyncMock(return_value="success"))
    monkeypatch.setattr(orchestrator, "_own_upstream_output_rejection", AsyncMock(return_value=None))
    monkeypatch.setattr(orchestrator.chaining, "chained_input_repo", AsyncMock(return_value="rayon/boss-c-output"))


async def test_the_own_upstream_release_refetches_the_compression_task_s_chunk_with_fresh_urls(world, monkeypatch):
    c = await world.c_task()
    task = await world.compression(c)
    old_train, old_test, hours, ds = task.training_data, task.test_data, task.hours_to_complete, task.ds
    c.training_data, c.test_data = "prepped-c-train", "prepped-c-test"
    _wire_release(monkeypatch, c)
    calls_before = len(world.content_calls)

    await orchestrator._release_own_upstream_task(task, world.config)

    assert task.status == TaskStatus.PENDING
    assert world.content_calls[calls_before:] == [{"train_index": X_INDEX}]  # the same chunk
    assert task.training_data == "https://minio/train-2?X-Amz-Signature=secret2" != old_train
    assert task.test_data == "https://minio/test-2?X-Amz-Signature=secret2" != old_test
    assert (task.ds, task.hours_to_complete) == (ds, hours)  # ds (label) and hours stay as created
    orchestrator.task_sql.set_starting_model_repo.assert_awaited_once()


async def test_the_own_upstream_release_of_a_continuous_compression_takes_the_upstream_s_held_out_rows(world, monkeypatch):
    c = await world.c_task()
    task = await world.compression(c)
    c.training_data, c.test_data = "prepped-c-train", "prepped-c-test"
    c.calibration_data = "prepped-c-calibration"
    _wire_release(monkeypatch, c)
    alert = AsyncMock()
    monkeypatch.setattr(orchestrator, "alert_seed_issue", alert)

    await orchestrator._release_own_upstream_task(task, world.config)

    assert task.status == TaskStatus.PENDING
    assert task.calibration_data == "prepped-c-calibration" and task.carve_calibration is False
    assert task.training_data == "https://minio/train-2?X-Amz-Signature=secret2"  # its data stays its own chunk
    alert.assert_not_awaited()


async def test_the_own_upstream_release_of_a_continuous_compression_on_an_upstream_with_no_held_out_rows_alerts(
    world, monkeypatch
):
    c = await world.c_task()
    task = await world.compression(c)
    c.training_data, c.test_data = "prepped-c-train", "prepped-c-test"
    _wire_release(monkeypatch, c)
    alert = AsyncMock()
    monkeypatch.setattr(orchestrator, "alert_seed_issue", alert)

    await orchestrator._release_own_upstream_task(task, world.config)

    assert task.calibration_data is None and task.carve_calibration is False  # its prep fails rather than carve
    alert.assert_awaited_once()


async def test_a_failed_chunk_refetch_holds_the_compression_task(world, monkeypatch):
    c = await world.c_task()
    task = await world.compression(c)
    old_train, old_test = task.training_data, task.test_data
    c.training_data, c.test_data = "prepped-c-train", "prepped-c-test"
    _wire_release(monkeypatch, c)
    fetch = AsyncMock(side_effect=RuntimeError("content service down"))
    monkeypatch.setattr(orchestrator, "_fetch_continuous_quant_chunk", fetch)

    await orchestrator._release_own_upstream_task(task, world.config)

    assert task.status == TaskStatus.AWAITING_UPSTREAM
    assert (task.training_data, task.test_data) == (old_train, old_test)
    orchestrator.task_sql.set_starting_model_repo.assert_not_awaited()
    orchestrator.task_sql.update_task.assert_not_awaited()


async def test_the_own_upstream_release_still_copies_an_instruct_upstream_s_data_for_its_compressions(world, monkeypatch):
    upstream = InstructTextRawTask(
        task_id=uuid4(), model_id=BASE, ds="org/ds", field_instruction="i", field_output="o",
        status=TaskStatus.SUCCESS.value, is_organic=False, account_id=uuid4(), hours_to_complete=3.0,
        created_at=datetime(2026, 1, 1), model_params_count=3_000_000_000,
        training_data="b-train", test_data="b-test",
    )
    task = await scheduler.create_own_upstream_quant_task(world.config, TaskType.QUANTFREEFORMTASK, upstream, quant_format=None)
    _wire_release(monkeypatch, upstream)

    await orchestrator._release_own_upstream_task(task, world.config)

    assert (task.training_data, task.test_data) == ("b-train", "b-test")
    assert world.content_calls == []  # the upstream instruct task's data path never touches the content service


async def test_the_own_upstream_release_copies_the_upstream_calibration_rows_and_is_not_flagged_to_carve(world, monkeypatch):
    upstream = InstructTextRawTask(
        task_id=uuid4(), model_id=BASE, ds="org/ds", field_instruction="i", field_output="o",
        status=TaskStatus.SUCCESS.value, is_organic=False, account_id=uuid4(), hours_to_complete=3.0,
        created_at=datetime(2026, 1, 1), model_params_count=3_000_000_000,
        training_data="b-train", test_data="b-test", calibration_data="b-calibration",
    )
    task = await scheduler.create_own_upstream_quant_task(world.config, TaskType.QUANTFREEFORMTASK, upstream, quant_format=None)
    _wire_release(monkeypatch, upstream)
    alert = AsyncMock()
    monkeypatch.setattr(orchestrator, "alert_seed_issue", alert)
    task.carve_calibration = True

    await orchestrator._release_own_upstream_task(task, world.config)

    assert task.calibration_data == "b-calibration" and task.carve_calibration is False
    alert.assert_not_awaited()


async def test_the_own_upstream_release_alerts_when_the_upstream_published_no_calibration_rows(world, monkeypatch):
    upstream = InstructTextRawTask(
        task_id=uuid4(), model_id=BASE, ds="org/ds", field_instruction="i", field_output="o",
        status=TaskStatus.SUCCESS.value, is_organic=False, account_id=uuid4(), hours_to_complete=3.0,
        created_at=datetime(2026, 1, 1), model_params_count=3_000_000_000,
        training_data="b-train", test_data="b-test",
    )
    task = await scheduler.create_own_upstream_quant_task(world.config, TaskType.QUANTFREEFORMTASK, upstream, quant_format=None)
    _wire_release(monkeypatch, upstream)
    alert = AsyncMock()
    monkeypatch.setattr(orchestrator, "alert_seed_issue", alert)
    task.carve_calibration = True

    await orchestrator._release_own_upstream_task(task, world.config)

    assert task.calibration_data is None and task.carve_calibration is False
    alert.assert_awaited_once()
    assert alert.await_args.args[0] == f"{task.task_id}:quant-upstream-no-calibration"
    assert "published no calibration rows" in alert.await_args.args[1]
    assert "fail data prep and be replaced" in alert.await_args.args[1]


# --- the compressions' share of one chunk -------------------------------------------------------------------------------------


def _chunk(n: int, order: list[int] | None = None) -> Dataset:
    ids = order if order is not None else list(range(n))
    return Dataset.from_dict({"id": ids, "text": [f"row {i}" for i in ids]})


def _compression_task(tournament: str, index: int = 3, task_type=TaskType.QUANTFREEFORMTASK) -> SimpleNamespace:
    return SimpleNamespace(task_id=uuid4(), task_type=task_type, ds=t_cst.continuous_quant_ds(index, tournament, "label"))


def _ids(dataset: Dataset) -> set[int]:
    return set(dataset["id"])


def test_the_selection_equals_the_per_row_hash_order_and_reads_the_table_once(monkeypatch):
    chunk = _chunk(300)
    monkeypatch.setattr(t_cst, "QUANT_CONTINUOUS_SHARE_ROWS", 100)
    task = _compression_task("t1")
    seed = t_cst.continuous_share_seed(task)
    expected_order = sorted(range(len(chunk)), key=lambda i: preparation._row_hash(seed, chunk[i]))

    reads = []
    original = Dataset.__getitem__
    monkeypatch.setattr(Dataset, "__getitem__", lambda self, key: (reads.append(key), original(self, key))[1])
    selected = preparation.select_continuous_rows(task, chunk)

    assert selected["id"] == sorted(expected_order[100:200])  # the share after the first 100 rows of the order
    assert not [key for key in reads if isinstance(key, int)]  # no single-row Arrow read


def test_the_share_is_exactly_40k_rows():
    chunk = _chunk(95_000)

    compression = preparation.select_continuous_rows(_compression_task("t1"), chunk)

    assert len(compression) == t_cst.QUANT_CONTINUOUS_SHARE_ROWS == 40_000


def test_the_freeform_and_fixed_compressions_get_identical_rows():
    chunk = _chunk(95_000)
    free = preparation.select_continuous_rows(_compression_task("t1", task_type=TaskType.QUANTFREEFORMTASK), chunk)
    fixed = preparation.select_continuous_rows(_compression_task("t1", task_type=TaskType.QUANTFIXEDTASK), chunk)

    assert _ids(free) == _ids(fixed)


def test_shares_are_reproducible_per_tournament_and_different_across_tournaments():
    chunk = _chunk(95_000)
    first = _ids(preparation.select_continuous_rows(_compression_task("t1"), chunk))

    assert first == _ids(preparation.select_continuous_rows(_compression_task("t1"), chunk))
    assert first != _ids(preparation.select_continuous_rows(_compression_task("t2"), chunk))


def test_shares_do_not_depend_on_the_row_order_of_the_file():
    ids = list(range(95_000))
    shuffled = random.Random(5).sample(ids, len(ids))

    task = _compression_task("t1")
    assert _ids(preparation.select_continuous_rows(task, _chunk(0, ids))) == _ids(
        preparation.select_continuous_rows(task, _chunk(0, shuffled))
    )


def test_a_small_chunk_gives_the_second_half_with_a_warning(caplog):
    chunk = _chunk(60_000)
    task = _compression_task("t1")
    seed = t_cst.continuous_share_seed(task)
    order = sorted(range(len(chunk)), key=lambda i: preparation._row_hash(seed, chunk[i]))

    with caplog.at_level(logging.WARNING):
        compression = preparation.select_continuous_rows(task, chunk)

    assert len(compression) == 30_000 and _ids(compression) == set(order[30_000:])
    assert "disjoint halves" in caplog.text


def test_other_tasks_are_never_split():
    chunk = _chunk(95_000)
    ordinary = SimpleNamespace(task_id=uuid4(), task_type=TaskType.CHATTASK, ds="org/ds")
    untagged_ds = "continuous-quant:3:label"  # no tournament id
    untagged = SimpleNamespace(task_id=uuid4(), task_type=TaskType.QUANTFIXEDTASK, ds=untagged_ds)
    assert preparation.select_continuous_rows(ordinary, chunk) is chunk
    assert preparation.select_continuous_rows(untagged, chunk) is chunk


def test_the_boss_round_roles_are_the_four_compression_seeds():
    assert {role.value for role in BOSS_ROUND_SEED_ROLES} == {
        "boss_r2_freeform",
        "boss_r3_fixed",
        "boss_continuous_freeform",
        "boss_continuous_fixed",
    }
    # The R2 model fixed and the R3 model freeform roles are not created; rows of them load as boss-round seeds.
    assert {role.value for role in UNCREATED_BOSS_ROUND_SEED_ROLES} == {"boss_r2_fixed", "boss_r3_freeform"}
    assert UNCREATED_BOSS_ROUND_SEED_ROLES <= ANY_BOSS_ROUND_SEED_ROLES and not UNCREATED_BOSS_ROUND_SEED_ROLES & set(
        SEED_CREATION_ORDER
    )
    assert SeedRole.BOSS_INSTRUCT not in BOSS_ROUND_SEED_ROLES  # the boss input is never attached to the round
    # The chained compressions are still read as boss-round seeds, but are not counted for a new round.
    assert RETIRED_BOSS_ROUND_SEED_ROLES == {SeedRole.BOSS_CHAINED_FREEFORM, SeedRole.BOSS_CHAINED_FIXED}
    assert ANY_BOSS_ROUND_SEED_ROLES == BOSS_ROUND_SEED_ROLES | UNCREATED_BOSS_ROUND_SEED_ROLES | RETIRED_BOSS_ROUND_SEED_ROLES
    assert not RETIRED_BOSS_ROUND_SEED_ROLES & BOSS_ROUND_SEED_ROLES


async def test_a_quant_boss_round_with_no_seeds_raises(monkeypatch):
    monkeypatch.setattr(task_creator.seeds_sql, "get_active_seed_tasks", AsyncMock(return_value=[]))

    with pytest.raises(task_creator.MissingBossRoundSeeds) as error:
        await task_creator._create_quant_boss_round_tasks(TID, "round_4", MagicMock())

    assert sorted(error.value.roles) == sorted(role.value for role in BOSS_ROUND_SEED_ROLES)


def _boss_seeds(roles=BOSS_ROUND_SEED_ROLES) -> dict[SeedRole, TournamentSeedTask]:
    return {
        role: TournamentSeedTask(tournament_id=TID, role=role, attempt=1, task_id=uuid4(), status=SeedStatus.TRAINED)
        for role in roles
    }


def _attach_world(monkeypatch, seeds):
    task_types = {
        role: TaskType.QUANTFREEFORMTASK if role.value.endswith("freeform") else TaskType.QUANTFIXEDTASK
        for role in ANY_BOSS_ROUND_SEED_ROLES
    }
    tasks = {seed.task_id: SimpleNamespace(task_id=seed.task_id, task_type=task_types[role]) for role, seed in seeds.items()}
    registered, attached = [], []

    async def register(task, tournament_id, round_id, config, pair_id=None):
        registered.append(task)

    async def mark_attached(task_id, round_id, _db):
        attached.append(task_id)

    monkeypatch.setattr(task_creator.seeds_sql, "get_active_seed_tasks", AsyncMock(return_value=list(seeds.values())))
    monkeypatch.setattr(task_creator.seeds_sql, "get_latest_seed_tasks", AsyncMock(return_value=list(seeds.values())))
    monkeypatch.setattr(task_creator.seeds_sql, "mark_seed_attached", mark_attached)
    monkeypatch.setattr(task_creator.task_sql, "get_task", AsyncMock(side_effect=lambda task_id, _db: tasks[task_id]))
    monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", register)
    return registered, attached


async def test_the_quant_boss_round_attaches_exactly_the_four_compression_seeds(monkeypatch):
    seeds = _boss_seeds()
    registered, attached = _attach_world(monkeypatch, seeds)

    returned = await task_creator._create_quant_boss_round_tasks(TID, "round_4", MagicMock())

    assert len(registered) == 4 and len(returned) == 4
    assert [task.task_type for task in registered].count(TaskType.QUANTFREEFORMTASK) == 2
    assert [task.task_type for task in registered].count(TaskType.QUANTFIXEDTASK) == 2
    assert all(task.task_type in QUANT_TASK_TYPES for task in registered)
    assert set(attached) == {seeds[role].task_id for role in BOSS_ROUND_SEED_ROLES}


async def test_a_tournament_holding_rows_of_the_r2_fixed_and_r3_freeform_roles_attaches_them_too(monkeypatch):
    seeds = _boss_seeds(BOSS_ROUND_SEED_ROLES | UNCREATED_BOSS_ROUND_SEED_ROLES)
    registered, attached = _attach_world(monkeypatch, seeds)

    returned = await task_creator._create_quant_boss_round_tasks(TID, "round_4", MagicMock())

    assert len(registered) == 6 and len(returned) == 6 and set(attached) == {seed.task_id for seed in seeds.values()}


async def test_the_final_round_dispatch_returns_the_four_task_ids_and_never_an_instruct_or_chat_task(monkeypatch):
    seeds = _boss_seeds()
    _attach_world(monkeypatch, seeds)
    round_data = SimpleNamespace(round_id="round_4")

    task_ids = await task_creator.create_quant_tournament_tasks(round_data, TID, MagicMock(), is_final_round=True)

    assert sorted(task_ids) == sorted(str(seeds[role].task_id) for role in BOSS_ROUND_SEED_ROLES)


def _rows(n: int) -> list[dict]:
    return [{"i": i} for i in range(n)]


def test_capped_eval_rows_pass_through_at_or_under_the_cap():
    for n in (0, 150, data_cst.MAX_TEST_DATA_POINTS):
        rows = _rows(n)
        assert requests._capped_eval_rows(rows, "any-ds") is rows  # R2/R3 and ordinary rows are untouched


def test_capped_eval_rows_sample_a_seeded_subset_in_original_order():
    rows = _rows(5 * data_cst.MAX_TEST_DATA_POINTS)

    kept = requests._capped_eval_rows(rows, "continuous-quant:7:chunk-7")

    assert len(kept) == data_cst.MAX_TEST_DATA_POINTS
    assert [row["i"] for row in kept] == sorted(row["i"] for row in kept) and len({row["i"] for row in kept}) == len(kept)
    assert kept == requests._capped_eval_rows(rows, "continuous-quant:7:chunk-7")  # same ds, same rows: slots 4 and 5 agree
    assert kept != requests._capped_eval_rows(rows, "continuous-quant:8:chunk-8")


def test_the_eval_row_seed_is_the_chunk_index_not_the_label():
    rows = _rows(5 * data_cst.MAX_TEST_DATA_POINTS)
    seed = requests._eval_rows_seed

    first = t_cst.continuous_quant_ds(7, TID, "continuous-sft-110202f75206")
    relabelled = t_cst.continuous_quant_ds(7, TID, "continuous-sft-074ddce8366f")
    assert seed(first) == seed(relabelled)
    assert requests._capped_eval_rows(rows, seed(first)) == requests._capped_eval_rows(rows, seed(relabelled))
    other_chunk = seed(t_cst.continuous_quant_ds(8, TID, "x"))
    assert requests._capped_eval_rows(rows, seed(first)) != requests._capped_eval_rows(rows, other_chunk)
    assert seed("org/ordinary-ds") == "org/ordinary-ds"


# --- the format path with the exact dataset_type used here ----------------------------------------------------------


class _FakeChatTokenizer:
    def render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        text = "\x01" + "".join(f"<{m['role']}>{m['content']}</>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        return [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]


class _FakeTokenizer:
    vocab_size = 256
    context_len = None
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self):
        self.chat_templates: list[str | None] = []

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return ([1] if add_special_tokens else []) + [ord(c) % 250 for c in text]

    def chat_labeller(self, chat_template: str | None) -> _FakeChatTokenizer:
        self.chat_templates.append(chat_template)
        return _FakeChatTokenizer()


def test_sharegpt_rows_build_an_eval_set_with_the_tasks_dataset_type():
    dataset_type = scheduler._continuous_quant_dataset_type()
    rows = [
        {"conversations": [{"from": "user", "value": f"question {i}"}, {"from": "assistant", "value": f"answer {i}"}]}
        for i in range(5)
    ]
    tokenizer = _FakeTokenizer()

    build = build_quant_eval_set(rows, dataset_type, tokenizer, 4096)

    assert tokenizer.chat_templates == ["tokenizer_default"]
    assert len(build.examples) == 5 and build.n_skipped_empty == build.n_skipped_render == 0
    assert all(any(label != -100 for label in example.labels) for example in build.examples)  # the answers are supervised


# --- no presigned URL in logs or errors ---------------------------------------------------------------------------


async def test_no_url_reaches_the_logs_during_creation(monkeypatch, caplog):
    world = World(monkeypatch)
    messages = []
    for module in (scheduler, task_creator):
        monkeypatch.setattr(module.logger, "info", lambda m, *a, **k: messages.append(str(m)))
        monkeypatch.setattr(module.logger, "warning", lambda m, *a, **k: messages.append(str(m)))
        monkeypatch.setattr(module.logger, "error", lambda m, *a, **k: messages.append(str(m)))
    with caplog.at_level(logging.DEBUG):
        c = await world.c_task()
        task = await world.compression(c)
        await task_creator.create_new_task_of_same_type(task, world.config)

    blob = "\n".join(messages) + caplog.text
    assert "X-Amz-Signature" not in blob and "https://minio" not in blob and "secret" not in blob


@pytest.mark.parametrize(
    "response",
    [None, "not-a-dict", {"train_s3_url": "https://minio/t?X-Amz-Signature=s3cr3t"}, {"test_s3_url": "https://minio/e?sig=s3cr3t"}],
)
async def test_malformed_responses_raise_without_a_url_in_the_error(monkeypatch, response):
    World(monkeypatch)
    monkeypatch.setattr(scheduler, "call_content_service", AsyncMock(return_value=response))

    with pytest.raises(ValueError) as error:
        await scheduler._fetch_continuous_quant_chunk(MagicMock(), 3)

    assert "s3cr3t" not in str(error.value) and "https://" not in str(error.value)


def test_the_latest_role_check_lists_every_seed_role():
    """Every role a row may carry, the ones nothing creates included, passes the newest tournament_seed_tasks CHECK."""
    migrations = Path(__file__).parents[3] / "validator" / "db" / "migrations"
    latest = sorted(path for path in migrations.glob("*.sql") if "tournament_seed_tasks_role_check" in path.read_text())[-1]
    up = latest.read_text().split("-- migrate:down")[0]

    assert set(re.findall(r"'([a-z0-9_]+)'", up)) == {role.value for role in SeedRole}
