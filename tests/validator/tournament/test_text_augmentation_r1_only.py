"""Quant tournaments augment the input seeds only; every other quant task plays its base unmodified.

maybe_get_augmentation_config always returns a config here, so any creation path that still rolls
augmentation shows up as a non-None augmentation_config.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from core.models.dataset_models import InstructTextDatasetType
from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from tests.validator.tournament.quant_source_fakes import FakeQuantSourceStore
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.synthetics import scheduler
from validator.tournament import constants as t_cst
from validator.tournament import task_creator
from validator.tournament.models import Group
from validator.tournament.models import GroupRound
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentSeedTask


AUG = AugmentationConfig(
    aug_type=AugmentationType.GAUSSIAN_NOISE, scope=AugmentationScope.ALL_LAYERS, seed=7, intensity=0.1
)
DATASET = SimpleNamespace(
    dataset_id="ds",
    num_rows=40_000,
    num_bytes_parquet_files=1,
    dpo_prompt_column="prompt",
    dpo_accepted_column="chosen",
    dpo_rejected_column="rejected",
)
DATASET_TYPE = InstructTextDatasetType(field_instruction="q", field_output="a")


async def _models():
    while True:
        yield "org/base"


def _with_id(task, psql_db):
    task.task_id = uuid4()
    return task


@pytest.fixture
def seams(monkeypatch):
    """Real task construction with augmentation always on offer; persistence, pools and the hub faked."""
    monkeypatch.setattr(scheduler, "add_task", AsyncMock(side_effect=_with_id))
    monkeypatch.setattr(scheduler, "get_model_num_params", lambda model_id: 1_500_000_000)
    monkeypatch.setattr(scheduler, "maybe_get_augmentation_config", lambda task_type: AUG)
    monkeypatch.setattr(scheduler, "get_dataset", AsyncMock(return_value=DATASET))
    columns = SimpleNamespace(field_instruction="q", field_input=None, field_output="a")
    monkeypatch.setattr(scheduler, "_get_columns_for_instruct_dataset", AsyncMock(return_value=columns))
    monkeypatch.setattr(task_creator, "_get_text_models", lambda *a, **k: _models())
    monkeypatch.setattr(task_creator, "_get_instruct_text_datasets", lambda *a, **k: MagicMock())
    monkeypatch.setattr(task_creator, "_get_dpo_datasets", lambda *a, **k: MagicMock())
    monkeypatch.setattr(task_creator, "filter_quant_compatible", lambda models: models)
    chunk = {"train_s3_url": "https://minio/train?sig=1", "test_s3_url": "https://minio/test?sig=2"}
    monkeypatch.setattr(scheduler, "call_content_service", AsyncMock(return_value=chunk))
    monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator, "_get_existing_tasks", AsyncMock(return_value=[]))
    register = AsyncMock()
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", register)
    return register


def _quant_fields(**overrides) -> dict:
    fields = dict(
        task_id=uuid4(),
        model_id="org/r1-winner",
        ds="ds-r1",
        dataset_type=DATASET_TYPE,
        status="prep_task_failure",
        is_organic=False,
        account_id=uuid4(),
        hours_to_complete=1.75,
        created_at=datetime(2026, 1, 1),
        model_params_count=1_543_714_304,
    )
    fields.update(overrides)
    return fields


def _fixed_task(**overrides) -> QuantFixedRawTask:
    return QuantFixedRawTask(**_quant_fields(quant_spec=QuantFormat.W3A16_ASYM_G128.spec(), **overrides))


@pytest.mark.parametrize("round_number", [1, 2, 3])
async def test_miner_round_task_never_augments(seams, monkeypatch, round_number):
    """R1-R3 chain on a text seed task (which may carry the augmentation): the round task itself rolls none."""
    role = t_cst.QUANT_ROUND_PLAN[round_number - 1].input_role
    seed = TournamentSeedTask(tournament_id="text_tourn", role=role, attempt=1, task_id=uuid4(), status=SeedStatus.TRAINING)
    FakeQuantSourceStore().install(monkeypatch).bind("tourn", "text_tourn")
    monkeypatch.setattr(task_creator.tournament_sql, "get_tournament_task_by_task_id", AsyncMock(return_value=None))
    monkeypatch.setattr(task_creator.seeds_sql, "get_active_seed_tasks", AsyncMock(return_value=[seed]))
    monkeypatch.setattr(task_creator.seeds_sql, "get_latest_seed_tasks", AsyncMock(return_value=[seed]))
    upstream = InstructTextRawTask(
        task_id=seed.task_id,
        is_organic=False,
        status=TaskStatus.TRAINING,
        model_id="org/base",
        ds="ds",
        account_id=uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 1, 1),
        field_instruction="q",
        field_output="a",
        augmentation_config=AUG,
        model_params_count=1_500_000_000,
    )
    monkeypatch.setattr(task_creator.task_sql, "get_task", AsyncMock(return_value=upstream))
    groups = [Group(member_ids=["a", "b"])]
    round_data = GroupRound(round_id=f"tourn_round_{round_number:03d}", round_number=round_number, groups=groups)

    await task_creator.create_quant_tournament_tasks(round_data, "tourn", MagicMock())

    (task,) = [call.args[0] for call in seams.call_args_list]
    assert task.augmentation_config is None


@pytest.mark.parametrize("task_type", [TaskType.QUANTFIXEDTASK, TaskType.QUANTFREEFORMTASK])
async def test_drawn_quant_task_never_augments(seams, task_type):
    """R2 and quant redraws build through create_synthetic_quant_task (the boss round's continuous compression tasks do not)."""
    task = await scheduler.create_synthetic_quant_task(
        MagicMock(), task_type, None, None, model_id_override="org/base", ds_override=("ds", DATASET_TYPE, 40_000)
    )

    assert task.augmentation_config is None


async def test_the_continuous_instruct_task_never_augments(seams):
    task = await scheduler.create_continuous_instruct_task(MagicMock(), model_id="org/base", train_index=3, tournament_id="tourn")

    assert task.augmentation_config is None


async def test_quant_replacement_never_augments(seams):
    """A chained quant task is replaced by an identical held copy, which drops the config of an older augmented one."""
    original = _fixed_task(augmentation_config=AUG, upstream_task_id=uuid4())

    task = await task_creator.create_new_task_of_same_type(original, MagicMock())

    assert task.augmentation_config is None


async def test_an_unchained_quant_task_has_no_replacement_draw(seams):
    with pytest.raises(ValueError, match="not chained"):
        await task_creator.create_new_task_of_same_type(_fixed_task(), MagicMock())
