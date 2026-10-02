"""Text and image round 1 use one group with three matches."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from core.models.image_models import ImageModelType
from core.models.task_models import TaskType
from core.oversampled_later_models import OVERSAMPLED_LATER_MODELS
from validator.tournament import constants as t_cst
from validator.tournament import task_creator
from validator.tournament.models import Group
from validator.tournament.models import GroupRound
from validator.tournament.models import KnockoutRound
from validator.tournament.models import TournamentType
from validator.tournament.tournament_manager import organise_tournament_round


POOL_IDS = {m.model_id for m in OVERSAMPLED_LATER_MODELS}


def _make_nodes(n: int) -> list:
    nodes = []
    for i in range(n):
        node = MagicMock()
        node.hotkey = f"hk_{i:03d}"
        nodes.append(node)
    return nodes


def _group_round(num_groups: int, members_per_group: int, round_number: int = 1) -> GroupRound:
    return GroupRound(
        round_id="tourn_round_001",
        round_number=round_number,
        groups=[
            Group(member_ids=[f"miner-{g}-{m}" for m in range(members_per_group)])
            for g in range(num_groups)
        ],
    )


def _patch_text_seams(monkeypatch) -> AsyncMock:
    for name in ("_get_text_models", "_get_instruct_text_datasets", "_get_dpo_datasets"):
        monkeypatch.setattr(task_creator, name, lambda *a, **k: MagicMock())
    monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator, "_get_existing_tasks", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", AsyncMock())
    instruct_mock = AsyncMock(
        return_value=SimpleNamespace(task_id="task", task_type=TaskType.INSTRUCTTEXTTASK)
    )
    monkeypatch.setattr(task_creator, "create_synthetic_instruct_text_task", instruct_mock)
    return instruct_mock


def _patch_image_seams(monkeypatch) -> AsyncMock:
    monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator, "_get_existing_tasks", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", AsyncMock())
    image_mock = AsyncMock(return_value=SimpleNamespace(task_id="image-task", task_type=TaskType.IMAGETASK))
    monkeypatch.setattr(task_creator, "_create_single_image_task_with_retry", image_mock)
    return image_mock


def test_text_round_one_always_single_group_even_with_large_field():
    result = organise_tournament_round(
        _make_nodes(40),
        MagicMock(),
        tournament_type=TournamentType.TEXT,
        round_id="tourn_round_001",
        round_number=1,
    )
    assert isinstance(result, GroupRound)
    assert len(result.groups) == 1
    assert len(result.groups[0].member_ids) == 40


def test_image_round_one_large_field_is_single_group():
    result = organise_tournament_round(
        _make_nodes(40),
        MagicMock(),
        tournament_type=TournamentType.IMAGE,
        round_id="tourn_round_001",
        round_number=1,
    )
    assert isinstance(result, GroupRound)
    assert len(result.groups) == 1
    assert len(result.groups[0].member_ids) == 40


def test_image_round_one_small_field_is_single_group():
    result = organise_tournament_round(
        _make_nodes(10),
        MagicMock(),
        tournament_type=TournamentType.IMAGE,
        round_id="tourn_round_001",
        round_number=1,
    )
    assert isinstance(result, GroupRound)
    assert len(result.groups) == 1
    assert len(result.groups[0].member_ids) == 10


async def test_text_round_one_single_group_creates_three_tasks(monkeypatch):
    instruct_mock = _patch_text_seams(monkeypatch)

    await task_creator._create_group_text_tasks(
        _group_round(1, members_per_group=60), "tourn", MagicMock(), is_final_round=False
    )

    overrides = [call.kwargs.get("model_id_override") for call in instruct_mock.call_args_list]
    assert len(overrides) == t_cst.SMALL_TOURNAMENT_GROUP_TASKS
    picked = [override for override in overrides if override is not None]
    assert len(picked) == 1
    assert picked[0] in POOL_IDS


async def test_text_later_single_group_still_creates_one_task(monkeypatch):
    instruct_mock = _patch_text_seams(monkeypatch)

    await task_creator._create_group_text_tasks(
        _group_round(1, members_per_group=12, round_number=2),
        "tourn",
        MagicMock(),
        is_final_round=False,
    )

    assert len(instruct_mock.call_args_list) == t_cst.TEXT_TASKS_PER_GROUP


async def test_image_round_one_single_large_group_creates_three_family_tasks(monkeypatch):
    image_mock = _patch_image_seams(monkeypatch)
    monkeypatch.setattr(task_creator, "_image_models_of_type", lambda _config, model_type: model_type)

    await task_creator._create_group_image_tasks(
        _group_round(1, members_per_group=20), "tourn", MagicMock(), image_models=MagicMock()
    )

    assert [call.args[1] for call in image_mock.call_args_list] == list(t_cst.ROUND_ONE_IMAGE_MODEL_TYPES)
    assert all(
        call.kwargs["max_num_prompts"] == t_cst.ROUND_ONE_IMAGE_MAX_SYNTH_PAIRS
        for call in image_mock.call_args_list
    )


async def test_image_round_one_small_band_creates_three_tasks(monkeypatch):
    image_mock = _patch_image_seams(monkeypatch)

    await task_creator._create_group_image_tasks(
        _group_round(1, members_per_group=8), "tourn", MagicMock(), image_models=MagicMock()
    )

    assert len(image_mock.call_args_list) == t_cst.SMALL_TOURNAMENT_GROUP_TASKS


async def test_image_round_two_uses_qwen_or_krea(monkeypatch):
    round_data = KnockoutRound(
        round_id="tourn_round_002",
        round_number=2,
        pairs=[("miner-a", "miner-b")],
    )
    selected_pool = object()
    knockout_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(task_creator, "_get_image_models", lambda _keypair: MagicMock())
    monkeypatch.setattr(task_creator.random, "choice", lambda choices: choices[0])
    monkeypatch.setattr(
        task_creator,
        "_image_models_of_type",
        lambda _config, model_type: selected_pool
        if model_type == ImageModelType.QWEN_IMAGE
        else None,
    )
    monkeypatch.setattr(task_creator, "_create_knockout_image_tasks", knockout_mock)

    await task_creator.create_image_tournament_tasks(round_data, "tourn", MagicMock())

    assert t_cst.ROUND_TWO_IMAGE_MODEL_TYPES == (
        ImageModelType.QWEN_IMAGE,
        ImageModelType.KREA2,
    )
    assert knockout_mock.await_args.args[3] is selected_pool
