from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from core.constants.environments import TrainingStartPoint
from core.models.image_models import ImageModelType
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.tournament import constants as t_cst
from validator.tournament import task_creator


def _patch_replace_seams(monkeypatch, original_task):
    monkeypatch.setattr(task_creator.task_sql, "get_task", AsyncMock(return_value=original_task))
    monkeypatch.setattr(task_creator.task_sql, "get_nodes_assigned_to_task", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator.task_sql, "detach_task_from_tournament", AsyncMock())
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", AsyncMock())


async def test_round_one_oversampled_replacement_preserves_model(monkeypatch):
    original = SimpleNamespace(
        task_id="orig-task",
        task_type=TaskType.INSTRUCTTEXTTASK,
        status=TaskStatus.PREP_TASK_FAILURE.value,
        model_id=task_creator.OVERSAMPLED_LATER_MODELS[0].model_id,
        model_params_count=2_000_000_000,
    )
    _patch_replace_seams(monkeypatch, original)

    replacement_mock = AsyncMock(return_value=SimpleNamespace(task_id="new-task", task_type=TaskType.INSTRUCTTEXTTASK))
    monkeypatch.setattr(task_creator, "_create_round_one_group_text_replacement_task", replacement_mock)

    new_task_id = await task_creator.replace_tournament_task(
        "orig-task", "tourn", "tourn_round_001", "tourn_round_001_group_001", None, MagicMock()
    )

    assert new_task_id == "new-task"
    replacement_mock.assert_awaited_once()
    assert replacement_mock.call_args.kwargs["model_id_override"] == original.model_id


async def test_round_one_non_oversampled_replacement_keeps_small_pool(monkeypatch):
    original = SimpleNamespace(
        task_id="orig-task",
        task_type=TaskType.INSTRUCTTEXTTASK,
        status=TaskStatus.PREP_TASK_FAILURE.value,
        model_id="Qwen/Qwen2.5-3B",
        model_params_count=3_000_000_000,
    )
    _patch_replace_seams(monkeypatch, original)

    replacement_mock = AsyncMock(return_value=SimpleNamespace(task_id="new-task", task_type=TaskType.INSTRUCTTEXTTASK))
    monkeypatch.setattr(task_creator, "_create_round_one_group_text_replacement_task", replacement_mock)

    await task_creator.replace_tournament_task(
        "orig-task", "tourn", "tourn_round_001", "tourn_round_001_group_001", None, MagicMock()
    )

    replacement_mock.assert_awaited_once()
    assert replacement_mock.call_args.kwargs["model_id_override"] is None


async def test_final_round_oversampled_replacement_passes_override_to_same_type(monkeypatch):
    original = SimpleNamespace(
        task_id="orig-task",
        task_type=TaskType.DPOTASK,
        status=TaskStatus.PREP_TASK_FAILURE.value,
        model_id=task_creator.OVERSAMPLED_LATER_MODELS[0].model_id,
        model_params_count=14_000_000_000,
    )
    _patch_replace_seams(monkeypatch, original)

    same_type_mock = AsyncMock(return_value=SimpleNamespace(task_id="new-task", task_type=TaskType.DPOTASK))
    monkeypatch.setattr(task_creator, "create_new_task_of_same_type", same_type_mock)

    new_task_id = await task_creator.replace_tournament_task(
        "orig-task", "tourn", "tourn_round_004", None, "tourn_round_004_pair_001", MagicMock(), is_final_round=True
    )

    assert new_task_id == "new-task"
    same_type_mock.assert_awaited_once()
    assert same_type_mock.call_args.kwargs["model_id_override"] == original.model_id


async def test_continuous_sft_replacement_recreates_the_same_lineage(monkeypatch):
    """Same lineage, same carried base; only the chunk is re-materialized. Falling through to
    create_new_task_of_same_type would drop the lineage (no CHATTASK route there) and weaken the
    win-all-continuous-SFT dethrone gate."""
    original = SimpleNamespace(
        task_id="orig-task",
        task_type=TaskType.CHATTASK,
        training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
        ds="continuous-sft:qwen3-8b:chunk-00003",
        status=TaskStatus.PREP_TASK_FAILURE.value,
        model_id="miner-org/carried-winner",
        model_params_count=0,
    )
    _patch_replace_seams(monkeypatch, original)
    same_type_mock = AsyncMock()
    monkeypatch.setattr(task_creator, "create_new_task_of_same_type", same_type_mock)
    recreate_mock = AsyncMock(return_value=SimpleNamespace(task_id="new-task", task_type=TaskType.CHATTASK))
    monkeypatch.setattr(task_creator, "create_continuous_sft_task", recreate_mock)

    new_task_id = await task_creator.replace_tournament_task(
        "orig-task", "tourn", "tourn_round_004", None, "tourn_round_004_pair_001", MagicMock()
    )

    same_type_mock.assert_not_awaited()
    assert new_task_id == "new-task"
    _, lineage, seed_model = recreate_mock.call_args.args
    assert lineage == "qwen3-8b"
    assert seed_model == t_cst.CONTINUOUS_SFT_LINEAGES["qwen3-8b"]


async def test_image_replacement_filters_to_original_model_type(monkeypatch):
    image_task = SimpleNamespace(task_type=TaskType.IMAGETASK, model_type=ImageModelType.Z_IMAGE)

    async def image_pool(_keypair):
        yield SimpleNamespace(model_id="repo/a", model_type=ImageModelType.KREA2)
        yield SimpleNamespace(model_id="repo/b", model_type=ImageModelType.Z_IMAGE)

    async def fake_create_synthetic_image_task(_config, models):
        selected = await anext(models)
        assert selected.model_type == ImageModelType.Z_IMAGE
        return SimpleNamespace(task_id="img-task", task_type=TaskType.IMAGETASK)

    monkeypatch.setattr(task_creator, "_get_image_models", image_pool)
    monkeypatch.setattr(task_creator, "create_synthetic_image_task", fake_create_synthetic_image_task)

    created = await task_creator.create_new_task_of_same_type(image_task, MagicMock())
    assert created.task_id == "img-task"


async def test_replacement_detaches_original_instead_of_deleting(monkeypatch):
    original = SimpleNamespace(
        task_id="orig-task",
        task_type=TaskType.INSTRUCTTEXTTASK,
        status=TaskStatus.PREP_TASK_FAILURE.value,
        model_id="Qwen/Qwen2.5-3B",
        model_params_count=3_000_000_000,
    )
    _patch_replace_seams(monkeypatch, original)
    delete_mock = AsyncMock()
    monkeypatch.setattr(task_creator.task_sql, "delete_task", delete_mock)
    monkeypatch.setattr(
        task_creator,
        "_create_round_one_group_text_replacement_task",
        AsyncMock(return_value=SimpleNamespace(task_id="new-task", task_type=TaskType.INSTRUCTTEXTTASK)),
    )

    await task_creator.replace_tournament_task(
        "orig-task", "tourn", "tourn_round_001", "tourn_round_001_group_001", None, MagicMock()
    )

    delete_mock.assert_not_awaited()
    task_creator.task_sql.detach_task_from_tournament.assert_awaited_once()
    args = task_creator.task_sql.detach_task_from_tournament.call_args.args
    assert args[:2] == ("orig-task", "tourn")
