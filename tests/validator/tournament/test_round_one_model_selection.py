from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from core.constants.environments import EnvironmentName
from core.constants.environments import TrainingStartPoint
from core.models.task_models import TaskType
from core.whitelisted_env_models import ENV_MODEL_SIZE_B
from core.whitelisted_env_models import SUPPORTED_ENV_MODELS
from validator.tasks.synthetics import scheduler
from validator.tournament import constants as t_cst
from validator.tournament import task_creator
from validator.tournament.gpu_requirements import get_tournament_gpu_requirement
from validator.tournament.models import GpuRequirement
from validator.tournament.models import Group
from validator.tournament.models import GroupRound


EXPECTED_ROUND_ONE_ONLY_MODELS = (
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "ibm-granite/granite-4.1-3b",
    "allenai/Olmo-3-7B-Instruct",
    "allenai/Olmo-Hybrid-Instruct-SFT-7B",
    "LiquidAI/LFM2.5-2.6B",
    "mistralai/Ministral-3-3B-Base-2512",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "google/gemma-4-E2B",
)


def test_round_one_only_catalog_and_size_metadata_are_complete():
    assert t_cst.ROUND_ONE_ONLY_TEXT_ENV_MODELS == EXPECTED_ROUND_ONE_ONLY_MODELS
    assert len(set(EXPECTED_ROUND_ONE_ONLY_MODELS)) == len(EXPECTED_ROUND_ONE_ONLY_MODELS)
    assert set(EXPECTED_ROUND_ONE_ONLY_MODELS).isdisjoint(SUPPORTED_ENV_MODELS)
    assert set(EXPECTED_ROUND_ONE_ONLY_MODELS) <= ENV_MODEL_SIZE_B.keys()
    assert ENV_MODEL_SIZE_B["Qwen/Qwen3.5-4B"] == 5.0
    assert ENV_MODEL_SIZE_B["google/gemma-4-E2B"] == 5.1


@pytest.mark.asyncio
async def test_text_model_generator_exposes_new_catalog_only_when_round_one_is_enabled(monkeypatch):
    content_models = [
        {"model_id": "legacy/model-a"},
        *({"model_id": model_id} for model_id in EXPECTED_ROUND_ONE_ONLY_MODELS),
        {"model_id": "legacy/model-b"},
        {"model_id": "legacy/model-a"},
    ]
    content_call = AsyncMock(return_value=content_models)
    monkeypatch.setattr(scheduler, "call_content_service", content_call)
    monkeypatch.setattr(scheduler.random, "shuffle", lambda values: None)

    normal_models = scheduler._get_text_models(MagicMock())
    assert [await anext(normal_models), await anext(normal_models)] == ["legacy/model-a", "legacy/model-b"]
    await normal_models.aclose()

    round_one_models = scheduler._get_text_models(MagicMock(), include_round_one_only_models=True)
    yielded = [await anext(round_one_models) for _ in range(2 + len(EXPECTED_ROUND_ONE_ONLY_MODELS))]
    await round_one_models.aclose()

    assert yielded == ["legacy/model-a", "legacy/model-b", *EXPECTED_ROUND_ONE_ONLY_MODELS]
    assert yielded.count("Qwen/Qwen3.5-4B") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("round_number", "expected_opt_in"), [(1, True), (2, False)])
async def test_text_group_model_pool_is_round_aware(monkeypatch, round_number, expected_opt_in):
    get_models = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(task_creator, "_get_text_models", get_models)
    monkeypatch.setattr(task_creator, "_get_instruct_text_datasets", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(task_creator, "_get_dpo_datasets", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(task_creator, "_create_single_group_text_tasks", AsyncMock(return_value=[]))

    round_data = GroupRound(
        round_id=f"tourn_round_{round_number:03d}",
        round_number=round_number,
        groups=[Group(member_ids=["a", "b", "c"])],
    )
    config = SimpleNamespace(keypair=MagicMock())

    await task_creator._create_group_text_tasks(round_data, "tourn", config, is_final_round=False)

    assert get_models.call_args.kwargs["include_round_one_only_models"] is expected_opt_in


@pytest.mark.asyncio
async def test_round_one_text_replacement_uses_round_one_model_pool(monkeypatch):
    get_models = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(task_creator, "_get_text_models", get_models)
    monkeypatch.setattr(task_creator, "_get_instruct_text_datasets", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(task_creator, "create_synthetic_instruct_text_task", AsyncMock(return_value=MagicMock()))

    await task_creator._create_round_one_group_text_replacement_task(SimpleNamespace(keypair=MagicMock()))

    assert get_models.call_args.kwargs["include_round_one_only_models"] is True


@pytest.mark.asyncio
async def test_environment_random_pool_only_includes_new_models_when_enabled(monkeypatch):
    candidate_pools: list[list[str]] = []

    def choose(candidates):
        candidate_pools.append(list(candidates))
        return candidates[-1]

    monkeypatch.setattr(scheduler.random, "choice", choose)
    monkeypatch.setattr(scheduler, "maybe_get_augmentation_config", lambda task_type: None)
    monkeypatch.setattr(scheduler, "add_task", AsyncMock(side_effect=lambda task, psql_db: task))
    config = SimpleNamespace(psql_db=MagicMock())
    kwargs = {"environment_names_override": [EnvironmentName.LIARS_DICE]}

    normal_task = await scheduler.create_synthetic_env_task(config, MagicMock(), MagicMock(), **kwargs)
    round_one_task = await scheduler.create_synthetic_env_task(
        config,
        MagicMock(),
        MagicMock(),
        include_round_one_only_models=True,
        **kwargs,
    )

    assert candidate_pools[0] == SUPPORTED_ENV_MODELS
    assert candidate_pools[1] == [*SUPPORTED_ENV_MODELS, *EXPECTED_ROUND_ONE_ONLY_MODELS]
    assert normal_task.model_id not in EXPECTED_ROUND_ONE_ONLY_MODELS
    assert round_one_task.model_id == EXPECTED_ROUND_ONE_ONLY_MODELS[-1]


@pytest.mark.asyncio
async def test_round_one_environment_replacement_propagates_model_pool_opt_in(monkeypatch):
    original = SimpleNamespace(task_type=TaskType.ENVIRONMENTTASK, model_params_count=0)
    monkeypatch.setattr(task_creator, "_get_text_models", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(task_creator, "_get_instruct_text_datasets", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(task_creator, "_get_dpo_datasets", MagicMock(return_value=MagicMock()))
    create_env = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(task_creator, "create_synthetic_env_task", create_env)

    await task_creator.create_new_task_of_same_type(
        original,
        SimpleNamespace(keypair=MagicMock()),
        include_round_one_only_models=True,
    )

    assert create_env.call_args.kwargs["include_round_one_only_models"] is True


@pytest.mark.asyncio
async def test_round_one_environment_prep_failure_routes_to_opted_in_replacement(monkeypatch):
    original = task_creator.EnvRawTask.model_construct(
        task_id="original-task",
        status="prep_task_failure",
        model_id="Qwen/Qwen3.5-2B",
        model_params_count=0,
        ds="env_task_dummy_dataset",
        environment_names=[EnvironmentName.LIARS_DICE],
        eval_seed=1234,
        training_start_point=TrainingStartPoint.DEFAULT,
        hours_to_complete=2.0,
    )
    replacement = SimpleNamespace(task_id="replacement-task", task_type=TaskType.ENVIRONMENTTASK)
    create_env = AsyncMock(return_value=replacement)
    monkeypatch.setattr(task_creator.task_sql, "get_task", AsyncMock(return_value=original))
    monkeypatch.setattr(task_creator.task_sql, "get_nodes_assigned_to_task", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator.task_sql, "delete_task", AsyncMock())
    monkeypatch.setattr(task_creator, "_get_text_models", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(task_creator, "_get_instruct_text_datasets", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(task_creator, "create_synthetic_env_task", create_env)
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", AsyncMock())
    config = SimpleNamespace(keypair=MagicMock(), psql_db=MagicMock())

    result = await task_creator.replace_tournament_task(
        "original-task",
        "tourn",
        "tourn_round_001",
        "tourn_round_001_group_001",
        None,
        config,
    )

    assert result == "replacement-task"
    replacement_kwargs = create_env.await_args.kwargs
    assert replacement_kwargs["include_round_one_only_models"] is True
    assert replacement_kwargs["model_id_override"] == "Qwen/Qwen3.5-2B"
    assert replacement_kwargs["environment_names_override"] == [EnvironmentName.LIARS_DICE]
    assert replacement_kwargs["eval_seed_override"] == 1234
    assert replacement_kwargs["hours_override"] == 2.0


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Qwen/Qwen3.5-0.8B", GpuRequirement.H100_1X),
        ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", GpuRequirement.H100_1X),
        ("Qwen/Qwen3.5-4B", GpuRequirement.H100_2X),
        ("allenai/Olmo-3-7B-Instruct", GpuRequirement.H100_2X),
        ("google/gemma-4-E2B", GpuRequirement.H100_2X),
    ],
)
def test_round_one_environment_models_have_expected_two_environment_gpu_size(model_id, expected):
    assert (
        get_tournament_gpu_requirement(
            TaskType.ENVIRONMENTTASK,
            model_params_count=0,
            model_id=model_id,
            environment_count=2,
        )
        == expected
    )
