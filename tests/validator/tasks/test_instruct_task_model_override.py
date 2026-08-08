"""create_synthetic_instruct_text_task honours model_id_override / allow_augmentation / allow_yarn.

These knobs exist for the pre-boss task: the model must come from the override (no pool is even
provided), and augmentation/YaRN must stay off so both competitors train the exact published
model. Everything else — dataset pull, computed hours — stays the standard path.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from validator.tasks.synthetics import scheduler


PRE_BOSS_MODEL = "Qwen/Qwen3-32B"


def _patch_seams(monkeypatch):
    """Stub dataset selection, column lookup, HF param fetch and persistence; return add_task."""
    dataset = SimpleNamespace(dataset_id="tatsu-lab/alpaca", num_rows=50_000)
    monkeypatch.setattr(scheduler, "get_dataset", AsyncMock(return_value=dataset))
    columns = SimpleNamespace(field_instruction="instruction", field_input="input", field_output="output")
    monkeypatch.setattr(scheduler, "_get_columns_for_instruct_dataset", AsyncMock(return_value=columns))
    monkeypatch.setattr(scheduler, "get_model_num_params", lambda model_id: 10_000_000_000)
    add_task = AsyncMock(side_effect=lambda task, psql_db: task)
    monkeypatch.setattr(scheduler, "add_task", add_task)
    return add_task


async def test_override_forces_model_without_pool_augmentation_or_yarn(monkeypatch):
    add_task = _patch_seams(monkeypatch)
    # Force yarn selection ON so the allow_yarn=False guard is what keeps it off.
    monkeypatch.setattr(scheduler, "maybe_get_yarn_factor", lambda: 4)

    task = await scheduler.create_synthetic_instruct_text_task(
        MagicMock(),
        None,  # no model pool — override must be enough
        MagicMock(),
        enable_kl=False,
        model_id_override=PRE_BOSS_MODEL,
        allow_augmentation=False,
        allow_yarn=False,
    )

    assert task.model_id == PRE_BOSS_MODEL
    assert task.augmentation_config is None
    assert task.yarn_factor is None
    assert task.use_kl is False
    assert task.ds == "tatsu-lab/alpaca"
    assert task.hours_to_complete > 0  # standard computed budget, nothing forced
    add_task.assert_awaited_once()


async def test_default_path_still_draws_from_the_pool(monkeypatch):
    _patch_seams(monkeypatch)

    async def _models():
        yield "unsloth/Llama-3.2-3B-Instruct"

    task = await scheduler.create_synthetic_instruct_text_task(MagicMock(), _models(), MagicMock())

    assert task.model_id == "unsloth/Llama-3.2-3B-Instruct"
