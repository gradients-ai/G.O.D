"""An image tournament task stores each entry's per-image loss vector next to its scalar; nothing reads it for a verdict."""

import uuid
from types import SimpleNamespace

import pytest

from core.models.payload_models import DiffusionLosses
from core.models.payload_models import EvaluationResultImage
from core.models.task_models import TaskType
from validator.scoring import tasks as scoring_tasks
from validator.scoring.models import MinerResultsText
from validator.scoring.tasks import _image_miner_result
from validator.scoring.tasks import _per_image_losses


def _eval(text_guided: list[float], no_text: list[float]) -> EvaluationResultImage:
    return EvaluationResultImage(
        eval_loss=DiffusionLosses(text_guided_losses=text_guided, no_text_losses=no_text), is_finetune=True
    )


def _task(task_type: TaskType) -> SimpleNamespace:
    return SimpleNamespace(task_id=uuid.uuid4(), task_type=task_type)


class Recorder:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, in_tournament: bool, paired: bool = False) -> None:
        self.writes: dict[str, dict] = {}

        async def set_task_node_losses(task_id, hotkey, test_loss, synth_loss, psql_db, **kwargs):
            self.writes[hotkey] = {"test_loss": test_loss, **kwargs}

        async def is_paired(task_id, psql_db):
            return paired

        async def tournament_task(task_id, psql_db):
            return object() if in_tournament else None

        monkeypatch.setattr(scoring_tasks, "set_task_node_losses", set_task_node_losses)
        monkeypatch.setattr(scoring_tasks, "is_paired_comparison_task", is_paired)
        monkeypatch.setattr(scoring_tasks.tournament_sql, "get_tournament_task_by_task_id", tournament_task)


class TestVector:
    def test_is_the_half_half_blend_per_image(self) -> None:
        assert _per_image_losses(_eval([0.2, 0.4], [0.1, 0.5])) == pytest.approx([0.15, 0.45])

    def test_mean_equals_the_scalar(self) -> None:
        result = _image_miner_result("hk", _eval([0.2, 0.4, 0.9], [0.1, 0.5, 0.2]), None)
        assert result.per_example_losses is not None
        assert sum(result.per_example_losses) / len(result.per_example_losses) == pytest.approx(result.test_loss)

    def test_mismatched_lengths_give_no_vector_and_keep_the_scalar(self) -> None:
        result = _image_miner_result("hk", _eval([0.2, 0.4, 0.6], [0.1, 0.3]), None)
        assert result.per_example_losses is None
        assert result.test_loss == pytest.approx(0.5 * 0.4 + 0.5 * 0.2)

    def test_invalid_entry_carries_no_vector(self) -> None:
        assert _image_miner_result("hk", _eval([], []), None).per_example_losses is None


class TestPersistence:
    @pytest.mark.asyncio
    async def test_tournament_image_task_persists_vector_and_scalar(self, monkeypatch) -> None:
        recorder = Recorder(monkeypatch, in_tournament=True)
        result = _image_miner_result("hk", _eval([0.2, 0.4], [0.1, 0.5]), None)
        await scoring_tasks._persist_raw_task_results(_task(TaskType.IMAGETASK), [result], "db")
        written = recorder.writes["hk"]
        assert written["per_example_losses"] == pytest.approx([0.15, 0.45])
        assert sum(written["per_example_losses"]) / 2 == pytest.approx(written["test_loss"])

    @pytest.mark.asyncio
    async def test_organic_image_task_persists_no_vector(self, monkeypatch) -> None:
        recorder = Recorder(monkeypatch, in_tournament=False)
        result = _image_miner_result("hk", _eval([0.2, 0.4], [0.1, 0.5]), None)
        await scoring_tasks._persist_raw_task_results(_task(TaskType.IMAGETASK), [result], "db")
        assert recorder.writes["hk"]["per_example_losses"] is None
        assert recorder.writes["hk"]["test_loss"] == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_mismatched_lengths_store_the_scalar_only(self, monkeypatch) -> None:
        recorder = Recorder(monkeypatch, in_tournament=True)
        result = _image_miner_result("hk", _eval([0.2, 0.4, 0.6], [0.1, 0.3]), None)
        await scoring_tasks._persist_raw_task_results(_task(TaskType.IMAGETASK), [result], "db")
        assert recorder.writes["hk"]["per_example_losses"] is None
        assert recorder.writes["hk"]["test_loss"] == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_text_task_still_keeps_vectors_only_when_paired(self, monkeypatch) -> None:
        result = MinerResultsText(
            hotkey="hk",
            test_loss=1.0,
            synth_loss=1.0,
            is_finetune=True,
            task_type=TaskType.INSTRUCTTEXTTASK,
            per_example_losses=[1.0],
        )
        unpaired = Recorder(monkeypatch, in_tournament=True, paired=False)
        await scoring_tasks._persist_raw_task_results(_task(TaskType.INSTRUCTTEXTTASK), [result], "db")
        assert unpaired.writes["hk"]["per_example_losses"] is None

        paired = Recorder(monkeypatch, in_tournament=True, paired=True)
        await scoring_tasks._persist_raw_task_results(_task(TaskType.INSTRUCTTEXTTASK), [result], "db")
        assert paired.writes["hk"]["per_example_losses"] == [1.0]
