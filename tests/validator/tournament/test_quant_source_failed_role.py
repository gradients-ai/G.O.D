# ruff: noqa: F811  (the qw / env fixtures are imported)
"""A text seed role that is out of replacements never stalls a quant tournament; the quant slot takes the fallback.

In memory, on the QuantWorld of test_quant_text_linkage (a bound quant tournament over an active text tournament with its four
seeds). Covers: binding with a failed role, binding to a text tournament completed this week, the wait alert, the final seed
row a slot is created on, release on a failed seed (fallback model, fresh dataset when the text task was never prepped), the
fallback grace timed from a seed row the seed loop has seen finish, and a held task on a replaced seed.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from core.models.dataset_models import InstructTextDatasetType
from core.models.task_models import TaskStatus
from tests.validator.tournament.test_quant_text_linkage import BOSS_SLOTS
from tests.validator.tournament.test_quant_text_linkage import FAILURE
from tests.validator.tournament.test_quant_text_linkage import QUANT_TID
from tests.validator.tournament.test_quant_text_linkage import SUCCESS
from tests.validator.tournament.test_quant_text_linkage import TEXT_TID
from tests.validator.tournament.test_quant_text_linkage import TRAINING
from tests.validator.tournament.test_quant_text_linkage import QuantWorld
from tests.validator.tournament.test_quant_text_linkage import _source
from tests.validator.tournament.test_quant_text_linkage import qw  # noqa: F401,F811  (fixture)
from tests.validator.tournament.test_seed_tasks import env  # noqa: F401  (fixture)
from validator.db.sql import tournaments as tournament_sql
from validator.tournament import constants as t_cst
from validator.tournament import orchestrator
from validator.tournament import quant_sources
from validator.tournament.models import BOSS_ROUND_SEED_ROLES
from validator.tournament.models import TEXT_SEED_ROLES
from validator.tournament.models import QuantSourceSlot
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType


COMPLETED_TID = "text_completed"
FRESH_TYPE = InstructTextDatasetType(field_instruction="question", field_output="answer")


def _fail(qw: QuantWorld, role: SeedRole) -> None:
    qw.text_rows[role].status = SeedStatus.FAILED


def _text_tournament(tournament_id: str, status: TournamentStatus) -> TournamentData:
    return TournamentData(tournament_id=tournament_id, tournament_type=TournamentType.TEXT, status=status)


def _seed_rows(tournament_id: str, status: SeedStatus, roles=TEXT_SEED_ROLES) -> list[TournamentSeedTask]:
    return [TournamentSeedTask(tournament_id=tournament_id, role=role, task_id=uuid4(), status=status) for role in roles]


# ------------------------------------------------------------------------------------------------------------- binding


class TestBindingWithAFailedRole:
    async def test_a_failed_role_does_not_keep_an_active_text_tournament_from_binding(self, qw):
        qw.sources.bindings.clear()
        _fail(qw, SeedRole.TEXT_CONTINUOUS)

        binding = await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config)

        assert binding is not None and binding.source_tournament_id == TEXT_TID

    async def test_a_role_with_its_replacement_pending_keeps_an_active_text_tournament_waiting(self, qw):
        qw.sources.bindings.clear()
        qw.text_rows[SeedRole.TEXT_R2].status = SeedStatus.REPLACED

        assert await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config) is None

    async def test_a_text_tournament_without_a_role_row_is_never_bound(self, qw):
        qw.sources.bindings.clear()
        qw.env.store.rows[:] = [row for row in qw.env.store.rows if row.role != SeedRole.TEXT_R3]

        assert await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config) is None


class TestBindingToACompletedTextTournament:
    @pytest.fixture
    def completed_only(self, qw):
        qw.sources.bindings.clear()
        qw.tournaments.clear()
        qw.completed[:] = [_text_tournament(COMPLETED_TID, TournamentStatus.COMPLETED)]
        return qw

    async def test_a_text_tournament_completed_this_week_binds_whatever_its_rows_say(self, completed_only):
        qw = completed_only
        rows = _seed_rows(COMPLETED_TID, SeedStatus.READY)
        rows[0].status, rows[1].status = SeedStatus.REPLACED, SeedStatus.FAILED
        qw.env.store.rows.extend(rows)

        binding = await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config)

        assert binding is not None and binding.source_tournament_id == COMPLETED_TID

    async def test_the_query_asks_only_for_the_current_week(self, completed_only):
        qw = completed_only
        qw.env.store.rows.extend(_seed_rows(COMPLETED_TID, SeedStatus.READY))

        await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config)

        tournament_type, since, _db = tournament_sql.get_completed_tournaments_since.await_args.args
        assert tournament_type == TournamentType.TEXT
        expected = datetime.now(timezone.utc) - timedelta(seconds=t_cst.QUANT_SOURCE_COMPLETED_WINDOW_SECONDS)
        assert abs((since - expected).total_seconds()) < 60 and t_cst.QUANT_SOURCE_COMPLETED_WINDOW_SECONDS == 7 * 24 * 3600

    async def test_a_completed_tournament_without_seed_rows_or_with_a_role_missing_is_not_bound(self, completed_only):
        qw = completed_only
        qw.env.store.rows[:] = [row for row in qw.env.store.rows if row.tournament_id != TEXT_TID]
        qw.env.store.rows.extend(_seed_rows(COMPLETED_TID, SeedStatus.READY, TEXT_SEED_ROLES - {SeedRole.TEXT_R1}))
        qw.completed.append(_text_tournament("text_unseeded", TournamentStatus.COMPLETED))

        assert await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config) is None

    async def test_an_active_text_tournament_is_preferred_to_a_completed_one(self, qw):
        qw.sources.bindings.clear()
        qw.completed[:] = [_text_tournament(COMPLETED_TID, TournamentStatus.COMPLETED)]
        qw.env.store.rows.extend(_seed_rows(COMPLETED_TID, SeedStatus.READY))

        first = await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config)
        second = await quant_sources.bind_quant_source_tournament("quant_other", qw.config)

        assert (first.source_tournament_id, second.source_tournament_id) == (TEXT_TID, COMPLETED_TID)

    async def test_a_text_tournament_already_feeding_a_quant_tournament_is_skipped(self, completed_only):
        qw = completed_only
        qw.env.store.rows.extend(_seed_rows(COMPLETED_TID, SeedStatus.READY))
        qw.sources.bind("quant_other", COMPLETED_TID)

        assert await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config) is None


class TestWaitAlert:
    @pytest.fixture
    def waiting(self, qw):
        qw.sources.bindings.clear()
        qw.tournaments.clear()
        return qw

    async def test_no_alert_before_the_delay_and_one_alert_after_it(self, waiting):
        qw = waiting

        for _ in range(3):
            assert await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config) is None
        assert qw.discord.await_count == 0 and QUANT_TID in quant_sources._WAITING_SINCE

        quant_sources._WAITING_SINCE[QUANT_TID] = datetime.now(timezone.utc) - timedelta(
            seconds=t_cst.QUANT_SOURCE_WAIT_ALERT_SECONDS - 60
        )
        await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config)
        assert qw.discord.await_count == 0

        quant_sources._WAITING_SINCE[QUANT_TID] = datetime.now(timezone.utc) - timedelta(
            seconds=t_cst.QUANT_SOURCE_WAIT_ALERT_SECONDS + 60
        )
        for _ in range(3):
            await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config)

        assert t_cst.QUANT_SOURCE_WAIT_ALERT_SECONDS == 2 * 3600
        assert qw.discord.await_count == 1 and QUANT_TID in qw.discord.await_args.kwargs["message"]

    async def test_binding_ends_the_wait(self, waiting):
        qw = waiting
        await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config)
        qw.tournaments.append(_text_tournament(TEXT_TID, TournamentStatus.ACTIVE))

        assert await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config) is not None
        assert QUANT_TID not in quant_sources._WAITING_SINCE and qw.discord.await_count == 0


# ----------------------------------------------------------------------------------------------- the seed row of a slot


class TestSlotTextSeed:
    async def test_a_failed_role_gives_its_failed_row(self, qw):
        _fail(qw, SeedRole.TEXT_R2)
        binding = qw.sources.bindings[QUANT_TID]

        seed = await quant_sources.slot_text_seed(binding, QuantSourceSlot.R2, qw.config)

        assert seed is qw.text_rows[SeedRole.TEXT_R2] and seed.status == SeedStatus.FAILED

    async def test_a_pending_replacement_in_an_active_text_tournament_gives_none(self, qw):
        qw.text_rows[SeedRole.TEXT_R2].status = SeedStatus.REPLACED

        assert await quant_sources.slot_text_seed(qw.sources.bindings[QUANT_TID], QuantSourceSlot.R2, qw.config) is None

    @pytest.mark.parametrize("status", [TournamentStatus.COMPLETED, TournamentStatus.CANCELLED])
    async def test_once_the_text_tournament_is_over_the_latest_row_is_final_whatever_its_status(self, qw, status):
        qw.text_rows[SeedRole.TEXT_R2].status = SeedStatus.REPLACED
        qw.tournaments[0].status = status

        seed = await quant_sources.slot_text_seed(qw.sources.bindings[QUANT_TID], QuantSourceSlot.R2, qw.config)

        assert seed is qw.text_rows[SeedRole.TEXT_R2]

    async def test_the_live_row_is_returned_as_before(self, qw):
        seed = await quant_sources.slot_text_seed(qw.sources.bindings[QUANT_TID], QuantSourceSlot.R1, qw.config)

        assert seed is qw.text_rows[SeedRole.TEXT_R1]


class TestCreationWithAFailedTextRole:
    async def test_a_round_task_is_created_on_the_failed_seeds_task_with_a_source_row(self, qw):
        _fail(qw, SeedRole.TEXT_R2)

        task = await qw.create_round_task(2)

        assert task.upstream_task_id == qw.text_task(SeedRole.TEXT_R2).task_id
        assert _source(qw, QuantSourceSlot.R2).source_task_id == task.upstream_task_id

    async def test_the_boss_compressions_of_a_failed_role_are_created_with_the_others(self, qw):
        _fail(qw, SeedRole.TEXT_R3)
        _fail(qw, SeedRole.TEXT_CONTINUOUS)

        seeds = await qw.create_seeds()

        assert set(seeds) == BOSS_ROUND_SEED_ROLES
        assert {source.slot for source in qw.sources.sources} == set(BOSS_SLOTS)
        failed_task = qw.text_task(SeedRole.TEXT_R3).task_id
        assert _source(qw, QuantSourceSlot.BOSS_R3_FREEFORM).source_task_id == failed_task


# ---------------------------------------------------------------------------------------- release on a failed text seed


class TestReleaseOnAFailedSeed:
    async def _setup(self, qw, boss_status=TRAINING, prepped=True):
        task = await qw.create_round_task(1)
        text_task = qw.text_task(SeedRole.TEXT_R1)
        _fail(qw, SeedRole.TEXT_R1)
        qw.training[str(text_task.task_id)] = boss_status
        qw.hub[text_task.model_id] = "full"
        if not prepped:
            text_task.training_data = text_task.test_data = None
        qw.mp.setattr(
            orchestrator,
            "_draw_quant_instruct_dataset",
            AsyncMock(return_value=("org/fresh-ds", FRESH_TYPE, 1000)),
        )
        return task, text_task

    @pytest.mark.parametrize("boss_status", [FAILURE, TRAINING, None])
    async def test_it_takes_the_fallback_at_once_whatever_the_boss_row_says(self, qw, boss_status):
        task, text_task = await self._setup(qw, boss_status)

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and task.model_id == text_task.model_id
        source = _source(qw, QuantSourceSlot.R1)
        assert (source.source_model_repo, source.source_is_fallback) == (text_task.model_id, True)
        assert task.ds == text_task.ds  # the upstream's data is copied: it was prepped
        (alert,) = qw.alerts.await_args_list
        assert "fallback" in alert.kwargs["message"] and "out of replacements" in alert.kwargs["message"]
        assert "human" not in alert.kwargs["message"]
        assert orchestrator._draw_quant_instruct_dataset.await_count == 0

    async def test_a_boss_run_that_landed_a_full_model_is_still_used(self, qw):
        task, text_task = await self._setup(qw, SUCCESS)
        raw = qw.raw_repo(text_task.task_id)
        qw.hub[raw] = "full"

        await orchestrator._release_chained_task(task, qw.config)

        assert task.model_id == raw and _source(qw, QuantSourceSlot.R1).source_is_fallback is False
        assert qw.alerts.await_count == 0

    async def test_a_succeeded_run_with_no_usable_output_takes_the_fallback_without_waiting(self, qw):
        task, text_task = await self._setup(qw, SUCCESS)
        qw.text_rows[SeedRole.TEXT_R1].updated_at = datetime.now(timezone.utc)

        await orchestrator._release_chained_task(task, qw.config)

        assert task.model_id == text_task.model_id and _source(qw, QuantSourceSlot.R1).source_is_fallback is True

    async def test_an_unprepped_text_task_releases_the_task_on_a_fresh_dataset_for_its_own_prep(self, qw):
        task, text_task = await self._setup(qw, FAILURE, prepped=False)

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and task.model_id == text_task.model_id
        assert (task.ds, task.dataset_type) == ("org/fresh-ds", FRESH_TYPE)
        assert task.training_data is None and task.test_data is None and task.carve_calibration is True
        assert _source(qw, QuantSourceSlot.R1).source_is_fallback is True
        (alert,) = qw.alerts.await_args_list
        assert "fresh dataset" in alert.kwargs["message"]
        assert orchestrator._draw_quant_instruct_dataset.await_count == 1

    async def test_a_recreated_copy_on_the_old_fresh_dataset_is_released_on_another_fresh_dataset(self, qw):
        task, _ = await self._setup(qw, FAILURE, prepped=False)
        task.ds = "org/an-earlier-fresh-ds"  # a recreated copy carries the failed task's dataset, not the upstream's

        await orchestrator._release_chained_task(task, qw.config)

        assert task.ds == "org/fresh-ds" and task.training_data is None

    async def test_a_failed_draw_holds_the_task(self, qw):
        task, _ = await self._setup(qw, FAILURE, prepped=False)
        orchestrator._draw_quant_instruct_dataset.side_effect = RuntimeError("no datasets")

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.AWAITING_UPSTREAM and _source(qw, QuantSourceSlot.R1).source_model_repo is None

    async def test_a_task_that_is_not_on_a_failed_seed_still_holds_when_the_text_task_is_not_prepped(self, qw):
        task, _ = await self._setup(qw, FAILURE, prepped=False)
        qw.text_rows[SeedRole.TEXT_R1].status = SeedStatus.TRAINED

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.AWAITING_UPSTREAM and qw.alerts.await_count == 0
        assert orchestrator._draw_quant_instruct_dataset.await_count == 0

    async def test_a_continuous_compression_keeps_its_own_chunk_and_draws_nothing(self, qw):
        await qw.create_seeds()
        task = qw.quant_task(SeedRole.BOSS_CONTINUOUS_FREEFORM)
        text_task = qw.text_task(SeedRole.TEXT_CONTINUOUS)
        _fail(qw, SeedRole.TEXT_CONTINUOUS)
        text_task.training_data = text_task.test_data = None
        qw.hub[text_task.model_id] = "full"
        qw.mp.setattr(orchestrator, "_draw_quant_instruct_dataset", AsyncMock())

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and task.model_id == text_task.model_id
        assert task.training_data.startswith("https://minio/train-fresh") and task.ds.startswith("continuous-quant:")
        assert orchestrator._draw_quant_instruct_dataset.await_count == 0


# ------------------------------------------------------------------------------------------------------ the grace


class TestFallbackGraceFollowsTheSeedLoop:
    GRACE = t_cst.QUANT_SOURCE_FALLBACK_GRACE_SECONDS

    async def test_a_row_still_training_holds_however_old_it_is_then_the_grace_runs_from_the_trained_row(self, qw):
        task = await qw.create_round_task(2)
        text_task = qw.text_task(SeedRole.TEXT_R2)
        qw.land(SeedRole.TEXT_R2, "empty")  # the boss's run succeeded, its output is not on the hub
        qw.hub[text_task.model_id] = "full"
        row = qw.text_rows[SeedRole.TEXT_R2]

        row.status, row.updated_at = SeedStatus.TRAINING, datetime.now(timezone.utc) - timedelta(hours=3)
        await orchestrator._release_chained_task(task, qw.config)
        assert task.status == TaskStatus.AWAITING_UPSTREAM and qw.alerts.await_count == 0

        row.status, row.updated_at = SeedStatus.TRAINED, datetime.now(timezone.utc)
        await orchestrator._release_chained_task(task, qw.config)
        assert task.status == TaskStatus.AWAITING_UPSTREAM and qw.alerts.await_count == 0

        row.updated_at = datetime.now(timezone.utc) - timedelta(seconds=self.GRACE + 60)
        await orchestrator._release_chained_task(task, qw.config)
        assert task.status == TaskStatus.PENDING and task.model_id == text_task.model_id
        assert _source(qw, QuantSourceSlot.R2).source_is_fallback is True

    async def test_a_ready_row_counts_like_a_trained_one(self, qw):
        task = await qw.create_round_task(2)
        text_task = qw.text_task(SeedRole.TEXT_R2)
        qw.land(SeedRole.TEXT_R2, "empty")
        qw.hub[text_task.model_id] = "full"
        row = qw.text_rows[SeedRole.TEXT_R2]
        row.status, row.updated_at = SeedStatus.READY, datetime.now(timezone.utc) - timedelta(seconds=self.GRACE + 60)

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING


# ----------------------------------------------------------------------------------- a task on a replaced text seed


class TestHeldOnAReplacedSeed:
    async def _replaced(self, qw, status=TournamentStatus.ACTIVE):
        task = await qw.create_round_task(2)
        old_text = qw.text_task(SeedRole.TEXT_R2)
        qw.hub[old_text.model_id] = "full"
        qw.text_rows[SeedRole.TEXT_R2].status = SeedStatus.REPLACED
        qw.tournaments[0].status = status
        return task, old_text

    async def test_no_successor_in_an_active_text_tournament_holds(self, qw):
        task, old_text = await self._replaced(qw)

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.AWAITING_UPSTREAM and task.upstream_task_id == old_text.task_id
        assert qw.alerts.await_count == 0

    async def test_no_successor_once_the_text_tournament_completed_takes_the_fallback(self, qw):
        task, old_text = await self._replaced(qw, TournamentStatus.COMPLETED)

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and task.model_id == old_text.model_id
        assert _source(qw, QuantSourceSlot.R2).source_is_fallback is True

    async def test_a_refused_repoint_holds_while_active_and_takes_the_fallback_once_completed(self, qw):
        task, old_text = await self._replaced(qw)
        qw.add_text_seed(SeedRole.TEXT_R2, attempt=2)
        qw.mp.setattr(orchestrator, "repoint_task_onto_seed", AsyncMock(return_value=False))

        await orchestrator._release_chained_task(task, qw.config)
        assert task.status == TaskStatus.AWAITING_UPSTREAM and task.upstream_task_id == old_text.task_id

        qw.tournaments[0].status = TournamentStatus.COMPLETED
        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and task.model_id == old_text.model_id
        assert _source(qw, QuantSourceSlot.R2).source_is_fallback is True

    async def test_a_live_successor_is_followed_and_the_next_pass_releases(self, qw):
        task, _ = await self._replaced(qw)
        new_seed = qw.add_text_seed(SeedRole.TEXT_R2, attempt=2)

        await orchestrator._release_chained_task(task, qw.config)
        assert task.upstream_task_id == new_seed.task_id and task.status == TaskStatus.AWAITING_UPSTREAM
        assert _source(qw, QuantSourceSlot.R2).source_task_id == new_seed.task_id

        raw = qw.land(SeedRole.TEXT_R2)  # add_text_seed made the successor the role's task
        await orchestrator._release_chained_task(task, qw.config)
        assert task.status == TaskStatus.PENDING and task.model_id == raw

    async def test_a_failed_successor_is_followed_and_the_task_then_falls_back(self, qw):
        task, _ = await self._replaced(qw)
        new_seed = qw.add_text_seed(SeedRole.TEXT_R2, attempt=2)
        new_seed.status = SeedStatus.FAILED
        new_text = qw.env.tasks[new_seed.task_id]
        qw.hub[new_text.model_id] = "full"

        await orchestrator._release_chained_task(task, qw.config)
        assert task.upstream_task_id == new_seed.task_id and task.status == TaskStatus.AWAITING_UPSTREAM

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and task.model_id == new_text.model_id
        assert _source(qw, QuantSourceSlot.R2).source_is_fallback is True
