"""Seed-time evaluation of the boss's boss-round entries (validator/tournament/README.md,
"Design notes").

The boss's entry on a boss-round seed is evaluated as soon as the boss's row trains and stored; the boss round evaluates
only the challenger and ranks the task from both stored rows. These tests drive the real lifecycle (evaluation-row
seeding, the evaluation pass, finalisation and ranking) over an in-memory store, and check that:

  * the boss's entry is evaluated at seed time, the challenger's alone at the boss round, each exactly once, and the task
    is finalised once, only after the challenger is scored;
  * the verdict equals the one from scoring both together at the boss round (instruct within 0.2 %, freeform cap and
    bytes, fixed paired gate across SKUs, a boss training failure);
  * an evaluation error of the boss's entry at seed time is retried, never scored; spent retries park it and the boss
    round evaluates it (fallback, alerted); a missing same-SKU reference is re-measured at seed time.
"""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import BaseModel

import validator.db.constants as db_cst
from core.models.quant_models import QuantEvalGpu
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from tests.validator.tournament.test_boss_round_seeds import _instruct
from tests.validator.tournament.test_boss_round_seeds import _own_upstream
from validator.db.sql import submissions_and_scoring as submissions_and_scoring_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_performance
from validator.db.sql import tournament_seeds as seeds_sql
from validator.db.sql import tournaments as tournament_sql
from validator.lifecycle import boss_seed_eval
from validator.lifecycle import constants as lifecycle_cst
from validator.lifecycle import eval_infra_errors
from validator.lifecycle import quant_reference as lifecycle_quant_reference
from validator.lifecycle import tasks as lifecycle_tasks
from validator.scoring import tasks as scoring_tasks
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import EvalHotkeyResults
from validator.scoring.models import MinerResultsText
from validator.scoring.models import QuantSkuReference
from validator.tasks.models import freeform_penalty_policy
from validator.tournament import constants as t_cst
from validator.tournament import orchestrator
from validator.tournament import round_results
from validator.tournament import thresholds
from validator.tournament.models import ANY_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import ATTACHED_SEED_ROLES
from validator.tournament.models import INPUT_SEED_ROLES
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TrainingStatus


BOSS = EMISSION_BURN_HOTKEY
CHALLENGER = "5Challenger"
CONFIG = SimpleNamespace(psql_db=None, discord_url=None)
SUCCESS = TrainingStatus.SUCCESS.value
FAILURE = TrainingStatus.FAILURE.value
FINGERPRINT = "fp-test-rows"
H100_SXM = QuantEvalGpu(name="NVIDIA H100 80GB HBM3", sm_count=132, sku="H100 SXM")
H100_PCIE = QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114, sku="H100 PCIe")
A100 = QuantEvalGpu(name="NVIDIA A100-SXM4-80GB", sm_count=108)
N_EXAMPLES = 400


class _StopPass(Exception):
    pass


async def _seeding_pass() -> None:
    """One pass of orchestrator.seed_tournament_evaluations_from_training: the sleep that ends the pass stops the loop."""
    with patch.object(orchestrator.asyncio, "sleep", AsyncMock(side_effect=_StopPass)):
        with pytest.raises(_StopPass):
            await orchestrator.seed_tournament_evaluations_from_training(CONFIG)


class Outcome(BaseModel):
    """What one evaluation of one entry produces: its task_nodes row, its vector and (quant) its own reference."""

    test_loss: float | None = None
    per_example: list[float] | None = None
    fingerprint: str | None = FINGERPRINT
    eval_gpu: QuantEvalGpu = H100_SXM
    artifact_bytes: int | None = None
    decoder_bytes: int | None = None
    quant_failure: str | None = None
    score_reason: str | None = None  # set only by the scorer, for its own errors
    reference_per_example: list[float] | None = None  # quant: the entry's own bf16 model on this SKU; None = not measured

    def row(self, hotkey: str) -> dict:
        return {
            db_cst.HOTKEY: hotkey,
            db_cst.TEST_LOSS: self.test_loss,
            db_cst.SYNTH_LOSS: self.test_loss,
            db_cst.SCORE_REASON: self.score_reason,
            db_cst.ARTIFACT_BYTES: self.artifact_bytes,
            db_cst.DECODER_BYTES: self.decoder_bytes,
            db_cst.PASSED_BAR: None,
            db_cst.FREEFORM_PENALTY: None,
            db_cst.QUANT_FAILURE: self.quant_failure,
            db_cst.EVAL_GPU: self.eval_gpu.model_dump(),
        }


class EvaluationError(Exception):
    pass


def _quant(reference: np.ndarray, ratio: np.ndarray | float, gpu: QuantEvalGpu, **fields) -> Outcome:
    losses = reference * ratio
    return Outcome(
        test_loss=float(losses.mean()),
        per_example=losses.tolist(),
        eval_gpu=gpu,
        reference_per_example=reference.tolist(),
        **fields,
    )


def _instruct_outcome(loss: float) -> Outcome:
    return Outcome(test_loss=loss, eval_gpu=A100)


class Store:
    """One boss-round seed task and every SQL seam the lifecycle, scoring and the round's verdict read, in memory."""

    def __init__(self, monkeypatch, task, role: SeedRole, outcomes: dict[str, list[Outcome | Exception]]) -> None:
        self.task = task
        self.task_id = str(task.task_id)
        self.seed = TournamentSeedTask(tournament_id="tid", role=role, task_id=task.task_id, status=SeedStatus.TRAINING)
        self.outcomes = {hotkey: list(queue) for hotkey, queue in outcomes.items()}
        self.attached = False
        self.assigned = [BOSS]
        self.training: dict[str, str] = {BOSS: TrainingStatus.TRAINING.value}
        self.eval_rows: dict[str, str] = {}
        self.node_rows: dict[str, dict] = {}
        self.vectors: dict[str, tuple[list[float] | None, str | None]] = {}
        self.references: dict[tuple[str, str], QuantSkuReference] = {}
        self.evaluated: list[list[str]] = []
        self.finalised: list[list[MinerResultsText]] = []
        self.alerts: list[str] = []

        mp = monkeypatch
        # Task, seed and assignment.
        mp.setattr(task_sql, "get_task", AsyncMock(side_effect=lambda task_id, db: self.task))
        mp.setattr(task_sql, "update_task", AsyncMock())
        mp.setattr(task_sql, "get_tasks_with_status", AsyncMock(side_effect=self._with_status))
        mp.setattr(task_sql, "get_nodes_assigned_to_task", AsyncMock(side_effect=self._nodes))
        mp.setattr(seeds_sql, "get_seed_task_by_task_id", AsyncMock(side_effect=self._seed))
        mp.setattr(seeds_sql, "is_task_attached_to_round", AsyncMock(side_effect=lambda task_id, db: self.attached))
        mp.setattr(seeds_sql, "tournament_is_decided", AsyncMock(return_value=False))
        mp.setattr(tournament_sql, "get_training_status_for_task", AsyncMock(side_effect=lambda task_id, db: dict(self.training)))
        mp.setattr(tournament_sql, "get_tasks_with_all_training_completed", AsyncMock(side_effect=self._all_completed))
        # Evaluation rows.
        mp.setattr(task_sql, "add_task_evaluation_pairs", AsyncMock(side_effect=self._add_pairs))
        mp.setattr(task_sql, "get_task_ids_with_evaluation_statuses", AsyncMock(side_effect=self._ids_with_rows))
        mp.setattr(task_sql, "get_task_evaluations_by_status", AsyncMock(side_effect=self._rows_by_status))
        mp.setattr(task_sql, "get_task_evaluation_rows", AsyncMock(side_effect=self._eval_rows))
        mp.setattr(task_sql, "update_task_evaluations_status", AsyncMock(side_effect=self._set_status))
        mp.setattr(task_sql, "clear_task_evaluation_deployments", AsyncMock())
        mp.setattr(lifecycle_tasks, "reconcile_eval_deployments", AsyncMock())
        # Exceptions that are not seed-time retries go to the infra-error recorder (re-queued and counted, never 'failure').
        self.infra_attempts: dict[str, int] = {}
        mp.setattr(task_sql, "record_task_evaluation_infra_errors", AsyncMock(side_effect=self._record_infra))
        mp.setattr(eval_infra_errors, "notify_evaluation_exception", AsyncMock())
        mp.setattr(eval_infra_errors, "task_deployment_ids_for_hotkeys", AsyncMock(return_value=[]))
        mp.setattr(boss_seed_eval, "notify_evaluation_exception", AsyncMock(side_effect=self._alert))
        # The evaluator and the stored rows.
        mp.setattr(lifecycle_tasks, "evaluate_and_score_hotkeys", self._evaluate)
        for module in (submissions_and_scoring_sql, scoring_tasks, lifecycle_quant_reference, boss_seed_eval):
            mp.setattr(module, "get_task_node_losses", AsyncMock(side_effect=lambda task_id, db: list(self.node_rows.values())))
        for module in (scoring_tasks, lifecycle_quant_reference, thresholds):
            mp.setattr(module, "get_quant_sku_references", AsyncMock(side_effect=lambda task_id, db: dict(self.references)))
        own_model = AsyncMock(side_effect=lambda task, hotkey, db: f"bf16-{hotkey}")
        mp.setattr(lifecycle_quant_reference, "quant_reference_model", own_model)
        mp.setattr(scoring_tasks, "set_task_node_quality_score", AsyncMock())
        mp.setattr(boss_seed_eval, "get_per_example_losses", AsyncMock(side_effect=self._vector))
        mp.setattr(round_results, "get_per_example_losses", AsyncMock(side_effect=self._vector))
        mp.setattr(thresholds, "get_task_node_eval_gpus", AsyncMock(side_effect=self._eval_gpus))
        real_finalise = scoring_tasks.finalize_task_scores_from_raw_losses

        async def finalise(task, hotkeys, config):
            results = await real_finalise(task, hotkeys, config)
            self.finalised.append(results)
            return results

        mp.setattr(lifecycle_tasks, "finalize_task_scores_from_raw_losses", finalise)

    # --- seams -------------------------------------------------------------------------------------------------------

    async def _with_status(self, status, psql_db=None, **_):
        return [self.task] if self.task.status == status else []

    async def _nodes(self, task_id, db):
        return [SimpleNamespace(hotkey=hotkey) for hotkey in self.assigned]

    async def _seed(self, task_id, db):
        return self.seed if str(task_id) == self.task_id else None

    async def _all_completed(self, db):
        terminal = {SUCCESS, FAILURE}
        done = self.task.status == TaskStatus.TRAINING and all(status in terminal for status in self.training.values())
        return [self.task_id] if done else []

    async def _add_pairs(self, task_id, db, include_failed_training=False):
        for hotkey, status in self.training.items():
            if status == SUCCESS:
                self.eval_rows.setdefault(hotkey, "pending")

    async def _ids_with_rows(self, statuses, db, task_statuses=None, tournament_only=False):
        live = any(status in statuses for status in self.eval_rows.values())
        return [self.task_id] if live and TaskStatus(self.task.status).value in (task_statuses or []) else []

    async def _rows_by_status(self, task_id, status, db):
        return [{"hotkey": hotkey} for hotkey, current in self.eval_rows.items() if current == status]

    async def _eval_rows(self, task_id, db):
        return [{"hotkey": hotkey, "evaluation_status": status} for hotkey, status in self.eval_rows.items()]

    async def _record_infra(self, task_id, hotkeys, max_attempts, held_status, db):
        for hotkey in hotkeys:
            self.infra_attempts[hotkey] = self.infra_attempts.get(hotkey, 0) + 1
            self.eval_rows[hotkey] = held_status if self.infra_attempts[hotkey] >= max_attempts else "pending"
        return {hotkey: self.infra_attempts[hotkey] for hotkey in hotkeys}

    async def _set_status(self, task_id, hotkeys, status, db):
        for hotkey in hotkeys:
            self.eval_rows[hotkey] = status

    async def _vector(self, task_id, hotkey, db):
        return self.vectors.get(hotkey, (None, None))

    async def _eval_gpus(self, task_id, hotkeys, db):
        return {hotkey: QuantEvalGpu.model_validate(self.node_rows[hotkey][db_cst.EVAL_GPU]) for hotkey in hotkeys}

    async def _alert(self, config, *, context, **_):
        self.alerts.append(context)

    async def _evaluate(self, task, hotkeys, num_gpus, config):
        """The per-hotkey evaluation pass: persist each entry's raw result, as scoring.process_miners_pool does."""
        self.evaluated.append(list(hotkeys))
        evaluated, failed = [], []
        for hotkey in hotkeys:
            queue = self.outcomes[hotkey]
            outcome = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(outcome, Exception):
                raise outcome
            self.node_rows[hotkey] = outcome.row(hotkey)
            self.vectors[hotkey] = (outcome.per_example, outcome.fingerprint)
            if outcome.reference_per_example is not None:
                key = (hotkey, outcome.eval_gpu.reference_key)
                self.references.setdefault(
                    key,
                    QuantSkuReference(
                        task_id=task.task_id,
                        hotkey=hotkey,
                        sku=outcome.eval_gpu.reference_key,
                        model=f"bf16-{hotkey}",
                        loss=float(np.mean(outcome.reference_per_example)),
                        per_example_losses=outcome.reference_per_example,
                        eval_set_fingerprint=FINGERPRINT,
                        eval_gpu=outcome.eval_gpu,
                    ),
                )
            evaluated.append(hotkey)
            if outcome.test_loss is None and outcome.quant_failure is None:
                failed.append(hotkey)
        return EvalHotkeyResults(evaluated=evaluated, failed=failed)

    # --- the timeline ------------------------------------------------------------------------------------------------

    def train(self, hotkey: str, status: str = SUCCESS) -> None:
        self.training[hotkey] = status

    async def cycle(self) -> None:
        """One orchestrator seeding pass, then one lifecycle evaluation pass over whatever it picks up."""
        await _seeding_pass()
        picked = await lifecycle_tasks._get_tasks_ready_for_evaluation(CONFIG)
        if any(str(task.task_id) == self.task_id for task in picked):
            await lifecycle_tasks._evaluate_pending_pairs_for_task(self.task, 1, CONFIG)

    def attach(self) -> None:
        """The boss round attaches the seed and adds the challenger's training row."""
        self.attached = True
        self.assigned.append(CHALLENGER)
        self.training[CHALLENGER] = TrainingStatus.PENDING.value

    async def seed_time_flow(self, boss_training: str = SUCCESS) -> None:
        """Boss trains at activation and is evaluated then; the boss round evaluates the challenger alone."""
        self.train(BOSS, boss_training)
        for _ in range(3):
            await self.cycle()
        assert self.task.status == TaskStatus.TRAINING and not self.finalised
        self.attach()
        await self.cycle()
        self.train(CHALLENGER)
        await self.cycle()
        await self.cycle()

    async def together_flow(self, boss_training: str = SUCCESS) -> None:
        """Nothing is evaluated until the challenger has trained, then both in one pass."""
        self.train(BOSS, boss_training)
        self.attach()
        self.train(CHALLENGER)
        await self.cycle()
        await self.cycle()

    @property
    def evaluations_of(self) -> dict[str, int]:
        counts = {}
        for batch in self.evaluated:
            for hotkey in batch:
                counts[hotkey] = counts.get(hotkey, 0) + 1
        return counts

    def result(self, hotkey: str) -> MinerResultsText | None:
        assert len(self.finalised) == 1, "finalised exactly once"
        return next((result for result in self.finalised[0] if result.hotkey == hotkey), None)


@pytest.fixture(autouse=True)
def _fresh_seed_time_state(monkeypatch):
    # The cycles below run back to back; the backoff has its own test.
    monkeypatch.setattr(lifecycle_cst, "SEED_TIME_BOSS_EVAL_BACKOFF_SECONDS", 0)
    for state in (
        boss_seed_eval._SEED_TIME_RETRIES,
        boss_seed_eval._LAST_RETRY_AT,
        boss_seed_eval._PARKED_ALERTED,
        boss_seed_eval._ROUND_OPEN_ALERTED,
        boss_seed_eval._EVAL_SET_MISMATCH_ALERTED,
    ):
        state.clear()
    yield


def _summary(result: MinerResultsText | None) -> tuple | None:
    if result is None:
        return None
    return (result.adjusted_loss, result.passed_bar, result.freeform_penalty, result.rank, result.score, result.score_reason)


async def _verdict(store: Store) -> tuple:
    """The round's decision on the task from the stored rows, as get_knockout_winners takes it."""
    boss, challenger = store.result(BOSS), store.result(CHALLENGER)
    task = store.task
    if task.task_type == TaskType.QUANTFREEFORMTASK:
        # As round_results._resolve_freeform_boss_round_task: within is only asked of a task the boss kept.
        verdict = thresholds.resolve_freeform_boss_task(boss, challenger, t_cst.QUANT_FREEFORM_BOSS_BYTE_MARGIN)
        within = verdict.winner_hotkey == BOSS and thresholds.challenger_within_freeform_allowed_loss(boss, challenger)
        return (verdict.winner_hotkey, within)
    if boss is None or not math.isfinite(boss.adjusted_loss or math.nan):
        return ("boss unusable",)
    verdict = await round_results._resolve_boss_round_task_winner(
        task.task_id, task, BOSS, CHALLENGER, boss.adjusted_loss, challenger.adjusted_loss, 0.0, None
    )
    return (verdict.winner_hotkey, verdict.is_draw, verdict.challenger_within_allowed_loss, verdict.decided_by)


async def _both_timings(monkeypatch, make_task, role, outcomes, boss_training: str = SUCCESS) -> tuple[Store, Store]:
    together = Store(monkeypatch, make_task(), role, outcomes)
    await together.together_flow(boss_training)
    together_verdict = await _verdict(together)
    seed_time = Store(monkeypatch, make_task(), role, outcomes)
    await seed_time.seed_time_flow(boss_training)
    assert await _verdict(seed_time) == together_verdict
    return together, seed_time


def _training_task(make):
    def build():
        task = make()
        task.status = TaskStatus.TRAINING
        return task

    return build


_REF = np.random.default_rng(7).uniform(0.6, 1.4, N_EXAMPLES)
# The challenger's own bf16 model scored on another SKU: per-example references differ, so alignment matters.
_REF_PCIE = _REF * np.random.default_rng(8).uniform(0.97, 1.03, N_EXAMPLES)


def _fixed_task():
    return _training_task(lambda: _own_upstream(_instruct(), freeform=False))()


def _freeform_task():
    return _training_task(lambda: _own_upstream(_instruct()))()


class TestVerdictsMatchScoringBothTogether:
    @pytest.mark.parametrize(
        ("challenger_ratio", "challenger_bytes", "expected"),
        [
            (1.03, 900_000, (CHALLENGER, False)),  # under the cap, 10 % fewer bytes: wins
            (1.03, 1_005_000, (BOSS, True)),  # 0.5 % more bytes: the boss keeps it, within the allowed loss
            (1.03, 1_020_000, (BOSS, False)),  # 2 % more bytes: not within
            (1.10, 500_000, (BOSS, False)),  # over the cap: never wins, never within
        ],
    )
    async def test_freeform_cap_and_bytes(self, monkeypatch, challenger_ratio, challenger_bytes, expected):
        outcomes = {
            BOSS: [_quant(_REF, 1.02, H100_SXM, artifact_bytes=1_000_000, decoder_bytes=0)],
            CHALLENGER: [_quant(_REF_PCIE, challenger_ratio, H100_PCIE, artifact_bytes=challenger_bytes, decoder_bytes=0)],
        }

        together, seed_time = await _both_timings(monkeypatch, _freeform_task, SeedRole.BOSS_CHAINED_FREEFORM, outcomes)

        assert await _verdict(seed_time) == expected
        assert seed_time.evaluations_of == {BOSS: 1, CHALLENGER: 1}
        for hotkey in (BOSS, CHALLENGER):
            assert _summary(seed_time.result(hotkey)) == _summary(together.result(hotkey))
        assert freeform_penalty_policy(seed_time.task).cap == pytest.approx(1.08)

    @pytest.mark.parametrize(("challenger_ratio", "challenger_wins"), [(1.01, True), (1.06, False), (1.05, False)])
    async def test_fixed_paired_gate_across_skus(self, monkeypatch, challenger_ratio, challenger_wins):
        noise = np.random.default_rng(11).uniform(0.995, 1.005, N_EXAMPLES)
        outcomes = {
            BOSS: [_quant(_REF, 1.05 * noise, H100_SXM)],
            CHALLENGER: [_quant(_REF_PCIE, challenger_ratio * noise[::-1], H100_PCIE)],
        }

        together, seed_time = await _both_timings(monkeypatch, _fixed_task, SeedRole.BOSS_CHAINED_FIXED, outcomes)

        verdict = await _verdict(seed_time)
        assert (verdict[0] == CHALLENGER) is challenger_wins
        assert seed_time.evaluations_of == {BOSS: 1, CHALLENGER: 1}
        # Each side normalised by its own model's reference on its own SKU; the boss's was measured at seed time.
        assert set(seed_time.references) == {(BOSS, "H100 SXM"), (CHALLENGER, "H100 PCIe")}

    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    async def test_boss_training_failure_is_the_boss_lost_slot(self, monkeypatch, make_task):
        outcomes = {
            BOSS: [EvaluationError("the boss must never be evaluated")],
            CHALLENGER: [_quant(_REF_PCIE, 1.03, H100_PCIE, artifact_bytes=900_000, decoder_bytes=0)],
        }
        role = SeedRole.BOSS_CHAINED_FIXED if make_task is _fixed_task else SeedRole.BOSS_CHAINED_FREEFORM

        together, seed_time = await _both_timings(monkeypatch, make_task, role, outcomes, boss_training=FAILURE)

        assert seed_time.evaluated == [[CHALLENGER]] and BOSS not in seed_time.eval_rows
        assert seed_time.result(BOSS) is None and seed_time.result(CHALLENGER).rank == 1
        assert seed_time.task.status == TaskStatus.SUCCESS


class TestSeedTimeTiming:
    async def test_boss_evaluated_once_at_seed_time_and_the_task_finalised_once_after_the_challenger(self, monkeypatch):
        store = Store(
            monkeypatch,
            _fixed_task(),
            SeedRole.BOSS_CHAINED_FIXED,
            {BOSS: [_quant(_REF, 1.05, H100_SXM)], CHALLENGER: [_quant(_REF, 1.04, H100_SXM)]},
        )
        store.train(BOSS)
        await store.cycle()
        assert store.evaluated == [[BOSS]] and store.eval_rows == {BOSS: "success"}
        assert store.node_rows[BOSS][db_cst.TEST_LOSS] is not None and store.vectors[BOSS][1] == FINGERPRINT
        assert (BOSS, "H100 SXM") in store.references
        for _ in range(3):
            await store.cycle()
        assert store.evaluated == [[BOSS]] and not store.finalised and store.task.status == TaskStatus.TRAINING

        store.attach()
        await store.cycle()
        store.train(CHALLENGER, TrainingStatus.TRAINING.value)
        await store.cycle()
        assert not store.finalised  # held until the challenger is scored

        store.train(CHALLENGER)
        await store.cycle()
        assert store.evaluated == [[BOSS], [CHALLENGER]]
        assert len(store.finalised) == 1 and store.task.status == TaskStatus.SUCCESS
        assert {result.hotkey for result in store.finalised[0]} == {BOSS, CHALLENGER}
        assert not store.alerts

    async def test_the_preevaluation_move_still_waits_for_the_challenger(self, monkeypatch):
        store = Store(monkeypatch, _fixed_task(), SeedRole.BOSS_CHAINED_FIXED, {BOSS: [_quant(_REF, 1.05, H100_SXM)]})
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value="tid"))
        store.train(BOSS)
        await store.cycle()

        await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)

        assert store.task.status == TaskStatus.TRAINING


class TestSeedTimeFailures:
    async def test_an_evaluation_exception_at_seed_time_is_retried_never_scored(self, monkeypatch):
        good = _quant(_REF, 1.05, H100_SXM)
        store = Store(
            monkeypatch,
            _fixed_task(),
            SeedRole.BOSS_CHAINED_FIXED,
            {BOSS: [EvaluationError("runpod has no capacity"), good], CHALLENGER: [_quant(_REF, 1.04, H100_SXM)]},
        )
        store.train(BOSS)
        await store.cycle()
        assert store.eval_rows[BOSS] == "pending" and BOSS not in store.node_rows  # not a scored fail

        await store.cycle()
        assert store.eval_rows[BOSS] == "success" and store.evaluations_of[BOSS] == 2
        assert not store.alerts

    async def test_a_scorer_side_failure_is_retried_but_the_evaluators_judgement_stands(self, monkeypatch):
        scorer_error = Outcome(score_reason="Evaluation failed: deployment vanished", eval_gpu=H100_SXM)
        store = Store(
            monkeypatch, _fixed_task(), SeedRole.BOSS_CHAINED_FIXED, {BOSS: [scorer_error, _quant(_REF, 1.05, H100_SXM)]}
        )
        store.train(BOSS)
        await store.cycle()
        assert store.eval_rows[BOSS] == "pending"
        await store.cycle()
        assert store.eval_rows[BOSS] == "success"

        judged = Store(monkeypatch, _fixed_task(), SeedRole.BOSS_CHAINED_FIXED, {BOSS: [Outcome(eval_gpu=H100_SXM)]})
        judged.train(BOSS)
        await judged.cycle()
        await judged.cycle()
        assert judged.eval_rows[BOSS] == "failure" and judged.evaluations_of[BOSS] == 1  # NaN loss, no scorer reason

    async def test_a_retried_evaluation_waits_out_the_backoff_before_it_is_dispatched_again(self, monkeypatch):
        monkeypatch.setattr(lifecycle_cst, "SEED_TIME_BOSS_EVAL_BACKOFF_SECONDS", 600)
        now = [1000.0]
        monkeypatch.setattr(boss_seed_eval.time, "monotonic", lambda: now[0])
        outcomes = [EvaluationError("hub down"), _quant(_REF, 1.05, H100_SXM)]
        store = Store(monkeypatch, _fixed_task(), SeedRole.BOSS_CHAINED_FIXED, {BOSS: outcomes})
        store.train(BOSS)
        await store.cycle()
        assert store.eval_rows[BOSS] == "pending" and store.evaluations_of[BOSS] == 1

        now[0] += 599
        assert not await lifecycle_tasks._get_tasks_ready_for_evaluation(CONFIG)
        now[0] += 2
        assert await lifecycle_tasks._get_tasks_ready_for_evaluation(CONFIG)
        await store.cycle()
        assert store.eval_rows[BOSS] == "success" and store.evaluations_of[BOSS] == 2

    async def test_spent_retries_park_the_row_and_the_boss_round_evaluates_it(self, monkeypatch):
        retries = lifecycle_cst.SEED_TIME_BOSS_EVAL_RETRIES
        store = Store(
            monkeypatch,
            _fixed_task(),
            SeedRole.BOSS_CHAINED_FIXED,
            {
                BOSS: [EvaluationError("no capacity")] * retries + [_quant(_REF, 1.05, H100_SXM)],
                CHALLENGER: [_quant(_REF, 1.01, H100_SXM)],
            },
        )
        store.train(BOSS)
        for _ in range(retries + 3):
            await store.cycle()
        assert store.evaluations_of[BOSS] == retries and store.eval_rows[BOSS] == "pending"
        assert store.alerts == ["Seed-time boss evaluation parked"]
        assert not await lifecycle_tasks._get_tasks_ready_for_evaluation(CONFIG)  # parked

        # The boss round opens: the missing seed-time result is alerted, and the row is evaluated now (the fallback).
        store.attach()
        assert await boss_seed_eval.alert_if_boss_evaluation_missing(store.task, CONFIG)
        assert await boss_seed_eval.alert_if_boss_evaluation_missing(store.task, CONFIG)
        assert store.alerts.count("Seed-time boss evaluation missing at the boss round") == 1
        await store.cycle()
        assert store.eval_rows[BOSS] == "success" and store.evaluations_of[BOSS] == retries + 1
        store.train(CHALLENGER)
        await store.cycle()
        assert store.result(CHALLENGER).rank == 1 and store.evaluations_of[CHALLENGER] == 1

    async def test_a_settled_seed_time_result_raises_no_round_open_alert(self, monkeypatch):
        store = Store(monkeypatch, _fixed_task(), SeedRole.BOSS_CHAINED_FIXED, {BOSS: [_quant(_REF, 1.05, H100_SXM)]})
        store.train(BOSS)
        await store.cycle()
        store.attach()

        assert not await boss_seed_eval.alert_if_boss_evaluation_missing(store.task, CONFIG)
        assert not store.alerts

    async def test_a_missing_same_sku_reference_is_measured_at_seed_time(self, monkeypatch):
        unmeasured = _quant(_REF, 1.05, H100_SXM).model_copy(update={"reference_per_example": None})
        store = Store(
            monkeypatch, _fixed_task(), SeedRole.BOSS_CHAINED_FIXED, {BOSS: [unmeasured, _quant(_REF, 1.05, H100_SXM)]}
        )
        store.train(BOSS)
        await store.cycle()
        assert store.eval_rows[BOSS] == "pending" and not store.references  # re-evaluated now, not at the boss round

        await store.cycle()
        assert store.eval_rows[BOSS] == "success" and (BOSS, "H100 SXM") in store.references
        await store.cycle()
        assert store.evaluations_of[BOSS] == 2

    async def test_after_attach_an_exception_is_an_infra_error_like_any_task(self, monkeypatch):
        # Attached: no seed-time retry; the exception is re-queued and counted (validator/lifecycle/eval_infra_errors.py).
        store = Store(monkeypatch, _fixed_task(), SeedRole.BOSS_CHAINED_FIXED, {BOSS: [EvaluationError("boom")]})
        store.attach()
        store.train(BOSS)
        await store.cycle()

        assert store.eval_rows[BOSS] == "pending"
        assert store.infra_attempts == {BOSS: 1}

    async def test_an_eval_set_mismatch_between_the_stored_entries_is_alerted(self, monkeypatch):
        store = Store(
            monkeypatch,
            _fixed_task(),
            SeedRole.BOSS_CHAINED_FIXED,
            {
                BOSS: [_quant(_REF, 1.05, H100_SXM)],
                CHALLENGER: [_quant(_REF, 1.04, H100_SXM).model_copy(update={"fingerprint": "other-rows"})],
            },
        )
        await store.seed_time_flow()

        assert store.alerts == ["Boss-round eval set mismatch"]
        verdict = await _verdict(store)
        assert verdict[0] == BOSS and verdict[3] == "unpairable per-example vectors"


class TestOtherTasksUnchanged:
    async def test_a_non_seed_tournament_task_finalises_on_its_rows(self, monkeypatch):
        store = Store(monkeypatch, _fixed_task(), SeedRole.BOSS_CHAINED_FIXED, {BOSS: [_quant(_REF, 1.05, H100_SXM)]})
        store.seed = None
        store.train(BOSS)
        await store.cycle()

        assert len(store.finalised) == 1 and store.task.status == TaskStatus.SUCCESS

    async def test_a_non_seed_task_exception_is_an_infra_error_not_a_failure(self, monkeypatch):
        store = Store(monkeypatch, _fixed_task(), SeedRole.BOSS_CHAINED_FIXED, {BOSS: [EvaluationError("boom")]})
        store.seed = None
        store.train(BOSS)
        await store.cycle()

        assert store.eval_rows[BOSS] == "pending"
        assert store.infra_attempts == {BOSS: 1}


class TestVectorsAreKeptForUnattachedSeeds:
    async def test_is_paired_comparison_task_counts_every_attached_seed(self):
        """The boss's seed-time vector and fingerprint must be stored before the round that attaches the seed exists."""
        calls = []

        class Connection:
            async def fetchval(self, query, *params):
                calls.append((query, params))
                return True

        class Db:
            async def connection(self):
                class Context:
                    async def __aenter__(self_inner):
                        return Connection()

                    async def __aexit__(self_inner, *exc):
                        return False

                return Context()

        assert await tournament_performance.is_paired_comparison_task("task", Db())
        query, params = calls[0]
        assert db_cst.TOURNAMENT_SEED_TASKS_TABLE in query
        # Every seed role a round attaches: boss-round roles (the chained ones included), text and quant round seeds.
        assert set(params[2]) == {role.value for role in ATTACHED_SEED_ROLES}
        assert {role.value for role in ANY_BOSS_ROUND_SEED_ROLES} <= set(params[2])
        assert not set(params[2]) & {role.value for role in INPUT_SEED_ROLES}
