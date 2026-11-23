"""Measure each own-model quant (task, hotkey) bf16 reference before the miner trains.

A separate module from validator/tournament/quant_reference.py on purpose: that one is imported by
validator.scoring.tasks, while this one must import evaluation code (run_evaluation_basilica_quant), which
would be circular from there.

A task-common quant task (the miner rounds R1-R3, whose shared input is the boss's output on an input seed; anything
that is not an own-model task) has one reference for all its miners, stored under
hotkey '' (quant_reference.reference_hotkey_for). It is measured once per task, the same way, and the orchestrator
defers every miner's training until it is there, however long that takes.

On an own-model task (OWN_UPSTREAM_MODEL or OWN_R1_MODEL: each player compresses its own upstream model) every player is
scored as a multiple of ITS OWN bf16 model, and the training request publishes that loss (and the freeform bar) to the
player. So the reference has to exist before training starts: this cycle measures it for every assigned hotkey that has a
starting model and no stored row, and the orchestrator defers the miner's training until the row is there
(orchestrator._reference_ready).

A quant task with a calibration set (either track) gets a second measurement per hotkey: the same bf16 repo on the calibration
rows, published to the miner with them (never used to score). It is keyed separately, and training waits for it too. A stored
calibration reference measured on another example set than the task's current calibration set counts as missing and is
replaced by the re-measurement.

When both are missing, one deployment loads the model once and scores both example sets; when one is missing, a deployment
measures that one alone. The calibration reference of a shared deployment is stored only after its test reference, and a
deployment that returns the test reference alone leaves the calibration one to a measurement of its own next cycle.

Every measurement is retried every cycle for as long as it is missing; a measurement that keeps failing pings a human at every
QUANT_REFERENCE_PREPASS_FAILURES_BEFORE_ALERT-th consecutive failure. A miner is never trained without them.
"""

import asyncio
import tempfile
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import model_validator

import validator.db.constants as db_cst
import validator.tournament.constants as cst
from core.downloads import download_s3_file
from core.logging import get_logger
from core.models.dataset_models import FileFormat
from core.models.payload_models import EvaluationResultText
from core.models.quant_models import QuantReferenceScore
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskStatus
from validator.app.config import Config
from validator.db.sql import gpu_costs as gpu_cost_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournaments as tournament_sql
from validator.db.sql.quant_calibration_references import add_quant_calibration_reference
from validator.db.sql.quant_calibration_references import get_quant_calibration_references
from validator.db.sql.quant_references import add_quant_sku_reference
from validator.db.sql.quant_references import get_quant_sku_references
from validator.db.sql.submissions_and_scoring import get_task_node_losses
from validator.evaluation.docker_evaluation import run_evaluation_basilica_quant
from validator.evaluation.errors import DeploymentNotReadyError
from validator.evaluation.errors import EvaluationCapacityUnavailable
from validator.evaluation.errors import describe_error
from validator.evaluation.notifications import notify_evaluation_exception
from validator.evaluation.quant_loss import load_examples
from validator.evaluation.quant_loss import quant_eval_set_fingerprint
from validator.lifecycle.tasks import compute_required_gpus
from validator.scoring.models import QuantCalibrationReferenceRecord
from validator.scoring.models import QuantSkuReference
from validator.tasks.models import AnyQuantRawTask
from validator.tasks.models import QuantRawTask
from validator.tournament.quant_reference import quant_reference_model
from validator.tournament.quant_reference import reference_hotkey_for


logger = get_logger(__name__)

_CANDIDATE_STATUSES = (TaskStatus.LOOKING_FOR_NODES, TaskStatus.TRAINING)
_COST_GPU_TYPE = "H100"  # the quant evaluator only accepts H100 SKUs

# Process-local state, like the orchestrator's alert sets: a restart re-measures at worst, never double-writes
# (add_quant_sku_reference is first-writer-wins).
_CALIBRATION = "calibration"
_PROBE_HOTKEY = "any-hotkey"  # reference_hotkey_for echoes a hotkey on an own-model task and returns "" otherwise
_IN_PROGRESS: set[tuple[str, ...]] = set()
_CONSECUTIVE_FAILURES: dict[tuple[str, ...], int] = {}
_EVAL_SET_FINGERPRINTS: dict[str, str] = {}


class ReferenceToMeasure(BaseModel):
    """A (task, reference hotkey) with a usable model and a missing bf16 reference; hotkey '' is the task-common one.

    test and calibration say which of the two is missing: the one on the task's test rows, the one on its calibration
    rows, or both (measured in one deployment).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: AnyQuantRawTask
    hotkey: str
    repo: str
    test: bool = True
    calibration: bool = False

    @model_validator(mode="after")
    def validate_something_is_missing(self) -> "ReferenceToMeasure":
        if not (self.test or self.calibration):
            raise ValueError("a reference to measure is missing on the test rows, the calibration rows or both")
        return self

    def key_of(self, calibration: bool) -> tuple[str, ...]:
        """(task, hotkey), plus a marker for the calibration measurement so the two are counted separately."""
        base = (str(self.task.task_id), self.hotkey)
        return base + (_CALIBRATION,) if calibration else base

    @property
    def keys(self) -> list[tuple[str, ...]]:
        """The key of every measurement this item covers."""
        return [self.key_of(calibration) for calibration, missing in ((False, self.test), (True, self.calibration)) if missing]

    @property
    def dataset(self) -> str:
        """The example set the deployment's own result is scored on: the test rows unless only calibration is missing."""
        return self.task.test_data if self.test else self.task.calibration_data

    @property
    def calibration_dataset(self) -> str | None:
        """The calibration rows as a second example set of the same deployment, when both references are missing."""
        return self.task.calibration_data if self.test and self.calibration else None


def _label(calibration: bool) -> str:
    return "calibration bf16 reference" if calibration else "bf16 reference"


def has_calibration_set(task: AnyQuantRawTask) -> bool:
    """Whether the task publishes calibration rows (and so a reference on them): a quant task with calibration_data."""
    return isinstance(task, QuantRawTask) and bool(task.calibration_data)


async def _candidate_tasks(config: Config) -> list[AnyQuantRawTask]:
    """Tournament quant tasks, own-model or task-common, that are prepped (their eval set exists) and not yet past training."""
    tasks = []
    for status in _CANDIDATE_STATUSES:
        tasks += await task_sql.get_tasks_with_status(status, config.psql_db, tournament_filter="only")
    return [task for task in tasks if task.task_type in QUANT_TASK_TYPES and task.test_data]


async def _failed_hotkeys(task: AnyQuantRawTask, config: Config) -> set[str]:
    """Hotkeys with a persisted scored fail (no usable R1 model): nothing to measure, they never train."""
    rows = await get_task_node_losses(task.task_id, config.psql_db)
    return {row[db_cst.HOTKEY] for row in rows if row[db_cst.QUANT_FAILURE]}


def _is_task_common(task: AnyQuantRawTask) -> bool:
    return reference_hotkey_for(task, _PROBE_HOTKEY) == ""


async def _missing_for_task(task: AnyQuantRawTask, config: Config) -> list[ReferenceToMeasure]:
    """A task-common task needs one reference (hotkey '', whatever the miners); an own-model task one per usable miner."""
    if _is_task_common(task):
        reference_hotkeys = [""]
    else:
        nodes = await task_sql.get_nodes_assigned_to_task(str(task.task_id), config.psql_db)
        failed = await _failed_hotkeys(task, config)
        reference_hotkeys = [node.hotkey for node in nodes if node.hotkey not in failed]
    measured = {hotkey for hotkey, _ in await get_quant_sku_references(task.task_id, config.psql_db)}
    calibrated = (
        {
            hotkey
            for hotkey, record in (await get_quant_calibration_references(task.task_id, config.psql_db)).items()
            if record.eval_set_fingerprint == task.calibration_fingerprint
        }
        if has_calibration_set(task)
        else set()
    )
    missing = []
    for hotkey in reference_hotkeys:
        needs_test = hotkey not in measured
        # A record on another example set is not a measurement.
        needs_calibration = has_calibration_set(task) and hotkey not in calibrated
        if not (needs_test or needs_calibration):
            continue
        repo = await quant_reference_model(task, hotkey, config.psql_db)
        if repo:
            missing.append(
                ReferenceToMeasure(task=task, hotkey=hotkey, repo=repo, test=needs_test, calibration=needs_calibration)
            )
    return missing


async def references_to_measure(config: Config) -> list[ReferenceToMeasure]:
    missing = []
    for task in await _candidate_tasks(config):
        missing += await _missing_for_task(task, config)
    return missing


async def _comparable_task_ids(task: AnyQuantRawTask, config: Config) -> list[str]:
    """The tasks whose measured rows this task may reuse: the previous round's, plus, for a task-common task, the other
    tasks of its own round (the fixed and freeform tasks of a slot share base and data). Reuse is gated on the same
    model and a byte-identical eval (or calibration) set, so a task that does not match is simply not copied from."""
    tournament_id = await tournament_sql.get_tournament_id_by_task_id(str(task.task_id), config.psql_db)
    if tournament_id is None:
        return []
    rounds = await tournament_sql.get_tournament_rounds(tournament_id, config.psql_db)
    tasks_by_round = {r.round_number: await tournament_sql.get_tournament_tasks(r.round_id, config.psql_db) for r in rounds}
    own_round = next(
        (number for number, tasks in tasks_by_round.items() if str(task.task_id) in {t.task_id for t in tasks}), None
    )
    if own_round is None:
        return []
    comparable = [t.task_id for t in tasks_by_round.get(own_round - 1, [])]
    if _is_task_common(task):
        comparable += [t.task_id for t in tasks_by_round[own_round] if t.task_id != str(task.task_id)]
    return comparable


async def _previous_round_references(task: AnyQuantRawTask, hotkey: str, config: Config) -> list[QuantSkuReference]:
    references = []
    for task_id in await _comparable_task_ids(task, config):
        stored = await get_quant_sku_references(UUID(task_id), config.psql_db)
        references += [reference for (ref_hotkey, _), reference in stored.items() if ref_hotkey == hotkey]
    return references


async def _previous_round_calibration_references(
    task: AnyQuantRawTask, hotkey: str, config: Config
) -> list[QuantCalibrationReferenceRecord]:
    records = []
    for task_id in await _comparable_task_ids(task, config):
        stored = await get_quant_calibration_references(UUID(task_id), config.psql_db)
        if hotkey in stored:
            records.append(stored[hotkey])
    return records


def _fingerprint_of_file(path: Path) -> str:
    return quant_eval_set_fingerprint(load_examples(path))


async def _eval_set_fingerprint(task: AnyQuantRawTask) -> str:
    """Fingerprint of the task's test_data, downloaded once per task per process (numpy only, no torch)."""
    key = str(task.task_id)
    if key not in _EVAL_SET_FINGERPRINTS:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = await download_s3_file(task.test_data, save_path=tmp_dir)
            _EVAL_SET_FINGERPRINTS[key] = await asyncio.to_thread(_fingerprint_of_file, Path(path))
    return _EVAL_SET_FINGERPRINTS[key]


async def _copy_previous_round_references(item: ReferenceToMeasure, config: Config) -> bool:
    """Reuse the previous round's row when it measured the same model on a byte-identical eval set.

    Returns True when at least one row was copied (a SKU the previous round did not measure is not needed
    yet: the scorer measures it in-deployment like any other missing SKU).
    """
    previous = await _previous_round_references(item.task, item.hotkey, config)
    if not previous:
        return False
    fingerprint = await _eval_set_fingerprint(item.task)
    reusable = [ref for ref in previous if ref.eval_set_fingerprint == fingerprint and ref.model == item.repo]
    for reference in reusable:
        copy = reference.model_copy(update={"task_id": item.task.task_id, "measured_at": None})
        await add_quant_sku_reference(copy, config.psql_db)
    if reusable:
        logger.info(
            f"Task {item.task.task_id} hotkey {item.hotkey}: copied {len(reusable)} bf16 reference row(s) for {item.repo} "
            f"from the previous round (same eval set {fingerprint})"
        )
    return bool(reusable)


async def _copy_previous_round_calibration_reference(item: ReferenceToMeasure, config: Config) -> bool:
    """Reuse the previous round's calibration row when it measured the same model on a byte-identical calibration set."""
    for record in await _previous_round_calibration_references(item.task, item.hotkey, config):
        if record.model == item.repo and record.eval_set_fingerprint == item.task.calibration_fingerprint:
            copy = record.model_copy(update={"task_id": item.task.task_id, "measured_at": None})
            await add_quant_calibration_reference(copy, config.psql_db)
            logger.info(
                f"Task {item.task.task_id} hotkey {item.hotkey}: copied the calibration bf16 reference for {item.repo} "
                f"from the previous round (same calibration set {record.eval_set_fingerprint})"
            )
            return True
    return False


async def _start_cost_run(run_key: str, item: ReferenceToMeasure, num_gpus: int, config: Config) -> None:
    try:
        await gpu_cost_sql.start_cost_run(
            run_key=run_key,
            task_id=str(item.task.task_id),
            category="evaluation",
            gpu_type=_COST_GPU_TYPE,
            gpu_count=max(1, num_gpus),
            psql_db=config.psql_db,
            metadata={
                "hotkey": item.hotkey,
                "repo": item.repo,
                "purpose": "quant_reference_prepass" if item.test else "quant_calibration_reference_prepass",
            },
        )
    except Exception as e:
        logger.error(f"Failed to start GPU cost run {run_key}: {e}", exc_info=True)


async def _finish_cost_run(run_key: str, success: bool, config: Config) -> None:
    try:
        await gpu_cost_sql.finish_cost_run(run_key=run_key, success=success, psql_db=config.psql_db)
    except Exception as e:
        logger.error(f"Failed to finalize GPU cost run {run_key}: {e}", exc_info=True)


async def _evaluate_reference(item: ReferenceToMeasure, num_gpus: int) -> EvaluationResultText | Exception:
    """Score the miner's own bf16 repo as both submission and reference in one deployment.

    When both references are missing the deployment also gets the calibration rows, and scores them with the model it
    loaded for the test rows (result.quant_calibration_reference).

    psql_db=None is deliberate. basilica.py reads and writes task_nodes (deployment ids, scores), the
    GPU reservations and the evaluation cost runs only when a psql_db is given (load_eval_pair_state_for_models,
    persist_deployment_ids_for_repo, reserve/release_evaluation_gpu_reservation and _start_evaluation_cost_run
    all skip on None), so this measurement can never touch the (task, hotkey) row it is a reference for.
    """
    results = await run_evaluation_basilica_quant(
        dataset=item.dataset,
        models=[item.repo],
        original_model=item.repo,
        dataset_type=item.task.dataset_type,
        file_format=FileFormat.JSON,
        num_gpus=num_gpus,
        quant_spec=None,
        quant_format=None,
        task_id=item.task.task_id,
        psql_db=None,
        decoded_model_urls=None,
        reference_models={item.repo: item.repo},
        cached_references={item.repo: {}},
        # A bare bf16 measurement may run on an A100 too: it is stored under its own SKU and only ever sets the
        # published target; scoring normalises by the reference on the entry's own SKU (reference_on).
        reference_only=True,
        calibration_dataset=item.calibration_dataset,
    )
    return results.results.get(item.repo, Exception("evaluation returned no result"))


def _problem_with(result: EvaluationResultText | Exception) -> str | None:
    """Why the evaluation produced no usable reference, None when it did."""
    if isinstance(result, Exception):
        return f"evaluation errored: {describe_error(result)}"
    if result.quant_reference is None:
        return "the deployment returned no reference"
    if result.quant_failure:
        return f"the reference did not score: {result.quant_failure}"
    return None


def _calibration_mismatch(item: ReferenceToMeasure, reference: QuantReferenceScore) -> str | None:
    """A calibration reference measured on other examples than the ones published would mislead the miner."""
    expected = item.task.calibration_fingerprint
    if expected and reference.eval_set_fingerprint != expected:
        return "the reference was measured on a different calibration set than the one published"
    return None


async def _record_failure(item: ReferenceToMeasure, calibration: bool, detail: str, config: Config) -> None:
    """Infra trouble, never the miner's fault: retried next cycle for as long as it takes (the miner's training waits for it), and
    a human is pinged at every QUANT_REFERENCE_PREPASS_FAILURES_BEFORE_ALERT-th consecutive failure.

    calibration says which measurement failed; the two are counted separately."""
    key, label = item.key_of(calibration), _label(calibration)
    count = _CONSECUTIVE_FAILURES.get(key, 0) + 1
    _CONSECUTIVE_FAILURES[key] = count
    logger.error(
        f"Task {item.task.task_id} hotkey {item.hotkey}: {label} measurement failed ({count} in a row, infra, "
        f"will retry; training waits for it): {detail}"
    )
    if count % cst.QUANT_REFERENCE_PREPASS_FAILURES_BEFORE_ALERT == 0:
        await notify_evaluation_exception(
            config,
            task_id=str(item.task.task_id),
            task_type=item.task.task_type,
            context=f"Quant {'calibration ' if calibration else ''}reference pre-pass keeps failing",
            error=f"{count} consecutive failures measuring the {label} of {item.repo}: {detail}",
            hotkeys=[item.hotkey] if item.hotkey else [],
            repos=[item.repo],
        )


async def _store_calibration_reference(item: ReferenceToMeasure, reference: QuantReferenceScore, config: Config) -> None:
    measured = QuantCalibrationReferenceRecord(
        task_id=item.task.task_id,
        hotkey=item.hotkey,
        model=reference.model,
        loss=reference.eval_loss,
        per_example_losses=reference.per_example_losses,
        eval_set_fingerprint=reference.eval_set_fingerprint,
        eval_gpu=reference.eval_gpu,
    )
    stored = await add_quant_calibration_reference(measured, config.psql_db)
    logger.info(
        f"Task {item.task.task_id} hotkey {item.hotkey}: calibration bf16 reference {stored.model} on {stored.sku} "
        f"is {stored.loss:.6f}"
    )


async def _store_shared_calibration_reference(item: ReferenceToMeasure, result: EvaluationResultText, config: Config) -> None:
    """Store the calibration reference a deployment scored next to its test reference (already stored).

    A deployment that returned none (the calibration rows did not score, or its image scores one example set) is not a
    failure of its own: the calibration reference is still missing, so the next cycle measures it alone, and that
    measurement counts its failures.
    """
    reference = result.quant_calibration_reference
    if reference is None:
        logger.warning(
            f"Task {item.task.task_id} hotkey {item.hotkey}: the deployment returned no {_label(True)} with the test "
            "reference; it is measured on its own next cycle"
        )
        return
    problem = _calibration_mismatch(item, reference)
    if problem is not None:
        await _record_failure(item, True, problem, config)
        return
    try:
        await _store_calibration_reference(item, reference, config)
    except Exception as e:
        await _record_failure(item, True, describe_error(e), config)
        return
    _CONSECUTIVE_FAILURES.pop(item.key_of(True), None)


async def _store_reference(item: ReferenceToMeasure, reference: QuantReferenceScore, config: Config) -> None:
    measured = QuantSkuReference(
        task_id=item.task.task_id,
        hotkey=item.hotkey,
        sku=reference.eval_gpu.reference_key,
        model=reference.model,
        loss=reference.eval_loss,
        per_example_losses=reference.per_example_losses,
        eval_set_fingerprint=reference.eval_set_fingerprint,
        eval_gpu=reference.eval_gpu,
    )
    stored = await add_quant_sku_reference(measured, config.psql_db)
    logger.info(
        f"Task {item.task.task_id} hotkey {item.hotkey}: bf16 reference {stored.model} on {stored.sku} is {stored.loss:.6f}"
    )


async def _measure(item: ReferenceToMeasure, config: Config) -> None:
    """One deployment for whatever the item is missing. Its own result is the test reference, or the calibration one
    when only that is missing; a failure there stores nothing. With both missing, the calibration reference comes back
    next to the test one and is stored after it."""
    calibration_only = not item.test
    label = _label(calibration_only)
    num_gpus = await asyncio.to_thread(compute_required_gpus, item.task)
    run_key = f"evaluation:{item.task.task_id}:prepass{'-calibration' if calibration_only else ''}:{item.hotkey}"
    await _start_cost_run(run_key, item, num_gpus, config)
    try:
        result = await _evaluate_reference(item, num_gpus)
    except (EvaluationCapacityUnavailable, DeploymentNotReadyError) as e:
        # Waiting for a GPU is the queue, not a failure: it does not count towards the failure alert.
        logger.info(f"Task {item.task.task_id} hotkey {item.hotkey}: {label} waits for capacity: {describe_error(e)}")
        await _finish_cost_run(run_key, False, config)
        return
    except Exception as e:
        result = e
    problem = _problem_with(result)
    if problem is None and calibration_only:
        problem = _calibration_mismatch(item, result.quant_reference)
    await _finish_cost_run(run_key, problem is None, config)
    if problem is not None:
        await _record_failure(item, calibration_only, problem, config)
        return
    _CONSECUTIVE_FAILURES.pop(item.key_of(calibration_only), None)
    if calibration_only:
        await _store_calibration_reference(item, result.quant_reference, config)
        return
    await _store_reference(item, result.quant_reference, config)
    if item.calibration:
        await _store_shared_calibration_reference(item, result, config)


async def _still_missing(item: ReferenceToMeasure, config: Config) -> ReferenceToMeasure | None:
    """The item less whatever the previous round's rows covered; None when nothing is left to measure."""
    test = item.test and not await _copy_previous_round_references(item, config)
    calibration = item.calibration and not await _copy_previous_round_calibration_reference(item, config)
    if not (test or calibration):
        return None
    return ReferenceToMeasure(task=item.task, hotkey=item.hotkey, repo=item.repo, test=test, calibration=calibration)


async def _fill_reference(item: ReferenceToMeasure, semaphore: asyncio.Semaphore, config: Config) -> None:
    keys = item.keys
    if any(key in _IN_PROGRESS for key in keys):
        return
    _IN_PROGRESS.update(keys)
    try:
        async with semaphore:
            remaining = await _still_missing(item, config)
            if remaining is not None:
                await _measure(remaining, config)
    except Exception as e:
        await _record_failure(item, not item.test, describe_error(e), config)
    finally:
        _IN_PROGRESS.difference_update(keys)


async def measure_missing_quant_references(config: Config) -> None:
    items = await references_to_measure(config)
    if not items:
        return
    logger.info(f"Quant reference pre-pass: {len(items)} (task, hotkey) reference(s) missing")
    semaphore = asyncio.Semaphore(cst.QUANT_REFERENCE_PREPASS_CONCURRENCY)
    await asyncio.gather(*(_fill_reference(item, semaphore, config) for item in items))


async def measure_quant_references_cycle(config: Config):
    """Same cadence as the other orchestrator cycles: the miner's training waits on this, so no slower."""
    while True:
        try:
            await measure_missing_quant_references(config)
        except Exception as e:
            logger.error(f"Error in measure_quant_references_cycle: {e}", exc_info=True)
        finally:
            await asyncio.sleep(cst.PROCESS_PENDING_TASKS_CYCLE_INTERVAL)
