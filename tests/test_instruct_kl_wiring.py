"""
Wiring tests for optional KL-regularised instruct training (CPU-only, no GPU/axolotl).

These guard the plumbing we control end-to-end:
  - the rollout gate (maybe_get_kl_config)
  - the pivot regression: KL is a *task* concern and must NOT ride on the dataset_type
    carrier (otherwise it leaks into axolotl's dataset config)
  - use_kl/kl_coef live on the task model and on TrainRequestText
  - the env-var contract shared by the trainer and the evaluator

The KL *numerics* (KL(finetuned||base), eval-loss folding) are GPU-only and are
verified separately on a node, not here.
"""
from datetime import datetime
from unittest import mock
from uuid import uuid4

# NOTE: importing task_creator first sidesteps a pre-existing synthetic_scheduler <-> tournament
# circular import that only triggers when synthetic_scheduler is imported first.
import validator.tournament.task_creator  # noqa: F401
import core.constants as core_cst
from core.models.payload_models import TrainRequestText
from core.models.utility_models import FileFormat
from core.models.utility_models import InstructTextDatasetType
from validator.core import constants as vcst
from validator.core.models import DpoRawTask
from validator.core.models import InstructTextRawTask
from validator.cycle.util_functions import prepare_text_task_request
from validator.tasks.synthetic_scheduler import maybe_get_kl_config


def _instruct_task(**overrides) -> InstructTextRawTask:
    defaults = dict(
        is_organic=False,
        status="pending",
        model_id="Qwen/Qwen2.5-0.5B",
        ds="some/dataset",
        account_id=uuid4(),
        hours_to_complete=1.0,
        created_at=datetime.utcnow(),
        field_instruction="instruction",
        field_input="input",
        field_output="output",
        training_data="s3://bucket/data.json",
    )
    defaults.update(overrides)
    return InstructTextRawTask(**defaults)


# --- rollout gate ---------------------------------------------------------------

def test_maybe_get_kl_config_enabled_below_probability():
    with mock.patch("validator.tasks.synthetic_scheduler.random.random", return_value=0.0):
        use_kl, kl_coef = maybe_get_kl_config()
    assert use_kl is True
    assert kl_coef is not None
    assert vcst.INSTRUCT_KL_COEFFICIENT_MIN <= kl_coef <= vcst.INSTRUCT_KL_COEFFICIENT_MAX


def test_maybe_get_kl_config_disabled_above_probability():
    with mock.patch("validator.tasks.synthetic_scheduler.random.random", return_value=0.999):
        use_kl, kl_coef = maybe_get_kl_config()
    assert use_kl is False
    assert kl_coef is None


def test_kl_probability_and_coef_are_sane():
    assert 0.0 <= vcst.INSTRUCT_KL_TASK_PROBABILITY <= 1.0
    assert 0 < vcst.INSTRUCT_KL_COEFFICIENT_MIN <= vcst.INSTRUCT_KL_COEFFICIENT_MAX


# --- pivot regression: dataset carrier must stay clean --------------------------

def test_instruct_dataset_type_has_no_kl_fields():
    """KL must never appear in the dataset_type dict (it would leak into axolotl config)."""
    dumped = InstructTextDatasetType().model_dump()
    assert "use_kl" not in dumped
    assert "kl_coef" not in dumped


def test_prepare_text_task_request_keeps_kl_out_of_dataset_type():
    task = _instruct_task(use_kl=True, kl_coef=0.1)
    req = prepare_text_task_request(task)
    assert isinstance(req.dataset_type, InstructTextDatasetType)
    assert "use_kl" not in req.dataset_type.model_dump()
    assert "kl_coef" not in req.dataset_type.model_dump()


# --- KL rides on the task and the request --------------------------------------

def test_train_request_text_kl_defaults_off():
    req = TrainRequestText(
        model="m", task_id="t", hours_to_complete=1.0,
        dataset="d", dataset_type=InstructTextDatasetType(), file_format=FileFormat.S3,
    )
    assert req.use_kl is False
    assert req.kl_coef is None


def test_prepare_text_task_request_propagates_kl_from_instruct_task():
    task = _instruct_task(use_kl=True, kl_coef=0.1)
    req = prepare_text_task_request(task)
    assert req.use_kl is True
    assert req.kl_coef == 0.1


def test_prepare_text_task_request_defaults_off_for_non_kl_instruct_task():
    req = prepare_text_task_request(_instruct_task())
    assert req.use_kl is False
    assert req.kl_coef is None


def test_instruct_task_model_carries_kl_but_dpo_does_not():
    assert "use_kl" in InstructTextRawTask.model_fields
    assert "kl_coef" in InstructTextRawTask.model_fields
    # The isinstance guard at the shared call sites exists because non-instruct text
    # tasks have no such fields.
    assert "use_kl" not in DpoRawTask.model_fields


# --- env-var contract shared by trainer (sets) and evaluator (reads) ------------

def test_kl_env_var_names_are_stable():
    assert core_cst.USE_KL_ENV == "USE_KL"
    assert core_cst.KL_COEF_ENV == "KL_COEF"
