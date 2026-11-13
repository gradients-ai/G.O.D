"""The trainer hands a freeform miner its calibration set as a file; any failure drops it without failing the job."""

import io
import json

import pytest

import core.constants as core_cst
from core.models.quant_models import QuantCalibrationReference
from core.models.quant_models import QuantCalibrationRequest
from core.models.quant_models import QuantCalibrationSet
from trainer import constants as cst
from trainer import quant_calibration
from validator.evaluation.quant_loss import QuantEvalExample
from validator.evaluation.quant_loss import quant_eval_set_fingerprint


TASK_ID = "task-123"
HOTKEY = "5HotkeyLongerThanEight"
URL = "https://minio.example/cal.jsonl?X-Amz-Signature=secret"
EXAMPLES = [
    QuantEvalExample(input_ids=[1, 2, 3], labels=[-100, 2, 3]),
    QuantEvalExample(input_ids=[1, 5], labels=[-100, -100]),
    QuantEvalExample(input_ids=[4, 6, 7, 8], labels=[-100, 6, 7, 8]),
]
JSONL = "".join(example.model_dump_json() + "\n" for example in EXAMPLES)
FINGERPRINT = quant_eval_set_fingerprint(EXAMPLES)
REFERENCE = QuantCalibrationReference(model="org/own-bf16", loss=2.0, per_example_losses=[1.5, None, 2.5], sku="h100-sxm")


def _request(reference=REFERENCE, fingerprint=FINGERPRINT, n=3) -> QuantCalibrationRequest:
    return QuantCalibrationRequest(examples_url=URL, eval_set_fingerprint=fingerprint, n_examples=n, reference=reference)


@pytest.fixture
def served(monkeypatch):
    """Whatever the 'bucket' serves; records the URLs asked for."""
    state = {"body": JSONL, "error": None, "asked": []}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(url, timeout):
        state["asked"].append(url)
        if state["error"]:
            raise state["error"]
        return Response(state["body"].encode())

    monkeypatch.setattr(quant_calibration.urllib.request, "urlopen", urlopen)
    return state


def test_filename_matches_the_baseline_stats_convention():
    assert quant_calibration.calibration_filename(TASK_ID, HOTKEY) == "quant_calibration_task-123_5HotkeyL.json"


async def test_file_is_written_with_zipped_losses_and_the_env_var_points_at_it(served, tmp_path):
    env = await quant_calibration.write_quant_calibration_file(TASK_ID, HOTKEY, _request(), str(tmp_path))

    assert env == {core_cst.QUANT_CALIBRATION_PATH_ENV: f"{cst.CACHE_ROOT_PATH}/quant_calibration_task-123_5HotkeyL.json"}
    written = QuantCalibrationSet.model_validate(json.loads((tmp_path / "quant_calibration_task-123_5HotkeyL.json").read_text()))
    assert (written.task_id, written.eval_set_fingerprint) == (TASK_ID, FINGERPRINT)
    assert (written.reference_model, written.reference_loss, written.reference_sku) == ("org/own-bf16", 2.0, "h100-sxm")
    assert [e.reference_loss for e in written.examples] == [1.5, None, 2.5]
    assert [(e.input_ids, e.labels) for e in written.examples] == [(x.input_ids, x.labels) for x in EXAMPLES]


async def test_without_a_reference_the_examples_carry_none_losses(served, tmp_path):
    env = await quant_calibration.write_quant_calibration_file(TASK_ID, HOTKEY, _request(reference=None), str(tmp_path))

    assert core_cst.QUANT_CALIBRATION_PATH_ENV in env
    written = QuantCalibrationSet.model_validate(json.loads((tmp_path / "quant_calibration_task-123_5HotkeyL.json").read_text()))
    assert written.reference_model is None and written.reference_loss is None and written.reference_sku is None
    assert [e.reference_loss for e in written.examples] == [None, None, None]


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"fingerprint": "not-the-fingerprint"},
        {"n": 2},
        {"reference": QuantCalibrationReference(model="m", loss=1.0, per_example_losses=[1.0], sku="s")},
    ],
)
async def test_a_mismatch_with_the_request_writes_nothing_and_sets_no_env(served, tmp_path, request_kwargs):
    env = await quant_calibration.write_quant_calibration_file(TASK_ID, HOTKEY, _request(**request_kwargs), str(tmp_path))

    assert env == {} and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", [OSError("connection reset"), TimeoutError()])
async def test_a_download_failure_writes_nothing_and_does_not_raise(served, tmp_path, failure):
    served["error"] = failure

    assert await quant_calibration.write_quant_calibration_file(TASK_ID, HOTKEY, _request(), str(tmp_path)) == {}
    assert list(tmp_path.iterdir()) == []


async def test_a_malformed_file_writes_nothing(served, tmp_path):
    served["body"] = "not json\n"

    assert await quant_calibration.write_quant_calibration_file(TASK_ID, HOTKEY, _request(), str(tmp_path)) == {}


async def test_an_unwritable_volume_does_not_raise(served, tmp_path):
    assert await quant_calibration.write_quant_calibration_file(TASK_ID, HOTKEY, _request(), str(tmp_path / "missing")) == {}


async def test_the_url_is_never_logged(served, tmp_path, monkeypatch):
    lines = []
    monkeypatch.setattr(quant_calibration.logger, "warning", lines.append)
    served["error"] = OSError(f"failed to reach {URL}")

    await quant_calibration.write_quant_calibration_file(TASK_ID, HOTKEY, _request(), str(tmp_path))

    assert lines and not any("X-Amz-Signature" in line or "minio.example" in line for line in lines)
