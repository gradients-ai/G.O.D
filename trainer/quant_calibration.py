"""Hand a freeform miner its calibration set as a file in the cache volume (QUANT_CALIBRATION_PATH).

The host has network, the training container does not, so the trainer fetches the validator's tokenised JSONL, checks
it is the set the request names, zips it with the bf16 reference's per-example losses into a QuantCalibrationSet and
writes it next to baseline_stats. Any failure logs the type only (the URL is presigned) and returns None: a missing
calibration set must never fail a training job.
"""

import asyncio
import json
import os
import urllib.request

import core.constants as core_cst
from core.models.quant_models import QuantCalibrationExample
from core.models.quant_models import QuantCalibrationRequest
from core.models.quant_models import QuantCalibrationSet
from trainer import constants as cst
from trainer.telemetry import logger
from validator.evaluation.quant_loss import QuantEvalExample
from validator.evaluation.quant_loss import quant_eval_set_fingerprint


def calibration_filename(task_id: str, hotkey: str) -> str:
    return f"quant_calibration_{task_id}_{hotkey[:8]}.json"


def _fetch(url: str) -> str:
    """The JSONL text, under a hard size cap."""
    with urllib.request.urlopen(url, timeout=cst.QUANT_CALIBRATION_FETCH_TIMEOUT_SECONDS) as response:
        data = response.read(cst.QUANT_CALIBRATION_MAX_BYTES + 1)
    if len(data) > cst.QUANT_CALIBRATION_MAX_BYTES:
        raise ValueError(f"calibration file exceeds {cst.QUANT_CALIBRATION_MAX_BYTES} bytes")
    return data.decode()


def build_calibration_set(task_id: str, request: QuantCalibrationRequest, jsonl: str) -> QuantCalibrationSet:
    """The file's content; raises ValueError when the examples are not the ones the request names."""
    examples = [QuantEvalExample.model_validate(json.loads(line)) for line in jsonl.splitlines() if line.strip()]
    if len(examples) != request.n_examples:
        raise ValueError(f"calibration file has {len(examples)} examples, the request says {request.n_examples}")
    if quant_eval_set_fingerprint(examples) != request.eval_set_fingerprint:
        raise ValueError("calibration file fingerprint does not match the request")
    reference = request.reference
    if reference is not None and len(reference.per_example_losses) != len(examples):
        raise ValueError("calibration reference has a loss count that does not match the examples")
    losses = reference.per_example_losses if reference is not None else [None] * len(examples)
    return QuantCalibrationSet(
        task_id=task_id,
        eval_set_fingerprint=request.eval_set_fingerprint,
        reference_model=reference.model if reference else None,
        reference_loss=reference.loss if reference else None,
        reference_sku=reference.sku if reference else None,
        examples=[
            QuantCalibrationExample(input_ids=example.input_ids, labels=example.labels, reference_loss=loss)
            for example, loss in zip(examples, losses)
        ],
    )


def _write(volume_dir: str, filename: str, calibration_set: QuantCalibrationSet) -> None:
    with open(os.path.join(volume_dir, filename), "w") as f:
        f.write(calibration_set.model_dump_json())


async def write_quant_calibration_file(
    task_id: str, hotkey: str, request: QuantCalibrationRequest, volume_dir: str
) -> dict[str, str]:
    """{QUANT_CALIBRATION_PATH_ENV: container path} once the file is in volume_dir; {} on any failure."""
    try:
        jsonl = await asyncio.to_thread(_fetch, request.examples_url)
        calibration_set = build_calibration_set(task_id, request, jsonl)
        filename = calibration_filename(task_id, hotkey)
        await asyncio.to_thread(_write, volume_dir, filename, calibration_set)
    except Exception as e:
        logger.warning(f"Task {task_id}: no calibration set for the miner ({type(e).__name__}); training without it")
        return {}
    return {core_cst.QUANT_CALIBRATION_PATH_ENV: os.path.join(cst.CACHE_ROOT_PATH, filename)}
