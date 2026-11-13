"""Carve a calibration set from the train side of a text task's split (see docs/miner_compression.md, "The calibration set").

Pure functions over row hashes and character counts, so the split code and the tests share one definition. The
test split is drawn first and never changes; the carve only takes rows out of what is left for training.
"""

import hashlib
import json
import random
from math import ceil

from datasets import Dataset
from pydantic import BaseModel

import validator.tasks.datasets.constants as cst
from core.logging import get_logger
from validator.tournament import constants as t_cst


logger = get_logger(__name__)

class CalibrationUnavailableError(ValueError):
    """A task that must carry a calibration set could not be given one; its data prep fails."""


ROW_HASH_COLUMN = "__calibration_row_hash__"
_ROW_HASH_HEX_CHARS = 32


def raw_row_hash(row: dict) -> str:
    """Hash of a raw dataset row, over every raw column, so it does not depend on which columns a task maps."""
    return hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:_ROW_HASH_HEX_CHARS]


def with_row_hashes(dataset: Dataset) -> Dataset:
    """dataset plus a ROW_HASH_COLUMN of each raw row's hash; later column renames and the split keep it aligned."""
    return dataset.add_column(ROW_HASH_COLUMN, [raw_row_hash(row) for row in dataset])


class CalibrationCarve(BaseModel):
    """Indices into the train side: the calibration rows in file order, and the train rows left, in original order."""

    calibration_indices: list[int]
    train_indices: list[int]


def shared_split_seed(ds: str) -> str:
    """Seed of every row sample drawn from one split: the chunk index for a continuous quant task, else its ds.

    The ds label of a continuous chunk comes from the content service and differs per call for the same chunk, so the
    two tasks on one chunk would otherwise sample different rows. Used by the eval-row cap (validator.tasks.requests)
    and by the given-split calibration carve, so both sibling tasks see identical rows.
    """
    train_index = t_cst.continuous_quant_train_index(ds)
    return ds if train_index is None else f"{t_cst.QUANT_CONTINUOUS_DS_PREFIX}:{train_index}"


def calibration_row_cap(n_train: int) -> int:
    """The most rows a task of n_train train rows may give up: the size cap and the share-of-train cap."""
    return min(cst.QUANT_CALIBRATION_MAX_ROWS, int(n_train * cst.QUANT_CALIBRATION_MAX_TRAIN_FRACTION))


def estimated_tokens(completion_chars: int) -> int:
    """Supervised tokens of a row from its completion's characters; only sizes the carve."""
    return ceil(completion_chars / cst.QUANT_CALIBRATION_CHARS_PER_TOKEN)


def carve_calibration(
    seed_key: str,
    train_hashes: list[str],
    test_hashes: set[str],
    completion_chars: list[int],
) -> CalibrationCarve | None:
    """Pick the calibration rows from the train side, or None when the task is too small to give any up.

    seed_key seeds the shuffle: the task id for a task that splits its own dataset, shared_split_seed(ds) for a carve
    from a given split, so sibling tasks on one chunk carve identical rows.

    Rows are taken in a seeded shuffle of the train indices, skipping any whose raw hash equals a test row's (a
    dataset duplicate must not leak test content), until at least QUANT_CALIBRATION_MIN_ROWS rows and
    QUANT_CALIBRATION_TARGET_TOKENS estimated tokens, or the row cap.
    """
    n_train = len(train_hashes)
    cap = calibration_row_cap(n_train)
    if cap < cst.QUANT_CALIBRATION_MIN_ROWS:
        logger.info(f"Calibration: carving nothing, {n_train} train rows allow only {cap} < {cst.QUANT_CALIBRATION_MIN_ROWS}")
        return None

    order = list(range(n_train))
    random.Random(f"{seed_key}:calibration").shuffle(order)

    chosen: list[int] = []
    tokens = 0
    for index in order:
        if train_hashes[index] in test_hashes:
            continue
        chosen.append(index)
        tokens += estimated_tokens(completion_chars[index])
        if len(chosen) >= cap:
            break
        if len(chosen) >= cst.QUANT_CALIBRATION_MIN_ROWS and tokens >= cst.QUANT_CALIBRATION_TARGET_TOKENS:
            break

    if len(chosen) < cst.QUANT_CALIBRATION_MIN_ROWS:
        logger.info(f"Calibration: carving nothing, only {len(chosen)} train rows are not duplicates of a test row")
        return None

    taken = set(chosen)
    logger.info(f"Calibration: carved {len(chosen)} of {n_train} train rows (~{tokens} supervised tokens estimated)")
    return CalibrationCarve(calibration_indices=chosen, train_indices=[i for i in range(n_train) if i not in taken])
