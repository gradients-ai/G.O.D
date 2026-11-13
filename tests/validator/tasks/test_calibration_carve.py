"""The calibration carve: size rule, determinism, disjointness from test, and the raw-row hash."""

import pytest

import validator.tasks.datasets.constants as cst
from validator.tasks.datasets.calibration import calibration_row_cap
from validator.tasks.datasets.calibration import carve_calibration
from validator.tasks.datasets.calibration import estimated_tokens
from validator.tasks.datasets.calibration import raw_row_hash


TASK_ID = "11111111-2222-3333-4444-555555555555"


def _hashes(n: int) -> list[str]:
    return [f"h{i}" for i in range(n)]


def test_raw_row_hash_ignores_key_order_and_covers_every_column():
    assert raw_row_hash({"a": 1, "b": "x"}) == raw_row_hash({"b": "x", "a": 1})
    assert raw_row_hash({"a": 1, "b": "x"}) != raw_row_hash({"a": 1, "b": "y", "extra": 2})
    assert len(raw_row_hash({"a": 1})) == 32


def test_row_cap_is_the_smaller_of_the_size_cap_and_ten_percent_of_train():
    assert calibration_row_cap(100_000) == cst.QUANT_CALIBRATION_MAX_ROWS
    assert calibration_row_cap(8_100) == 810


def test_short_rows_take_the_floor_not_the_token_target():
    # 20 chars -> 5 tokens a row: 250 rows is only 1,250 tokens, but the token target is far away and rows stop at the cap.
    n = 20_000
    carve = carve_calibration(TASK_ID, _hashes(n), set(), [20] * n)
    assert len(carve.calibration_indices) == cst.QUANT_CALIBRATION_MAX_ROWS


def test_long_rows_stop_at_the_token_target_but_never_under_the_row_floor():
    n = 20_000
    # 4,000 chars -> 1,000 tokens a row: 50 rows reach 50k tokens, the 250-row floor still applies.
    carve = carve_calibration(TASK_ID, _hashes(n), set(), [4_000] * n)
    assert len(carve.calibration_indices) == cst.QUANT_CALIBRATION_MIN_ROWS


def test_token_target_decides_between_floor_and_cap():
    n = 20_000
    # 400 chars -> 100 tokens a row: 500 rows reach 50k tokens.
    carve = carve_calibration(TASK_ID, _hashes(n), set(), [400] * n)
    assert len(carve.calibration_indices) == cst.QUANT_CALIBRATION_TARGET_TOKENS // 100


def test_tiny_dataset_carves_nothing():
    n = 2_499  # ten percent is 249 rows, under the floor
    assert carve_calibration(TASK_ID, _hashes(n), set(), [400] * n) is None
    n = 2_500
    assert len(carve_calibration(TASK_ID, _hashes(n), set(), [400] * n).calibration_indices) == 250


def test_same_task_id_same_rows_and_another_id_other_rows():
    n = 10_000
    first = carve_calibration(TASK_ID, _hashes(n), set(), [100] * n)
    again = carve_calibration(TASK_ID, _hashes(n), set(), [100] * n)
    other = carve_calibration("another-task", _hashes(n), set(), [100] * n)

    assert first == again
    assert first.calibration_indices != other.calibration_indices


def test_train_is_the_full_train_minus_calibration_in_original_order():
    n = 10_000
    carve = carve_calibration(TASK_ID, _hashes(n), set(), [100] * n)

    assert set(carve.calibration_indices).isdisjoint(carve.train_indices)
    assert sorted(carve.calibration_indices + carve.train_indices) == list(range(n))
    assert carve.train_indices == sorted(carve.train_indices)
    assert len(set(carve.calibration_indices)) == len(carve.calibration_indices)


def test_rows_equal_to_a_test_row_are_never_calibration_rows():
    n = 10_000
    hashes = _hashes(n)
    duplicated = {7, 8, 9, 1234}
    for index in duplicated:
        hashes[index] = "dup-of-test"
    carve = carve_calibration(TASK_ID, hashes, {"dup-of-test"}, [100] * n)

    assert duplicated.isdisjoint(carve.calibration_indices)
    assert duplicated <= set(carve.train_indices)  # still trained on


def test_a_pool_that_is_all_test_duplicates_carves_nothing():
    n = 10_000
    assert carve_calibration(TASK_ID, ["dup"] * n, {"dup"}, [100] * n) is None


@pytest.mark.parametrize("chars,expected", [(0, 0), (1, 1), (4, 1), (5, 2)])
def test_token_estimate_is_ceil_of_chars_over_four(chars, expected):
    assert estimated_tokens(chars) == expected
