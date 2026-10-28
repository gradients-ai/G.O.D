"""Instruct datasets drop duplicate rows before the train/test split, so test can't leak into train."""

import pytest
from datasets import Dataset

from validator.tasks.datasets.preparation import drop_duplicate_instruct_rows
from validator.tasks.datasets.preparation import train_test_split


def _ds(rows: list[tuple[str, str]]) -> Dataset:
    return Dataset.from_dict({"instruct": [r[0] for r in rows], "output": [r[1] for r in rows]})


def test_exact_duplicates_dropped_first_kept():
    ds = drop_duplicate_instruct_rows(_ds([("a", "1"), ("b", "2"), ("a", "1"), ("c", "3"), ("b", "2")]))
    assert list(zip(ds["instruct"], ds["output"])) == [("a", "1"), ("b", "2"), ("c", "3")]


def test_no_duplicates_is_unchanged():
    rows = [("a", "1"), ("b", "2")]
    ds = drop_duplicate_instruct_rows(_ds(rows))
    assert list(zip(ds["instruct"], ds["output"])) == rows


@pytest.mark.asyncio
async def test_no_test_row_appears_in_train():
    rows = [(f"q{i % 50}", f"a{i % 50}") for i in range(1000)]  # 20x duplication
    split = await train_test_split(drop_duplicate_instruct_rows(_ds(rows)))
    train = set(zip(split["train"]["instruct"], split["train"]["output"]))
    test = set(zip(split["test"]["instruct"], split["test"]["output"]))
    assert train.isdisjoint(test)
