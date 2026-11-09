"""Near-duplicate datasets are rejected in data prep, before model prep spends a GPU."""

import random

import pytest
from datasets import Dataset

from core.near_duplicate import FORWARD_SAMPLE_SIZE
from core.near_duplicate import LENGTH_SAMPLE_SEED
from core.near_duplicate import LENGTH_SAMPLE_SIZE
from core.near_duplicate import near_duplicate_sample_indices
from validator.tasks.datasets import preparation
from validator.tasks.models import ChatRawTask
from validator.tasks.models import DpoRawTask
from validator.tasks.models import InstructTextRawTask


def _instruct_task(is_organic: bool = False) -> InstructTextRawTask:
    return InstructTextRawTask.model_construct(ds="some/dataset", is_organic=is_organic)


def _near_duplicate_rows(n: int = 50) -> Dataset:
    shared = "the quick brown fox jumps over the lazy dog near the old river bank today"
    return Dataset.from_dict({"instruct": [f"{shared} {i}" for i in range(n)], "output": ["same answer"] * n})


def _distinct_rows(n: int = 50) -> Dataset:
    rng = random.Random(0)
    return Dataset.from_dict(
        {
            "instruct": [" ".join(str(rng.randrange(10**9)) for _ in range(30)) for _ in range(n)],
            "output": [" ".join(str(rng.randrange(10**9)) for _ in range(30)) for _ in range(n)],
        }
    )


def test_sample_matches_model_prep_record_sampling():
    records = list(range(5_000))
    expected = random.Random(LENGTH_SAMPLE_SEED).sample(records, LENGTH_SAMPLE_SIZE)[:FORWARD_SAMPLE_SIZE]
    assert near_duplicate_sample_indices(len(records)) == expected


def test_sample_of_a_small_dataset_keeps_its_order():
    assert near_duplicate_sample_indices(3) == [0, 1, 2]


def test_instruct_text_joins_prompt_columns_then_output():
    dataset = Dataset.from_dict({"instruct": ["ask"], "input": ["more"], "output": ["answer"]})
    assert preparation._near_duplicate_texts(_instruct_task(), dataset) == ["ask more answer"]


def test_dpo_text_is_the_prompt_only():
    task = DpoRawTask.model_construct(ds="some/dataset", is_organic=False)
    dataset = Dataset.from_dict({"prompt": ["ask"], "chosen": ["good"], "rejected": ["bad"]})
    assert preparation._near_duplicate_texts(task, dataset) == ["ask"]


@pytest.mark.asyncio
async def test_near_duplicate_dataset_is_rejected():
    with pytest.raises(preparation.NearDuplicateDatasetError):
        await preparation.reject_near_duplicate_dataset(_instruct_task(), _near_duplicate_rows())


@pytest.mark.asyncio
async def test_distinct_dataset_is_allowed():
    await preparation.reject_near_duplicate_dataset(_instruct_task(), _distinct_rows())


@pytest.mark.asyncio
async def test_organic_task_is_exempt():
    await preparation.reject_near_duplicate_dataset(_instruct_task(is_organic=True), _near_duplicate_rows())


@pytest.mark.asyncio
async def test_chat_task_is_not_measured():
    task = ChatRawTask.model_construct(ds="some/dataset", is_organic=False)
    await preparation.reject_near_duplicate_dataset(task, _near_duplicate_rows())
