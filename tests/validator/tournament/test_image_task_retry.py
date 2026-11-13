from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

import validator.tournament.constants as t_cst
from validator.tournament import task_creator


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(task_creator.asyncio, "sleep", AsyncMock())


async def test_gives_up_after_the_attempt_limit(monkeypatch):
    create = AsyncMock(side_effect=ValueError("no models of this family"))
    monkeypatch.setattr(task_creator, "create_synthetic_image_task", create)

    with pytest.raises(RuntimeError, match="after 5 attempts"):
        await task_creator._create_single_image_task_with_retry(MagicMock(), [], 0, is_final=True)

    assert create.await_count == t_cst.IMAGE_TASK_CREATION_MAX_ATTEMPTS


async def test_returns_the_task_once_an_attempt_succeeds(monkeypatch):
    task = object()
    create = AsyncMock(side_effect=[ValueError("transient"), task])
    monkeypatch.setattr(task_creator, "create_synthetic_image_task", create)

    assert await task_creator._create_single_image_task_with_retry(MagicMock(), [], 0, group_index=0) is task
    assert create.await_count == 2
