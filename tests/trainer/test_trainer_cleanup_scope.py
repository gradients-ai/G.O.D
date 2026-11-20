import re
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trainer import cleanup
from trainer import constants as cst


TRAINER_DIR = Path(cleanup.__file__).parent


def _container(name: str, status: str = "exited", age_hours: float = 2.0) -> MagicMock:
    container = MagicMock()
    container.name = name
    container.id = f"{name}-id"
    container.status = status
    container.attrs = {"Created": (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()}
    return container


def _client(containers: list[MagicMock]) -> MagicMock:
    client = MagicMock()
    client.containers.list.return_value = containers
    return client


@pytest.mark.parametrize("prefix", cst.TRAINER_CONTAINER_NAME_PREFIXES)
def test_removes_stale_trainer_containers_with_their_anonymous_volumes(prefix):
    container = _container(f"{prefix}abc123")
    cleanup.remove_stale_trainer_containers(_client([container]))
    container.remove.assert_called_once_with(force=True, v=True)


@pytest.mark.parametrize("name", ["postgres", "other-minio", "my-text-trainer-x", "trainer-vector", ""])
def test_leaves_other_containers_on_the_host_alone(name):
    container = _container(name)
    cleanup.remove_stale_trainer_containers(_client([container]))
    container.remove.assert_not_called()


@pytest.mark.parametrize(
    "status, age_hours, removed",
    [("exited", 2.0, True), ("created", 2.0, True), ("exited", 0.5, False), ("running", 5.0, False)],
)
def test_only_stale_stopped_trainer_containers_go(status, age_hours, removed):
    container = _container("text-trainer-abc", status=status, age_hours=age_hours)
    cleanup.remove_stale_trainer_containers(_client([container]))
    assert container.remove.called is removed


def test_never_prunes_host_volumes():
    client = _client([_container("text-trainer-abc")])
    cleanup.remove_stale_trainer_containers(client)
    client.volumes.prune.assert_not_called()


def test_every_trainer_container_name_is_under_a_cleanup_prefix():
    # Every f-string container name the trainer assigns; a new one must be added to TRAINER_CONTAINER_NAME_PREFIXES.
    pattern = re.compile(r"""(?:container_name\s*=|name=)\s*f["']([a-z-]+)""")
    prefixes = {match for path in TRAINER_DIR.rglob("*.py") for match in pattern.findall(path.read_text())}
    assert {"text-trainer-", "downloader-", "quant-decode-"} <= prefixes  # the scan finds the call sites
    assert all(prefix.startswith(cst.TRAINER_CONTAINER_NAME_PREFIXES) for prefix in prefixes), prefixes
