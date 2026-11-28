import pytest

from trainer import constants as cst
from trainer import host_resources
from trainer.host_resources import HostTopology
from trainer.host_resources import plan_host


@pytest.fixture(autouse=True)
def fake_host_plan(monkeypatch: pytest.MonkeyPatch):
    """Trainer tests run without GPUs: container starts see a 40-CPU, 2-GPU, 256 GiB host."""
    topology = HostTopology(
        cpus=list(range(40)),
        cores=[[2 * i, 2 * i + 1] for i in range(20)],
        memory_bytes=256 * 1024**3,
        gpu_local_cpus=[None, None],
    )
    monkeypatch.setattr(host_resources, "_host_plan", plan_host(topology))


@pytest.fixture(autouse=True)
def decoded_files_are_kept_whatever_the_free_disk(monkeypatch: pytest.MonkeyPatch):
    """Whether a pending upload keeps its decoded files must not depend on the disk the tests run on."""
    monkeypatch.setattr(cst, "QUANT_DECODE_UPLOAD_RETRY_MIN_FREE_DISK_BYTES", 0)
