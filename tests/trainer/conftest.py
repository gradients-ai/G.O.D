import pytest

from trainer import constants as cst
from trainer import host_resources
from trainer import image_builds
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
def image_build_cache_state_in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Trainer tests never read or write the real image build cache state file."""
    monkeypatch.setattr(cst, "IMAGE_BUILD_CACHE_STATE_PATH", str(tmp_path / "image_build_cache_state.json"))
    monkeypatch.setattr(image_builds, "_emptied_at_in_memory", None)
