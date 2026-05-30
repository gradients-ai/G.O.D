import sys
import types

import pytest

from core.constants import VALIDATOR_DOCKER_IMAGE_INTERCODE
from core.constants import EnvironmentName
from validator.utils.model_prep import _build_env_configs


def test_model_prep_configs_include_intercode_sidecar():
    cfg = _build_env_configs()[EnvironmentName.INTERCODE]

    assert cfg.env_image == VALIDATOR_DOCKER_IMAGE_INTERCODE
    assert cfg.env_server_command == [
        "python",
        "-m",
        "uvicorn",
        "validator.evaluation.intercode_server:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]


def test_start_env_sidecars_passes_intercode_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", types.ModuleType("pynvml"))

    from trainer import image_manager

    intercode_cfg = _build_env_configs()[EnvironmentName.INTERCODE]
    calls = []

    async def fake_run_environment_server_container(env_name, log_labels, image=None, command=None):
        calls.append(
            {
                "env_name": env_name,
                "image": image,
                "command": command,
            }
        )
        return object()

    async def fake_resolve_container_ip(container):
        return "10.0.0.42"

    monkeypatch.setattr(image_manager, "ensure_internal_network", lambda: None)
    monkeypatch.setattr(image_manager, "run_environment_server_container", fake_run_environment_server_container)
    monkeypatch.setattr(image_manager, "_resolve_container_ip", fake_resolve_container_ip)

    env_url_map, containers = image_manager._start_env_sidecars({EnvironmentName.INTERCODE: intercode_cfg}, {})

    assert env_url_map == {EnvironmentName.INTERCODE: "http://10.0.0.42:8000"}
    assert len(containers) == 1
    assert calls == [
        {
            "env_name": EnvironmentName.INTERCODE,
            "image": VALIDATOR_DOCKER_IMAGE_INTERCODE,
            "command": intercode_cfg.env_server_command,
        }
    ]


def test_training_env_server_selection_skips_intercode(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", types.ModuleType("pynvml"))

    from trainer import image_manager

    assert image_manager._select_training_env_server_name(
        [EnvironmentName.INTERCODE, EnvironmentName.LIARS_DICE]
    ) == EnvironmentName.LIARS_DICE
    assert image_manager._select_training_env_server_name(
        [EnvironmentName.INTERCODE]
    ) is None
    assert image_manager._select_training_env_server_name([]) is None


@pytest.mark.asyncio
async def test_run_environment_server_container_resolves_intercode_config(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", types.ModuleType("pynvml"))

    from trainer import image_manager

    captured = {}
    expected_container = object()

    class FakeContainers:
        def run(self, **kwargs):
            captured.update(kwargs)
            return expected_container

    class FakeDockerClient:
        containers = FakeContainers()

    monkeypatch.setattr(image_manager, "ensure_internal_network", lambda: None)
    monkeypatch.setattr(image_manager.docker, "from_env", lambda: FakeDockerClient())

    container = await image_manager.run_environment_server_container(EnvironmentName.INTERCODE, {})

    assert container is expected_container
    assert captured["image"] == VALIDATOR_DOCKER_IMAGE_INTERCODE
    assert captured["command"] == _build_env_configs()[EnvironmentName.INTERCODE].env_server_command
