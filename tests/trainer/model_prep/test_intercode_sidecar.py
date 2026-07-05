import importlib
import sys
import types

import pytest

from core.constants.docker import VALIDATOR_DOCKER_IMAGE_INTERCODE
from core.constants.environments import EnvironmentName
from validator.tasks.prep.model import _build_env_configs


def test_build_env_configs_restricts_to_task_envs():
    only = _build_env_configs([EnvironmentName.GOOFSPIEL])
    assert set(only) == {EnvironmentName.GOOFSPIEL}

    pair = _build_env_configs([EnvironmentName.GOOFSPIEL, EnvironmentName.OTHELLO])
    assert set(pair) == {EnvironmentName.GOOFSPIEL, EnvironmentName.OTHELLO}


def test_build_env_configs_defaults_to_all_envs():
    from core.constants.environments import ENVIRONMENT_CONFIGS

    assert set(_build_env_configs()) == set(ENVIRONMENT_CONFIGS)
    assert set(_build_env_configs([])) == set(ENVIRONMENT_CONFIGS)


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


def test_model_prep_configs_are_time_budgeted_not_fixed_counts():
    configs = _build_env_configs()

    assert configs[EnvironmentName.LIARS_DICE].num_episodes == 0
    assert configs[EnvironmentName.GIN_RUMMY].num_episodes == 0
    assert configs[EnvironmentName.INTERCODE].num_episodes == 0


def test_intercode_sidecar_formats_empty_exceptions(monkeypatch):
    fake_eval_intercode = types.ModuleType("validator.evaluation.evaluators.intercode")
    fake_eval_intercode.DEFAULT_MAX_TOKENS_PER_CALL = 512
    fake_eval_intercode.DEFAULT_MAX_TURNS = 10
    fake_eval_intercode.DEFAULT_PER_TASK_TIMEOUT_SECONDS = 150
    fake_eval_intercode.InterCodeAssets = object
    fake_eval_intercode.load_intercode_assets = lambda: object()

    async def fake_run_intercode_task(*args, **kwargs):
        return 0.0

    fake_eval_intercode.run_intercode_task = fake_run_intercode_task
    monkeypatch.setitem(sys.modules, "validator.evaluation.evaluators.intercode", fake_eval_intercode)
    sys.modules.pop("validator.evaluation.intercode_server", None)
    try:
        intercode_server = importlib.import_module("validator.evaluation.intercode_server")

        assert intercode_server._format_exception(TimeoutError()) == "TimeoutError"
    finally:
        sys.modules.pop("validator.evaluation.intercode_server", None)


def test_start_env_sidecars_passes_intercode_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", types.ModuleType("pynvml"))

    import trainer.runtime as trainer_runtime

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

    monkeypatch.setattr(trainer_runtime, "ensure_internal_network", lambda: None)
    monkeypatch.setattr(trainer_runtime, "run_environment_server_container", fake_run_environment_server_container)
    monkeypatch.setattr(trainer_runtime, "_resolve_container_ip", fake_resolve_container_ip)

    env_url_map, containers = trainer_runtime._start_env_sidecars({EnvironmentName.INTERCODE: intercode_cfg}, {})

    assert env_url_map == {EnvironmentName.INTERCODE: "http://10.0.0.42:8000"}
    assert len(containers) == 1
    assert calls == [
        {
            "env_name": EnvironmentName.INTERCODE,
            "image": VALIDATOR_DOCKER_IMAGE_INTERCODE,
            "command": intercode_cfg.env_server_command,
        }
    ]


def test_start_env_sidecars_skips_in_harness_games(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", types.ModuleType("pynvml"))

    import trainer.runtime as trainer_runtime

    configs = _build_env_configs()
    calls = []

    async def fake_run_environment_server_container(env_name, log_labels, image=None, command=None):
        calls.append(image)
        return object()

    async def fake_resolve_container_ip(container):
        return "10.0.0.42"

    monkeypatch.setattr(trainer_runtime, "ensure_internal_network", lambda: None)
    monkeypatch.setattr(trainer_runtime, "run_environment_server_container", fake_run_environment_server_container)
    monkeypatch.setattr(trainer_runtime, "_resolve_container_ip", fake_resolve_container_ip)

    env_url_map, containers = trainer_runtime._start_env_sidecars(
        {
            EnvironmentName.OTHELLO: configs[EnvironmentName.OTHELLO],
            EnvironmentName.INTERCODE: configs[EnvironmentName.INTERCODE],
        },
        {},
    )

    assert calls == [VALIDATOR_DOCKER_IMAGE_INTERCODE]
    assert env_url_map == {EnvironmentName.INTERCODE: "http://10.0.0.42:8000"}
    assert len(containers) == 1


def test_start_env_sidecars_failure_degrades_instead_of_raising(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", types.ModuleType("pynvml"))

    import trainer.runtime as trainer_runtime

    intercode_cfg = _build_env_configs()[EnvironmentName.INTERCODE]

    async def exploding_run_environment_server_container(env_name, log_labels, image=None, command=None):
        raise RuntimeError("image pull failed")

    monkeypatch.setattr(trainer_runtime, "ensure_internal_network", lambda: None)
    monkeypatch.setattr(
        trainer_runtime, "run_environment_server_container", exploding_run_environment_server_container
    )

    env_url_map, containers = trainer_runtime._start_env_sidecars({EnvironmentName.INTERCODE: intercode_cfg}, {})

    assert env_url_map == {}
    assert containers == []


def test_in_harness_envs_match_agent_registry():
    pytest.importorskip("pyspiel")
    pytest.importorskip("open_spiel")

    from core.constants.environments import ENVIRONMENT_CONFIGS
    from core.constants.environments import EvalType
    from core.pvp.game_eval import _AGENT_REGISTRY

    pvp_envs = {name for name, cfg in ENVIRONMENT_CONFIGS.items() if cfg.eval_type == EvalType.PVP}
    assert pvp_envs == set(_AGENT_REGISTRY)


def test_training_env_server_selection_skips_intercode(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", types.ModuleType("pynvml"))

    import trainer.runtime as trainer_runtime

    assert trainer_runtime._select_training_env_server_name(
        [EnvironmentName.INTERCODE, EnvironmentName.LIARS_DICE]
    ) == EnvironmentName.LIARS_DICE
    assert trainer_runtime._select_training_env_server_name(
        [EnvironmentName.INTERCODE]
    ) is None
    assert trainer_runtime._select_training_env_server_name([]) is None


@pytest.mark.asyncio
async def test_run_environment_server_container_resolves_intercode_config(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", types.ModuleType("pynvml"))

    import trainer.runtime as trainer_runtime

    captured = {}
    expected_container = object()

    class FakeContainers:
        def run(self, **kwargs):
            captured.update(kwargs)
            return expected_container

    class FakeDockerClient:
        containers = FakeContainers()

    monkeypatch.setattr(trainer_runtime, "ensure_internal_network", lambda: None)
    monkeypatch.setattr(trainer_runtime.docker, "from_env", lambda: FakeDockerClient())

    container = await trainer_runtime.run_environment_server_container(EnvironmentName.INTERCODE, {})

    assert container is expected_container
    assert captured["image"] == VALIDATOR_DOCKER_IMAGE_INTERCODE
    assert captured["command"] == _build_env_configs()[EnvironmentName.INTERCODE].env_server_command
