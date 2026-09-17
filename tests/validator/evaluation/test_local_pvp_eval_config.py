import pytest

import core.constants.environments as env_cst
import validator.evaluation.constants as vcst
from ops.validator_ops.run_evaluation import _parse_base_chain_args
from validator.evaluation.local_evaluation import _build_pvp_pair_config
from validator.evaluation.local_evaluation import run_evaluation_local_pvp_pairs
from validator.evaluation.pvp.models import PvPEvalMetadata
from validator.evaluation.pvp.models import PvPGroupResults
from validator.evaluation.pvp.models import PvPPairResult


def test_build_pvp_pair_config_places_base_chains_and_ports():
    config = _build_pvp_pair_config(
        model_a_repo="org/model-a",
        model_b_repo="org/model-b",
        base_model="org/foundation",
        environment_names=[env_cst.EnvironmentName.OTHELLO],
        seed=1824484340,
        temperature=0.0,
        base_chain_a=["org/prev-a"],
        base_chain_b=["org/prev-b"],
    )

    assert config.model_a is not None
    assert config.model_b is not None
    assert config.model_a.repo == "org/model-a"
    assert config.model_b.repo == "org/model-b"
    assert config.model_a.base_chain == ["org/prev-a"]
    assert config.model_b.base_chain == ["org/prev-b"]
    assert config.model_a.gpu_id == 0
    assert config.model_b.gpu_id == 1
    assert config.model_a.port == vcst.PVP_SGLANG_PORT_A
    assert config.model_b.port == vcst.PVP_SGLANG_PORT_B
    assert config.seed == 1824484340
    assert config.temperature == 0.0
    assert env_cst.EnvironmentName.OTHELLO in config.matchups


@pytest.mark.asyncio
async def test_run_evaluation_local_pvp_pairs_uses_canonical_order_and_base_chains(monkeypatch):
    calls: list[dict] = []

    async def fake_pair(**kwargs):
        calls.append(kwargs)
        return PvPGroupResults(
            base_model=kwargs["base_model"],
            hotkeys=[kwargs["hotkey_a"], kwargs["hotkey_b"]],
            pair_results=[
                PvPPairResult(
                    hotkey_a=kwargs["hotkey_a"],
                    hotkey_b=kwargs["hotkey_b"],
                    results={},
                )
            ],
            metadata=PvPEvalMetadata(seed=kwargs["seed"], temperature=kwargs["temperature"], wall_time_seconds=0),
        )

    monkeypatch.setattr("validator.evaluation.local_evaluation.run_evaluation_local_pvp_pair", fake_pair)
    monkeypatch.setattr("validator.evaluation.local_evaluation._get_shared_pvp_eval_image", lambda _envs: "pvp-eval:test")

    # Insertion order deliberately non-canonical (Z before A).
    miner_repos = {
        "hkZ": "org/repo-z",
        "hkA": "org/repo-a",
    }
    base_chains = {
        "hkA": ["org/prev-a"],
        "hkZ": ["org/prev-z"],
    }

    results = await run_evaluation_local_pvp_pairs(
        miner_repos=miner_repos,
        original_model="org/foundation",
        environment_names=[env_cst.EnvironmentName.CLOBBER],
        gpu_ids=[0, 1],
        eval_seed=42,
        temperature=0.0,
        base_chains=base_chains,
    )

    assert len(calls) == 1
    assert calls[0]["hotkey_a"] == "hkA"
    assert calls[0]["hotkey_b"] == "hkZ"
    assert calls[0]["model_a_repo"] == "org/repo-a"
    assert calls[0]["model_b_repo"] == "org/repo-z"
    assert calls[0]["base_chain_a"] == ["org/prev-a"]
    assert calls[0]["base_chain_b"] == ["org/prev-z"]
    assert results.hotkeys == ["hkA", "hkZ"]
    assert results.pair_results[0].hotkey_a == "hkA"
    assert results.pair_results[0].hotkey_b == "hkZ"


def test_parse_base_chain_args_single_and_multi_repo():
    miner_repos = {"hkA": "org/a", "hkB": "org/b"}
    parsed = _parse_base_chain_args(
        ["hkA=org/prev-a", "hkB=org/prev-b1,org/prev-b2"],
        miner_repos,
    )
    assert parsed == {
        "hkA": ["org/prev-a"],
        "hkB": ["org/prev-b1", "org/prev-b2"],
    }


def test_parse_base_chain_args_rejects_unknown_hotkey():
    miner_repos = {"hkA": "org/a", "hkB": "org/b"}
    with pytest.raises(ValueError, match="Unknown hotkey"):
        _parse_base_chain_args(["hkC=org/prev-c"], miner_repos)


def test_parse_base_chain_args_rejects_malformed_and_duplicates():
    miner_repos = {"hkA": "org/a"}
    with pytest.raises(ValueError, match="Invalid --base_chain"):
        _parse_base_chain_args(["hkA"], miner_repos)
    with pytest.raises(ValueError, match="Duplicate --base_chain"):
        _parse_base_chain_args(["hkA=org/prev-a", "hkA=org/other"], miner_repos)


def test_parse_base_chain_args_empty():
    assert _parse_base_chain_args(None, {"hkA": "org/a"}) == {}
    assert _parse_base_chain_args([], {"hkA": "org/a"}) == {}
