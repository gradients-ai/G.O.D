"""Round 1 of an environment tournament always includes FORCED_R1_ENVIRONMENT (intercode)."""

from core.constants.environments import EnvironmentName
from validator.tournament import constants as t_cst
from validator.tournament import task_creator


def test_default_forced_r1_environment_is_intercode():
    assert t_cst.FORCED_R1_ENVIRONMENT == EnvironmentName.INTERCODE


def test_forced_environment_is_always_included_regardless_of_seen_last_tournament():
    for seen_last_tournament in (set(), {EnvironmentName.INTERCODE}, set(EnvironmentName)):
        selected = task_creator._select_r1_env_names(2, seen_last_tournament)
        assert t_cst.FORCED_R1_ENVIRONMENT in selected


def test_forced_environment_survives_across_many_shuffles():
    """The forced env is pulled to the front before truncation, not left to the random shuffle."""
    for _ in range(50):
        selected = task_creator._select_r1_env_names(2, seen_last_tournament=set())
        assert t_cst.FORCED_R1_ENVIRONMENT in selected


def test_num_envs_still_respected(monkeypatch):
    monkeypatch.setattr(t_cst, "FORCED_R1_ENVIRONMENT", None)
    selected = task_creator._select_r1_env_names(3, seen_last_tournament=set())
    assert len(selected) == 3


def test_forcing_disabled_when_constant_is_none(monkeypatch):
    monkeypatch.setattr(t_cst, "FORCED_R1_ENVIRONMENT", None)
    for _ in range(20):
        selected = task_creator._select_r1_env_names(1, seen_last_tournament=set())
        # With no forcing, a single-env pick need not be intercode every time.
        if EnvironmentName.INTERCODE not in selected:
            return
    raise AssertionError("expected at least one draw without intercode when forcing is disabled")
