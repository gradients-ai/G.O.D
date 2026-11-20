"""Ops CLI for freeform publications: read-only by default, every write behind an explicit flag (mocked, no DB)."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from ops.tools.quant import reconcile_freeform_publications as cli
from validator.tournament.models import FreeformReconcileReport
from validator.tournament.models import PendingDecoderRelease


PENDING = PendingDecoderRelease(tournament_id="tourn_1", hotkey="5Champion")
BLOCKED = PendingDecoderRelease(tournament_id="tourn_0", hotkey="5Old")


@pytest.fixture
def calls(monkeypatch):
    report = FreeformReconcileReport(
        pending_releases=[PENDING], blocked_releases=[BLOCKED], pending_decoded_tournaments=["tourn_1"], applied=False
    )
    state = SimpleNamespace(
        reconcile=AsyncMock(return_value=report),
        release=AsyncMock(return_value="https://huggingface.co/datasets/o/r/resolve/main/x"),
        publish=AsyncMock(return_value=["org/m-decoded"]),
        wait=AsyncMock(),
        config=SimpleNamespace(psql_db="db"),
    )
    monkeypatch.setattr(cli, "reconcile_freeform_publications", state.reconcile)
    monkeypatch.setattr(cli, "release_freeform_decoder", state.release)
    monkeypatch.setattr(cli, "publish_boss_round_decoded_models", state.publish)
    monkeypatch.setattr(cli, "wait_for_background_publications", state.wait)
    return state


def _args(*argv: str):
    return cli.build_parser().parse_args(list(argv))


async def test_default_is_read_only_and_prints_the_report(calls, capsys) -> None:
    assert await cli.run(_args(), calls.config) == 0

    calls.reconcile.assert_awaited_once_with(calls.config, "db", apply=False)
    calls.release.assert_not_awaited()
    calls.publish.assert_not_awaited()
    out = capsys.readouterr().out
    assert "tourn_1" in out and "5Champion" in out and "5Old" in out


async def test_json_output_is_the_report(calls, capsys) -> None:
    assert await cli.run(_args("--json"), calls.config) == 0

    data = json.loads(capsys.readouterr().out)
    assert data["pending_releases"] == [{"tournament_id": "tourn_1", "hotkey": "5Champion"}]
    assert data["blocked_releases"][0]["hotkey"] == "5Old"


async def test_apply_runs_the_reconcile_for_real_and_waits(calls) -> None:
    assert await cli.run(_args("--apply"), calls.config) == 0

    calls.reconcile.assert_awaited_once_with(calls.config, "db", apply=True)
    calls.wait.assert_awaited_once()


async def test_release_goes_through_the_guarded_function(calls) -> None:
    args = _args("--release", "--tournament-id", "tourn_1", "--hotkey", "5Champion")

    assert await cli.run(args, calls.config) == 0

    calls.release.assert_awaited_once_with("tourn_1", "5Champion", calls.config, "db", allow_boss=False)
    calls.reconcile.assert_not_awaited()


async def test_release_passes_allow_boss(calls) -> None:
    await cli.run(_args("--release", "--tournament-id", "t", "--hotkey", "5Burn", "--allow-boss"), calls.config)

    assert calls.release.await_args.kwargs == {"allow_boss": True}


async def test_release_exits_1_when_refused(calls) -> None:
    calls.release.return_value = None

    assert await cli.run(_args("--release", "--tournament-id", "t", "--hotkey", "h"), calls.config) == 1


async def test_publish_decoded_publishes_one_tournament(calls, capsys) -> None:
    assert await cli.run(_args("--publish-decoded", "--tournament-id", "tourn_1"), calls.config) == 0

    calls.publish.assert_awaited_once_with("tourn_1", calls.config, "db")
    assert "org/m-decoded" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["--release", "--tournament-id", "t"],
        ["--release", "--hotkey", "h"],
        ["--publish-decoded"],
        ["--allow-boss"],
        ["--apply", "--release", "--tournament-id", "t", "--hotkey", "h"],
        ["--apply", "--publish-decoded", "--tournament-id", "t"],
    ],
)
def test_invalid_combinations_are_rejected(argv) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(argv)


async def test_main_connects_and_always_closes(monkeypatch, calls) -> None:
    config = MagicMock()
    config.psql_db.connect = AsyncMock()
    config.psql_db.close = AsyncMock()
    monkeypatch.setattr(cli, "load_config", MagicMock(return_value=config))
    monkeypatch.setattr(cli, "run", AsyncMock(return_value=1))

    assert await cli.main([]) == 1

    config.psql_db.connect.assert_awaited_once()
    config.psql_db.close.assert_awaited_once()
