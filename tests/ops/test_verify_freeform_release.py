"""The standalone third-party verifier: scorecard nulls and byte arithmetic, and the decode sandbox sizing."""

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import core.constants.quant as quant_cst
import trainer.constants as trainer_cst
from ops.tools.quant import verify_freeform_release as verifier


GIB = 1024**3


def _inputs(**bytes_fields) -> verifier.VerifyInputs:
    fields = {"artifact_bytes": 100, "decoder_xz_bytes": 20, "total_bytes": 120, "ln_penalty": None, "effective_bytes": 120.0}
    scorecard = verifier.Scorecard.model_validate(
        {
            "bytes": {**fields, **bytes_fields},
            "artifact": {},
            "decoder": {},
            "decoded": {},
        }
    )
    return verifier.VerifyInputs.model_construct(scorecard=scorecard)


def _effective(report: verifier.VerificationReport) -> verifier.CheckResult:
    return next(check for check in report.checks if check.name == "effective_bytes")


def test_null_penalty_is_factor_one() -> None:
    report = verifier.VerificationReport()
    verifier.check_byte_arithmetic(_inputs(), report)

    assert report.ok


def test_null_effective_bytes_from_a_non_finite_row_fails_without_crashing() -> None:
    report = verifier.VerificationReport()
    verifier.check_byte_arithmetic(_inputs(ln_penalty=None, effective_bytes=None), report)

    assert not _effective(report).ok


def test_overflowing_penalty_fails_without_crashing() -> None:
    report = verifier.VerificationReport()
    verifier.check_byte_arithmetic(_inputs(ln_penalty=1e6, effective_bytes=None), report)

    assert not _effective(report).ok


def _docker_runner_argv(
    monkeypatch, tmp_path: Path, runner: verifier.DockerDecodeRunner, decoded_bytes: int, build: bool = False
):
    """The argv of every docker call the runner makes for one invocation (subprocess.run faked)."""
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(verifier.subprocess, "run", fake_run)
    decoder = tmp_path / "decoder"
    decoder.mkdir()
    if build:
        (decoder / verifier.DECODER_BUILD_SCRIPT).write_text("true")
    invocation = verifier.DecodeInvocation(
        decoder_dir=decoder,
        artifact_dir=tmp_path / "artifact",
        output_dir=tmp_path / "output",
        build_dir=tmp_path / "build",
        decoded_bytes=decoded_bytes,
    )
    runner(invocation)
    return calls


def _memory(argv: list[str]) -> str:
    return argv[argv.index("--memory") + 1]


def test_run_limit_is_two_and_a_half_times_the_decoded_size(monkeypatch, tmp_path: Path) -> None:
    (run,) = _docker_runner_argv(monkeypatch, tmp_path, verifier.DockerDecodeRunner(), 8 * GIB)

    assert _memory(run) == str(20 * GIB)


def test_small_model_gets_the_floor_on_both_phases(monkeypatch, tmp_path: Path) -> None:
    build, run = _docker_runner_argv(monkeypatch, tmp_path, verifier.DockerDecodeRunner(), 1 * GIB, build=True)

    assert _memory(build) == "8g"
    assert _memory(run) == str(8 * GIB)


def test_build_phase_stays_at_the_floor_for_a_large_model(monkeypatch, tmp_path: Path) -> None:
    build, run = _docker_runner_argv(monkeypatch, tmp_path, verifier.DockerDecodeRunner(), 8 * GIB, build=True)

    assert _memory(build) == "8g"
    assert _memory(run) == str(20 * GIB)


def test_memory_override_wins_for_the_run_phase(monkeypatch, tmp_path: Path) -> None:
    build, run = _docker_runner_argv(monkeypatch, tmp_path, verifier.DockerDecodeRunner(memory="32g"), 8 * GIB, build=True)

    assert _memory(run) == "32g"
    assert _memory(build) == "8g"


def test_check_decode_passes_the_scorecard_size_to_the_runner(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as tar:
        data = b"print()"
        info = tarfile.TarInfo(f"{verifier.DECODER_DIR_NAME}/{verifier.DECODER_ENTRYPOINT}")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    xz = tmp_path / "decoder.tar.xz"
    xz.write_bytes(buffer.getvalue())
    shard = {"name": "a.safetensors", "bytes": 5 * GIB, "sha256": "a" * 64}
    scorecard = verifier.Scorecard.model_validate(
        {
            "bytes": {},
            "artifact": {},
            "decoder": {},
            "decoded": {"files": [shard, {**shard, "name": "b.safetensors", "bytes": 3 * GIB}]},
        }
    )
    inputs = verifier.VerifyInputs.model_construct(scorecard=scorecard, decoder_xz=xz, artifact_dir=tmp_path)
    seen: list[int] = []

    verifier.check_decode(inputs, verifier.VerificationReport(), lambda invocation: seen.append(invocation.decoded_bytes))

    assert seen == [8 * GIB]


def test_cli_memory_reaches_the_docker_runner(monkeypatch) -> None:
    made: list[verifier.DockerDecodeRunner] = []

    class Recording(verifier.DockerDecodeRunner):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(verifier, "DockerDecodeRunner", Recording)
    monkeypatch.setattr(verifier, "_resolve_inputs", lambda args, work: _inputs())
    monkeypatch.setattr(verifier, "verify_release", lambda inputs, runner=None: verifier.VerificationReport())

    verifier.main(["--scorecard", "x", "--decode", "--memory", "48g"])
    verifier.main(["--scorecard", "x", "--decode"])

    assert [runner.memory for runner in made] == ["48g", None]


def test_local_constants_mirror_the_repo() -> None:
    assert verifier.DECODE_MEMORY_MULTIPLIER == quant_cst.DECODE_MEMORY_MULTIPLIER
    assert verifier.DECODE_MIN_MEMORY_GB == trainer_cst.QUANT_DECODE_MIN_MEMORY_GB
