"""Text registration build lint: immutable dependencies, rule by rule, plus the 1 MB review listing.

The fixture repos under fixtures/registration_lint are a build that passes everything (clean), one violation of
each rule (dirty), and a build with a tagged base image and unhashed requirements (tagged_unhashed), which fails on unhashed pip
and a fiber git tag and nothing else.
"""

from pathlib import Path

import pytest

from core.build_context import find_large_files
from core.constants.docker import DEFAULT_TEXT_DOCKERFILE_PATH
from core.constants.docker import LEGACY_TEXT_DOCKERFILE_PATH
from core.constants.quant import QUANT_LARGE_FILE_REVIEW_BYTES
from validator.tournament import registration_lint
from validator.tournament.registration_lint import LintRule
from validator.tournament.registration_lint import RegistrationLintFailed
from validator.tournament.registration_lint import RegistrationLintReport
from validator.tournament.registration_lint import check_registration_lint
from validator.tournament.registration_lint import lint_repo


FIXTURES = Path(__file__).parent / "fixtures" / "registration_lint"
SHA = "0123456789abcdef0123456789abcdef01234567"
HASH = "a" * 64


def _repo(tmp_path: Path, dockerfile: str, files: dict[str, str] | None = None) -> Path:
    (tmp_path / LEGACY_TEXT_DOCKERFILE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / LEGACY_TEXT_DOCKERFILE_PATH).write_text("FROM python:3.11\nWORKDIR /app\n" + dockerfile)
    for name, content in (files or {}).items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(content)
    return tmp_path


def _rules(report: RegistrationLintReport) -> set[LintRule]:
    return {violation.rule for violation in report.violations}


class TestFixtureRepos:
    def test_clean_repo_passes_and_lints_its_build_script(self):
        report = lint_repo(FIXTURES / "clean")

        assert report.passed, report.rejection_reason()
        assert report.dockerfile == LEGACY_TEXT_DOCKERFILE_PATH
        assert report.build_scripts == ["scripts/install_extras.sh"]

    def test_dirty_repo_fails_every_rule_with_located_reasons(self):
        report = lint_repo(FIXTURES / "dirty")

        assert _rules(report) == {
            LintRule.UNPINNED_BASE_IMAGE,
            LintRule.REMOTE_FETCH,
            LintRule.UNHASHED_PIP,
            LintRule.UNPINNED_GIT,
        }
        described = [violation.describe() for violation in report.violations]
        assert any("scripts/requirements.txt" in line and "torch, transformers" in line for line in described)
        assert any("missing_requirements.txt is not in the build context" in line for line in described)
        assert any(line.startswith("scripts/setup.sh:2 [remote_fetch]") for line in described)
        assert any(line.startswith("scripts/setup.sh:3 [unhashed_pip] uv sync") for line in described)
        assert "build-lint violation(s)" in report.rejection_reason()

    def test_tagged_unhashed_repo_fails_only_pip_hashes_and_git_tag(self):
        report = lint_repo(FIXTURES / "tagged_unhashed")

        assert _rules(report) == {LintRule.UNHASHED_PIP, LintRule.UNPINNED_GIT}
        # The hashed kernel requirements file run fail-soft passes; `apt-get install wget` is not a fetch.
        assert not any("kernel_requirements" in violation.file for violation in report.violations)


class TestHashPinnedPip:
    @pytest.mark.parametrize(
        "run",
        [
            "RUN pip install tenacity==9.1.2",
            "RUN pip install -U pip",
            "RUN python -m pip install numpy",
            "RUN uv pip install --system numpy==2.0",
            "RUN /venv/bin/pip3 install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0",
            "RUN pip install https://example.com/x-1.0-py3-none-any.whl",
            "RUN uv sync",
            "RUN uv add numpy",
        ],
    )
    def test_unhashed_installs_fail(self, tmp_path, run):
        assert _rules(lint_repo(_repo(tmp_path, run + "\n"))) == {LintRule.UNHASHED_PIP}

    @pytest.mark.parametrize(
        "run",
        [
            f"RUN pip install https://example.com/x-1.0-py3-none-any.whl#sha256={HASH}",
            "RUN pip install -e . && pip install /app/dist/pkg-1.0-py3-none-any.whl",
            "RUN uv sync --locked",
            "RUN uv pip install --python /venv/bin/python --torch-backend cu129 --index-strategy unsafe-best-match -e .",
            "RUN pip uninstall -y mamba-ssm",
        ],
    )
    def test_hashed_local_and_locked_installs_pass(self, tmp_path, run):
        report = lint_repo(_repo(tmp_path, run + "\n"))
        assert report.passed, report.rejection_reason()

    def test_requirements_file_needs_a_hash_on_every_entry(self, tmp_path):
        files = {"req.txt": f"torch==2.8.0 --hash=sha256:{HASH}\nnumpy==2.0\n"}
        report = lint_repo(_repo(tmp_path, "COPY req.txt .\nRUN pip install -r req.txt\n", files))

        assert [violation.describe() for violation in report.violations] == [
            "req.txt:2 [unhashed_pip] 1 requirement(s) without --hash (e.g. numpy)"
        ]

    def test_fully_hashed_requirements_with_options_and_nested_file_pass(self, tmp_path):
        files = {
            "reqs/main.txt": f"--index-url https://pypi.org/simple\n-r extra.txt\ntorch==2.8.0 \\\n    --hash=sha256:{HASH}\n",
            "reqs/extra.txt": f"# comment\nnumpy==2.0 --hash=sha256:{HASH}  # trailing\n",
        }
        report = lint_repo(_repo(tmp_path, "COPY reqs /deps\nRUN pip install --require-hashes -r /deps/main.txt\n", files))
        assert report.passed, report.rejection_reason()

    def test_nested_unhashed_requirements_fail(self, tmp_path):
        files = {"reqs/main.txt": "-r extra.txt\n", "reqs/extra.txt": "numpy==2.0\n"}
        report = lint_repo(_repo(tmp_path, "COPY reqs /deps\nRUN pip install -r /deps/main.txt\n", files))
        assert [violation.file for violation in report.violations] == ["reqs/extra.txt"]

    def test_requirements_copied_after_the_run_cannot_be_checked(self, tmp_path):
        files = {"req.txt": f"torch==2.8.0 --hash=sha256:{HASH}\n"}
        report = lint_repo(_repo(tmp_path, "RUN pip install -r /app/req.txt\nCOPY req.txt /app/req.txt\n", files))
        assert "is not in the build context" in report.rejection_reason()

    def test_uv_pip_sync_reads_positional_requirement_files(self, tmp_path):
        files = {"req.txt": "numpy==2.0\n"}
        report = lint_repo(_repo(tmp_path, "COPY req.txt .\nRUN uv pip sync req.txt\n", files))
        assert [violation.file for violation in report.violations] == ["req.txt"]


class TestShaPinnedGit:
    @pytest.mark.parametrize("ref", ["2.4.0", "main", SHA[:12], ""])
    def test_git_requirement_without_full_sha_fails(self, tmp_path, ref):
        at = f"@{ref}" if ref else ""
        report = lint_repo(_repo(tmp_path, f'RUN pip install "fiber @ git+https://github.com/rayonlabs/fiber.git{at}"\n'))
        assert _rules(report) == {LintRule.UNPINNED_GIT}

    def test_git_requirement_with_sha_passes_including_via_arg(self, tmp_path):
        dockerfile = f'ARG REF={SHA}\nRUN pip install "git+https://github.com/x/y.git@${{REF}}#egg=y"\n'
        report = lint_repo(_repo(tmp_path, dockerfile))
        assert report.passed, report.rejection_reason()

    def test_git_entry_in_requirements_file(self, tmp_path):
        files = {"req.txt": "y @ git+https://github.com/x/y.git@v1\n"}
        report = lint_repo(_repo(tmp_path, "COPY req.txt .\nRUN pip install -r req.txt\n", files))
        assert _rules(report) == {LintRule.UNPINNED_GIT}

    def test_git_clone_needs_a_sha_checkout_in_the_same_run(self, tmp_path):
        unpinned = lint_repo(_repo(tmp_path / "a", "RUN git clone https://github.com/x/y.git /y\n"))
        pinned = lint_repo(
            _repo(tmp_path / "b", f"RUN git clone https://github.com/x/y.git /y && cd /y && git checkout {SHA}\n")
        )
        separate_step = lint_repo(
            _repo(tmp_path / "c", f"RUN git clone https://github.com/x/y.git /y\nRUN git -C /y checkout {SHA}\n")
        )

        assert _rules(unpinned) == {LintRule.UNPINNED_GIT}
        assert pinned.passed
        assert _rules(separate_step) == {LintRule.UNPINNED_GIT}


class TestRemoteFetch:
    @pytest.mark.parametrize(
        "run",
        [
            "RUN curl -fsSL https://miner.example/x.bin -o /x.bin",
            "RUN wget -q https://miner.example/x.bin",
            "RUN cd /tmp && FOO=1 wget https://huggingface.co/miner/repo/resolve/main/w.bin",
            "ADD https://miner.example/x.tar.gz /x/",
        ],
    )
    def test_unchecked_fetch_fails(self, tmp_path, run):
        assert _rules(lint_repo(_repo(tmp_path, run + "\n"))) == {LintRule.REMOTE_FETCH}

    @pytest.mark.parametrize(
        "run",
        [
            "RUN apt-get update && apt-get install -y curl wget",
            f'RUN curl -o /x.bin https://miner.example/x.bin && echo "{HASH}  /x.bin" | sha256sum -c -',
            "RUN curl -LsSf https://astral.sh/uv/0.4.0/install.sh | sh",
            f"ADD --checksum=sha256:{HASH} https://miner.example/x.tar.gz /x/",
            f"ADD https://github.com/x/y.git#{SHA} /y",
        ],
    )
    def test_checked_trusted_or_non_fetch_passes(self, tmp_path, run):
        report = lint_repo(_repo(tmp_path, run + "\n"))
        assert report.passed, report.rejection_reason()

    def test_build_script_is_linted_but_runtime_entrypoint_is_not(self, tmp_path):
        files = {
            "scripts/build.sh": "#!/bin/sh\ncurl -o /w.bin https://miner.example/w.bin\n",
            "scripts/entrypoint.sh": "#!/bin/sh\ncurl -o /w.bin https://miner.example/w.bin\npip install foo\n",
        }
        dockerfile = (
            "COPY scripts /app/scripts\n"
            "RUN bash scripts/build.sh && chmod +x /app/scripts/entrypoint.sh\n"
            'ENTRYPOINT ["./scripts/entrypoint.sh"]\n'
        )
        report = lint_repo(_repo(tmp_path, dockerfile, files))

        assert report.build_scripts == ["scripts/build.sh"]
        assert [violation.describe() for violation in report.violations] == [
            "scripts/build.sh:2 [remote_fetch] curl https://miner.example/w.bin: remote fetch without a checksum "
            "from a host that is not on the trusted list"
        ]

    def test_trusted_prefixes_are_a_constant(self, tmp_path, monkeypatch):
        monkeypatch.setattr(registration_lint.t_cst, "QUANT_TRUSTED_FETCH_URL_PREFIXES", ("https://miner.example/",))
        assert lint_repo(_repo(tmp_path, "RUN wget https://miner.example/x\n")).passed


class TestBaseImage:
    @pytest.mark.parametrize(
        "dockerfile",
        [
            "FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04\n",
            "FROM python:3.11-slim\n",
            f"FROM ghcr.io/miner/base@sha256:{HASH}\n",
            "FROM --platform=linux/amd64 nvidia/cuda:12.8.1-devel-ubuntu22.04 AS build\nFROM build\n",
            "FROM python:3.11 AS tools\nFROM python:3.11\nCOPY --from=tools /a /a\n",
        ],
    )
    def test_trusted_or_digest_pinned_bases_pass(self, tmp_path, dockerfile):
        (tmp_path / LEGACY_TEXT_DOCKERFILE_PATH).parent.mkdir(parents=True)
        (tmp_path / LEGACY_TEXT_DOCKERFILE_PATH).write_text(dockerfile)
        report = lint_repo(tmp_path)
        assert report.passed, report.rejection_reason()

    @pytest.mark.parametrize(
        "dockerfile",
        ["FROM miner/base:latest\n", "FROM python:3.11\nCOPY --from=miner/assets:v1 /w /w\n"],
    )
    def test_miner_owned_tags_fail(self, tmp_path, dockerfile):
        (tmp_path / LEGACY_TEXT_DOCKERFILE_PATH).parent.mkdir(parents=True)
        (tmp_path / LEGACY_TEXT_DOCKERFILE_PATH).write_text(dockerfile)
        assert _rules(lint_repo(tmp_path)) == {LintRule.UNPINNED_BASE_IMAGE}


class TestDockerfileSelection:
    def test_no_dockerfile_fails(self, tmp_path):
        assert _rules(lint_repo(tmp_path)) == {LintRule.NO_DOCKERFILE}

    def test_default_path_wins_over_legacy_like_the_trainer(self, tmp_path):
        _repo(tmp_path, "RUN pip install numpy\n")
        (tmp_path / DEFAULT_TEXT_DOCKERFILE_PATH).parent.mkdir(parents=True)
        (tmp_path / DEFAULT_TEXT_DOCKERFILE_PATH).write_text("FROM python:3.11\n")

        report = lint_repo(tmp_path)

        assert report.dockerfile == DEFAULT_TEXT_DOCKERFILE_PATH
        assert report.passed

    def test_run_heredoc_body_is_linted(self, tmp_path):
        report = lint_repo(_repo(tmp_path, "RUN <<EOF\nset -e\npip install numpy\nEOF\nRUN echo done\n"))
        assert _rules(report) == {LintRule.UNHASHED_PIP}


class TestLargeFiles:
    def test_files_over_the_review_size_are_listed_but_never_fail(self, tmp_path):
        repo = _repo(tmp_path, "")
        (repo / "payload.npy").write_bytes(b"\0" * (QUANT_LARGE_FILE_REVIEW_BYTES + 1))
        (repo / "small.bin").write_bytes(b"\0" * QUANT_LARGE_FILE_REVIEW_BYTES)
        (repo / ".git").mkdir()
        (repo / ".git" / "pack.bin").write_bytes(b"\0" * (QUANT_LARGE_FILE_REVIEW_BYTES + 1))
        (repo / "link.npy").symlink_to(repo / "payload.npy")

        report = lint_repo(repo)

        assert report.passed
        assert [(large.path, large.size_bytes) for large in report.large_files] == [
            ("payload.npy", QUANT_LARGE_FILE_REVIEW_BYTES + 1)
        ]

    def test_find_large_files_orders_largest_first(self, tmp_path):
        (tmp_path / "a").write_bytes(b"x" * 20)
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b").write_bytes(b"x" * 30)

        assert [large.path for large in find_large_files(tmp_path, 10)] == ["sub/b", "a"]


class TestCheckRegistrationLint:
    def test_enforced_failure_raises_with_the_report(self, monkeypatch):
        monkeypatch.setattr(registration_lint.t_cst, "QUANT_REGISTRATION_LINT_ENFORCE", True)

        with pytest.raises(RegistrationLintFailed) as raised:
            check_registration_lint(FIXTURES / "dirty")

        assert not raised.value.report.passed
        assert "unpinned_base_image" in str(raised.value)

    def test_warn_only_logs_and_returns(self, monkeypatch):
        monkeypatch.setattr(registration_lint.t_cst, "QUANT_REGISTRATION_LINT_ENFORCE", False)
        warnings: list[str] = []
        monkeypatch.setattr(registration_lint.logger, "warning", warnings.append)

        report = check_registration_lint(FIXTURES / "tagged_unhashed")

        assert not report.passed
        assert len(warnings) == 1 and warnings[0].startswith("Build lint (warn-only, entrant kept)")

    def test_large_files_are_logged_for_review(self, tmp_path, monkeypatch):
        repo = _repo(tmp_path, "")
        (repo / "delta.bin").write_bytes(b"\0" * (QUANT_LARGE_FILE_REVIEW_BYTES + 1))
        warnings: list[str] = []
        monkeypatch.setattr(registration_lint.logger, "warning", warnings.append)

        assert check_registration_lint(repo).passed
        assert len(warnings) == 1 and "delta.bin" in warnings[0] and "manual review" in warnings[0]
