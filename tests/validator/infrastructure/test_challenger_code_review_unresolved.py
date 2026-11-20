from validator.infrastructure.challenger_code_review import _demote_materialization_noise
from validator.infrastructure.challenger_code_review import _is_access_unresolved
from validator.infrastructure.challenger_code_review import _merge_materialization_into_verdict
from validator.infrastructure.challenger_dependency_review import _parse_requirement_token
from validator.infrastructure.challenger_dependency_review import _trusted_git_repo
from validator.tournament.models import IntegrityVerdict


def test_pypi_lookup_failure_does_not_force_needs_review():
    verdict = IntegrityVerdict(
        flagged=False,
        needs_review=False,
        confidence=0.9,
        reason="Live entrypoint trains only on validator data.",
        evidence=["challenger/train.py:10"],
    )
    merged = _merge_materialization_into_verdict(
        verdict,
        ["https: pypi lookup failed: no files for version 0.0"],
    )
    assert merged.needs_review is False
    assert merged.flagged is False
    assert merged.reason == "Live entrypoint trains only on validator data."
    assert any("non-blocking" in item.lower() for item in merged.evidence)
    assert merged.unresolved_dependencies == [
        "https: pypi lookup failed: no files for version 0.0"
    ]


def test_download_failure_is_access_unresolved():
    assert _is_access_unresolved("foo: download/extract failed: timed out")
    assert _is_access_unresolved("bar: pypi lookup failed: 503")
    assert not _is_access_unresolved("secret-lib (git dependency is not commit-pinned)")
    assert not _is_access_unresolved("evil: sha256 mismatch (expected abc)")


def test_policy_unresolved_still_forces_needs_review():
    verdict = IntegrityVerdict(
        flagged=False,
        needs_review=False,
        confidence=0.8,
        reason="Looks clean from available source.",
    )
    merged = _merge_materialization_into_verdict(
        verdict,
        [
            "secret-lib (git dependency is not commit-pinned)",
            "noise: pypi lookup failed: timeout",
        ],
    )
    assert merged.needs_review is True
    assert "git dependency is not commit-pinned" in merged.reason
    assert "pypi lookup failed" not in merged.reason


def test_http_url_tokens_are_not_parsed_as_pypi_packages():
    assert _parse_requirement_token("https://example.com/weights.bin", "Dockerfile") is None
    assert _parse_requirement_token("http://files.example/data.tgz", "requirements.txt") is None


def test_demote_needs_review_that_only_cites_missing_musubi_materialization():
    verdict = IntegrityVerdict(
        flagged=False,
        needs_review=True,
        confidence=0.85,
        reason=(
            "First-party code is clean, but musubi-tuner and ai-toolkit were NOT "
            "materialized under _dependencies and could not be inspected."
        ),
        evidence=[
            "challenger/train.py:10 training rows only from validator zip",
            "_dependencies absent; PACKAGE_FILE_INVENTORY.json missing",
        ],
    )
    demoted = _demote_materialization_noise(verdict)
    assert demoted.needs_review is False
    assert demoted.flagged is False
    assert any("demoted needs_review" in item.lower() for item in demoted.evidence)


def test_keep_needs_review_when_exploit_signal_is_present():
    verdict = IntegrityVerdict(
        flagged=False,
        needs_review=True,
        confidence=0.7,
        reason="Dependency may inject bundled dataset rows via fingerprint lookup.",
        evidence=["_dependencies/python/secret-lib/loader.py:40 fingerprint select"],
    )
    kept = _demote_materialization_noise(verdict)
    assert kept.needs_review is True


def test_keep_needs_review_for_custom_uninspectable_dependency():
    verdict = IntegrityVerdict(
        flagged=False,
        needs_review=True,
        confidence=0.6,
        reason="Custom secret-lib on the training path could not be inspected.",
        evidence=["challenger/train.py:20 imports secret_lib.load_rows"],
    )
    kept = _demote_materialization_noise(verdict)
    assert kept.needs_review is True


def test_trusted_git_repos_are_recognized():
    assert _trusted_git_repo("git+https://github.com/kohya-ss/musubi-tuner.git@30c658c4")
    assert _trusted_git_repo("https://github.com/ostris/ai-toolkit.git")
    assert not _trusted_git_repo("git+https://github.com/evil/backdoor.git@abc123")
