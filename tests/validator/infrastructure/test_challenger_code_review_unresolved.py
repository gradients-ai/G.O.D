from validator.infrastructure.challenger_code_review import _is_access_unresolved
from validator.infrastructure.challenger_code_review import _merge_materialization_into_verdict
from validator.infrastructure.challenger_dependency_review import _parse_requirement_token
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
