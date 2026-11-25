import pytest
from pydantic import ValidationError

from core.git import build_authenticated_git_url
from core.git import git_subprocess_env
from core.git import sanitize_git_text
from core.git import validate_github_repo_url
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainingRepoResponse


def test_token_is_sent_as_password_with_x_access_token_user():
    url = build_authenticated_git_url("https://github.com/o/r", "github_pat_ABC")
    assert url == "https://x-access-token:github_pat_ABC@github.com/o/r"


def test_existing_credentials_are_replaced():
    url = build_authenticated_git_url("https://old@github.com/o/r.git", "tok")
    assert url == "https://x-access-token:tok@github.com/o/r.git"


def test_token_is_url_quoted():
    assert build_authenticated_git_url("https://github.com/o/r", "a/b") == "https://x-access-token:a%2Fb@github.com/o/r"


def test_no_token_or_bad_url_unchanged():
    assert build_authenticated_git_url("https://github.com/o/r", None) == "https://github.com/o/r"
    assert build_authenticated_git_url("github.com/o/r", "tok") == "github.com/o/r"


def test_sanitize_strips_credentials():
    url = build_authenticated_git_url("https://github.com/o/r", "github_pat_ABC")
    out = sanitize_git_text(f"fatal: could not read {url}", "github_pat_ABC")
    assert "github_pat_ABC" not in out
    assert "x-access-token" not in out


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/owner/repo",
        "https://github.com/owner/repo.git",
        "https://github.com/owner/repo/",
        "https://github.com/Own-er_1/re.po_2.git/",
    ],
)
def test_github_repo_url_accepted(url):
    assert validate_github_repo_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/owner/repo",
        "https://gitlab.com/owner/repo",
        "https://github.com.evil.com/owner/repo",
        "https://user:pass@github.com/owner/repo",
        "https://github.com/owner",
        "https://github.com/owner/repo/tree/main",
        "https://github.com/../repo",
        "https://github.com/owner/..",
        "ext::sh -c touch% /tmp/x",
        "--upload-pack=touch /tmp/x",
        "file:///etc",
        "/tmp/repo",
        "https://github.com/owner/repo\n",
        "https://github.com/owner/repo?x=1",
    ],
)
def test_github_repo_url_rejected(url):
    with pytest.raises(ValueError):
        validate_github_repo_url(url)
    with pytest.raises(ValidationError):
        TrainingRepoResponse(github_repo=url, commit_hash="a" * 40)


def test_trainer_request_rejects_a_non_github_repo():
    payload = {
        "training_data": {
            "model": "org/model", "task_id": "t", "hours_to_complete": 1, "expected_repo_name": "r",
            "dataset": "s3://b/d", "dataset_type": {"field_instruction": "i"}, "file_format": "s3",
        },
        "gpu_ids": [0],
        "hotkey": "hk",
    }
    TrainerProxyRequest.model_validate({**payload, "github_repo": "https://github.com/o/r"})
    with pytest.raises(ValidationError):
        TrainerProxyRequest.model_validate({**payload, "github_repo": "ext::sh -c id"})


def test_git_subprocess_env_allows_only_https():
    assert git_subprocess_env({"PATH": "/bin"}) == {"PATH": "/bin", "GIT_ALLOW_PROTOCOL": "https"}
