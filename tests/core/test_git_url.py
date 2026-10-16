from core.git import build_authenticated_git_url
from core.git import sanitize_git_text


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
