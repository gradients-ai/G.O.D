from urllib.parse import quote
from urllib.parse import urlsplit
from urllib.parse import urlunsplit


def build_authenticated_git_url(repo_url: str, github_token: str | None) -> str:
    """Return a clone URL with embedded token without mutating the stored repo URL."""
    if not github_token:
        return repo_url

    parsed = urlsplit(repo_url)
    if not parsed.scheme or not parsed.netloc:
        return repo_url

    token = quote(github_token, safe="")
    netloc = f"{token}@{parsed.netloc.split('@', 1)[-1]}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def sanitize_git_text(text: str, *tokens: str | None) -> str:
    """
    Remove credentials from git-related strings before logging.
    It redacts provided tokens and strips URL credentials like https://user:pass@host.
    """
    if not text:
        return text

    sanitized = text
    for token in tokens:
        if token:
            sanitized = sanitized.replace(token, "***")
            sanitized = sanitized.replace(quote(token, safe=""), "***")

    if "://" in sanitized and "@" in sanitized:
        return _strip_url_credentials(sanitized)
    return sanitized


def _strip_url_credentials(text: str) -> str:
    parts = text.split()
    sanitized_parts: list[str] = []
    for part in parts:
        if "://" in part and "@" in part:
            scheme, rest = part.split("://", 1)
            if "@" in rest:
                rest = rest.split("@", 1)[1]
                sanitized_parts.append(f"{scheme}://{rest}")
                continue
        sanitized_parts.append(part)
    return " ".join(sanitized_parts)
