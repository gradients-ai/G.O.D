import os
from urllib.parse import quote
from urllib.parse import urlparse
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import aiohttp


async def download_s3_file(file_url: str, save_path: str = None, tmp_dir: str = "/tmp") -> str:
    """Download a file from an S3 URL and save it locally.

    Args:
        file_url (str): The URL of the file to download.
        save_path (str, optional): The path where the file should be saved. If a directory is provided,
            the file will be saved with its original name in that directory. If a file path is provided,
            the file will be saved at that exact location. Defaults to None.
        tmp_dir (str, optional): The temporary directory to use when save_path is not provided.
            Defaults to "/tmp".

    Returns:
        str: The local file path where the file was saved.

    Raises:
        Exception: If the download fails with a non-200 status code.

    Example:
        >>> file_path = await download_s3_file("https://example.com/file.txt", save_path="/data")
        >>> print(file_path)
        /data/file.txt
    """
    parsed_url = urlparse(file_url)
    file_name = os.path.basename(parsed_url.path)
    if save_path:
        if os.path.isdir(save_path):
            local_file_path = os.path.join(save_path, file_name)
        else:
            local_file_path = save_path
    else:
        local_file_path = os.path.join(tmp_dir, file_name)

    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as response:
            if response.status == 200:
                with open(local_file_path, "wb") as f:
                    f.write(await response.read())
            else:
                raise Exception(f"Failed to download file: {response.status}")

    return local_file_path


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
