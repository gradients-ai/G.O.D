import logging
import os
from urllib.parse import urlparse

import aiohttp


logger = logging.getLogger(__name__)


class S3DownloadError(Exception):
    """Raised when an S3 file download fails."""

    def __init__(self, url: str, status_code: int, message: str = ""):
        self.url = url
        self.status_code = status_code
        super().__init__(message or f"Failed to download file from {url}: HTTP {status_code}")


def resolve_download_path(file_url: str, save_path: str | None = None, tmp_dir: str = "/tmp") -> str:
    """Determine the local file path for a downloaded file.

    Args:
        file_url: The URL of the file to download.
        save_path: Optional save location (directory or exact file path).
        tmp_dir: Fallback temporary directory when save_path is not provided.

    Returns:
        The resolved local file path.

    Raises:
        ValueError: If the URL contains no extractable filename and no explicit save_path file is given.
    """
    parsed_url = urlparse(file_url)
    file_name = os.path.basename(parsed_url.path)

    if not file_name and save_path and not os.path.isdir(save_path):
        return save_path

    if not file_name:
        raise ValueError(
            f"Cannot determine filename from URL '{file_url}'. "
            "Provide an explicit file path via save_path."
        )

    if save_path:
        if os.path.isdir(save_path):
            return os.path.join(save_path, file_name)
        return save_path

    return os.path.join(tmp_dir, file_name)


async def download_s3_file(
    file_url: str,
    save_path: str | None = None,
    tmp_dir: str = "/tmp",
    timeout_seconds: int = 300,
) -> str:
    """Download a file from an S3 URL and save it locally.

    Args:
        file_url: The URL of the file to download.
        save_path: The path where the file should be saved. If a directory is provided,
            the file will be saved with its original name in that directory. If a file path
            is provided, the file will be saved at that exact location. Defaults to None.
        tmp_dir: The temporary directory to use when save_path is not provided.
            Defaults to "/tmp".
        timeout_seconds: Maximum time in seconds to wait for the download.
            Defaults to 300 (5 minutes).

    Returns:
        The local file path where the file was saved.

    Raises:
        S3DownloadError: If the download fails with a non-200 status code.
        ValueError: If the filename cannot be determined from the URL.
        aiohttp.ClientError: If a network-level error occurs.

    Example:
        >>> file_path = await download_s3_file("https://example.com/file.txt", save_path="/data")
        >>> print(file_path)
        /data/file.txt
    """
    local_file_path = resolve_download_path(file_url, save_path, tmp_dir)

    # Ensure parent directory exists
    parent_dir = os.path.dirname(local_file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    logger.info("Downloading %s -> %s", file_url, local_file_path)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(file_url) as response:
            if response.status == 200:
                with open(local_file_path, "wb") as f:
                    f.write(await response.read())
            else:
                raise S3DownloadError(
                    url=file_url,
                    status_code=response.status,
                    message=f"Failed to download file: {response.status}",
                )

    file_size = os.path.getsize(local_file_path)
    logger.info("Downloaded %s (%d bytes)", local_file_path, file_size)

    return local_file_path
