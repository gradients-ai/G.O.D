import asyncio
import ipaddress
import os
import socket
from pathlib import Path
from urllib.parse import urljoin
from urllib.parse import urlsplit

import httpx


MAX_DOWNLOAD_BYTES = 2_147_483_646
MAX_REDIRECTS = 5


async def _validate_public_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Download URL must use http or https and include a host")
    if parsed.username or parsed.password:
        raise ValueError("Download URL credentials are not allowed")
    loop = asyncio.get_running_loop()
    try:
        addresses = [ipaddress.ip_address(parsed.hostname)]
    except ValueError:
        records = await loop.getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM)
        addresses = [ipaddress.ip_address(record[4][0]) for record in records]
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Download URL resolved to a non-public address")


async def download_s3_file(
    file_url: str,
    save_path: str = None,
    tmp_dir: str = "/tmp",
    *,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> str:
    """Download a file from an S3 URL and save it locally."""
    parsed_url = urlsplit(file_url)
    file_name = os.path.basename(parsed_url.path) or "download"
    if save_path:
        if os.path.isdir(save_path):
            local_file_path = os.path.join(save_path, file_name)
        else:
            local_file_path = save_path
    else:
        local_file_path = os.path.join(tmp_dir, file_name)

    current_url = file_url
    Path(local_file_path).parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(follow_redirects=False, timeout=120) as client:
        for _ in range(MAX_REDIRECTS + 1):
            await _validate_public_url(current_url)
            async with client.stream("GET", current_url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise RuntimeError("Download redirect had no Location header")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_bytes:
                    raise ValueError("Download exceeds the configured size limit")
                received = 0
                try:
                    with open(local_file_path, "wb") as output:
                        async for chunk in response.aiter_bytes():
                            received += len(chunk)
                            if received > max_bytes:
                                raise ValueError("Download exceeds the configured size limit")
                            output.write(chunk)
                except Exception:
                    Path(local_file_path).unlink(missing_ok=True)
                    raise
                return local_file_path

    raise ValueError("Download exceeded the redirect limit")
