import os
from urllib.parse import urlparse

import aiohttp


async def download_s3_file(file_url: str, save_path: str = None, tmp_dir: str = "/tmp") -> str:
    """Download a file from an S3 URL and save it locally."""
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
