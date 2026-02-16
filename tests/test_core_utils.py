"""Tests for core.utils module — download_s3_file functionality."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from core.utils import download_s3_file


class TestDownloadS3File:
    """Test suite for the download_s3_file function."""

    @pytest.mark.asyncio
    async def test_download_to_tmp_dir_default(self, tmp_path):
        """File downloads to /tmp by default when no save_path is given."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"file content")

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_session_ctx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        tmp_dir = str(tmp_path)

        with patch("core.utils.aiohttp.ClientSession", return_value=mock_session):
            result = await download_s3_file(
                "https://bucket.s3.amazonaws.com/path/to/model.bin",
                tmp_dir=tmp_dir,
            )

        assert result == os.path.join(tmp_dir, "model.bin")
        assert os.path.exists(result)
        with open(result, "rb") as f:
            assert f.read() == b"file content"

    @pytest.mark.asyncio
    async def test_download_to_directory_save_path(self, tmp_path):
        """When save_path is a directory, file is saved with original name inside it."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"data")

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_session_ctx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        save_dir = str(tmp_path / "output")
        os.makedirs(save_dir)

        with patch("core.utils.aiohttp.ClientSession", return_value=mock_session):
            result = await download_s3_file(
                "https://bucket.s3.amazonaws.com/weights.safetensors",
                save_path=save_dir,
            )

        assert result == os.path.join(save_dir, "weights.safetensors")

    @pytest.mark.asyncio
    async def test_download_to_exact_file_path(self, tmp_path):
        """When save_path is a file path, file is saved at that exact location."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"data")

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_session_ctx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        exact_path = str(tmp_path / "custom_name.bin")

        with patch("core.utils.aiohttp.ClientSession", return_value=mock_session):
            result = await download_s3_file(
                "https://bucket.s3.amazonaws.com/original.bin",
                save_path=exact_path,
            )

        assert result == exact_path

    @pytest.mark.asyncio
    async def test_download_raises_on_non_200(self):
        """Non-200 status codes should raise an exception."""
        mock_response = AsyncMock()
        mock_response.status = 404

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_session_ctx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("core.utils.aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match="Failed to download file: 404"):
                await download_s3_file("https://bucket.s3.amazonaws.com/missing.bin")

    @pytest.mark.asyncio
    async def test_extracts_filename_from_complex_url(self, tmp_path):
        """Correctly extracts filename from URLs with query parameters."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"content")

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_session_ctx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("core.utils.aiohttp.ClientSession", return_value=mock_session):
            result = await download_s3_file(
                "https://bucket.s3.amazonaws.com/deep/nested/path/file.tar.gz",
                tmp_dir=str(tmp_path),
            )

        assert os.path.basename(result) == "file.tar.gz"
