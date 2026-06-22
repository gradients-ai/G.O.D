import asyncio
import datetime
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from validator.utils.logging import get_logger


logger = get_logger(__name__)


# Retry decorator for MinIO operations
retry_minio_operation = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((S3Error, ConnectionError, TimeoutError)),
    reraise=True,
)


@dataclass
class TransferProgress:
    """Track progress of file transfers."""
    file_path: str
    object_name: str
    bytes_transferred: int = 0
    total_bytes: int = 0
    start_time: float = 0.0
    transfer_rate: float = 0.0  # bytes per second
    completed: bool = False
    error: str | None = None

    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()

    def update(self, bytes_transferred: int, total_bytes: int | None = None):
        """Update transfer progress."""
        self.bytes_transferred = bytes_transferred
        if total_bytes:
            self.total_bytes = total_bytes
        
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.transfer_rate = bytes_transferred / elapsed

    def get_percentage(self) -> float:
        """Get transfer percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.bytes_transferred / self.total_bytes) * 100.0


@dataclass
class FileMetadata:
    """Metadata for stored files."""
    bucket_name: str
    object_name: str
    file_size: int
    content_type: str | None = None
    etag: str | None = None
    last_modified: datetime.datetime | None = None
    tags: dict[str, str] | None = None
    version_id: str | None = None


class AsyncMinioClient:
    """
    Enhanced async MinIO client with caching, retry logic, batch operations,
    integrity checks, and monitoring capabilities.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        region: str,
        secure: bool = True,
        redis_client: Any = None,
        cache_ttl: int = 3600,  # 1 hour default cache TTL
        max_retries: int = 3,
        enable_integrity_check: bool = True,
    ):
        self.endpoint = endpoint
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
        )
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.enable_integrity_check = enable_integrity_check
        self._transfer_progress: dict[str, TransferProgress] = {}
        self._stats = {
            "uploads": 0,
            "downloads": 0,
            "deletes": 0,
            "total_bytes_uploaded": 0,
            "total_bytes_downloaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def _get_from_cache(self, key: str) -> Any | None:
        """Get value from Redis cache if available."""
        if not self.redis_client:
            return None
        try:
            cached = await self.redis_client.get(key)
            if cached:
                self._stats["cache_hits"] += 1
                return json.loads(cached)
            self._stats["cache_misses"] += 1
            return None
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None

    async def _set_cache(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in Redis cache."""
        if not self.redis_client:
            return
        try:
            ttl = ttl or self.cache_ttl
            await self.redis_client.set(key, json.dumps(value), ex=ttl)
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")

    def _calculate_file_hash(self, file_path: str, algorithm: str = "sha256") -> str:
        """Calculate file hash for integrity checking."""
        hash_obj = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    @retry_minio_operation
    async def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        verify_integrity: bool | None = None,
    ) -> bool:
        """
        Upload a file to MinIO with enhanced features.
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in the bucket
            file_path: Local file path to upload
            content_type: MIME type of the file
            metadata: Additional metadata tags
            verify_integrity: Whether to verify file integrity (defaults to self.enable_integrity_check)
        
        Returns:
            True if upload successful, False otherwise
        """
        verify_integrity = verify_integrity if verify_integrity is not None else self.enable_integrity_check
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False

        file_size = os.path.getsize(file_path)
        progress = TransferProgress(file_path=file_path, object_name=object_name, total_bytes=file_size)
        self._transfer_progress[object_name] = progress

        try:
            # Calculate hash before upload if integrity check enabled
            file_hash = None
            if verify_integrity:
                file_hash = self._calculate_file_hash(file_path)
                logger.debug(f"Calculated hash for {file_path}: {file_hash[:16]}...")

            func = self.client.fput_object
            args = (bucket_name, object_name, file_path)
            kwargs = {}
            if content_type:
                kwargs["content_type"] = content_type
            if metadata:
                kwargs["metadata"] = metadata

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: func(*args, **kwargs) if kwargs else func(*args)
            )

            progress.update(file_size, file_size)
            progress.completed = True
            
            # Store metadata in cache
            metadata_obj = FileMetadata(
                bucket_name=bucket_name,
                object_name=object_name,
                file_size=file_size,
                content_type=content_type,
                etag=file_hash,
            )
            await self._set_cache(f"metadata:{bucket_name}:{object_name}", metadata_obj.__dict__)

            self._stats["uploads"] += 1
            self._stats["total_bytes_uploaded"] += file_size
            
            logger.info(
                f"Successfully uploaded {object_name} ({file_size} bytes) "
                f"to bucket {bucket_name} at {progress.transfer_rate / 1024 / 1024:.2f} MB/s"
            )
            
            return True

        except Exception as e:
            progress.error = str(e)
            logger.error(f"Failed to upload {file_path} to {bucket_name}/{object_name}: {e}")
            return False
        finally:
            # Clean up progress tracking after 5 minutes
            asyncio.create_task(self._cleanup_progress(object_name, delay=300))

    async def _cleanup_progress(self, object_name: str, delay: int = 300):
        """Clean up progress tracking after delay."""
        await asyncio.sleep(delay)
        self._transfer_progress.pop(object_name, None)

    @retry_minio_operation
    async def download_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        verify_integrity: bool | None = None,
    ) -> bool:
        """
        Download a file from MinIO with integrity verification.
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object to download
            file_path: Local file path to save the file
            verify_integrity: Whether to verify file integrity after download
        
        Returns:
            True if download successful, False otherwise
        """
        verify_integrity = verify_integrity if verify_integrity is not None else self.enable_integrity_check
        
        try:
            # Get file stats first to know size
            stats = await self.get_stats(bucket_name, object_name)
            file_size = stats.size if hasattr(stats, "size") else 0
            
            progress = TransferProgress(
                file_path=file_path,
                object_name=object_name,
                total_bytes=file_size
            )
            self._transfer_progress[object_name] = progress

            func = self.client.fget_object
            args = (bucket_name, object_name, file_path)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, func, *args)

            # Verify integrity if enabled
            if verify_integrity:
                # Get ETag from stats
                etag = stats.etag if hasattr(stats, "etag") else None
                if etag:
                    downloaded_hash = self._calculate_file_hash(file_path)
                    # MinIO ETag might be quoted or have extra characters
                    etag_clean = etag.strip('"')
                    if downloaded_hash != etag_clean:
                        logger.warning(
                            f"Integrity check failed for {object_name}: "
                            f"expected {etag_clean[:16]}..., got {downloaded_hash[:16]}..."
                        )
                        return False
                    logger.debug(f"Integrity check passed for {object_name}")

            progress.update(file_size, file_size)
            progress.completed = True

            self._stats["downloads"] += 1
            self._stats["total_bytes_downloaded"] += file_size

            logger.info(
                f"Successfully downloaded {object_name} ({file_size} bytes) "
                f"from bucket {bucket_name} at {progress.transfer_rate / 1024 / 1024:.2f} MB/s"
            )
            
            return True

        except Exception as e:
            logger.error(f"Failed to download {bucket_name}/{object_name} to {file_path}: {e}")
            if object_name in self._transfer_progress:
                self._transfer_progress[object_name].error = str(e)
            return False
        finally:
            asyncio.create_task(self._cleanup_progress(object_name, delay=300))

    @retry_minio_operation
    async def delete_file(self, bucket_name: str, object_name: str) -> bool:
        """Delete a file from MinIO."""
        try:
            func = self.client.remove_object
            args = (bucket_name, object_name)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, func, *args)
            
            # Remove from cache
            await self._set_cache(f"metadata:{bucket_name}:{object_name}", None, ttl=1)
            await self._set_cache(f"presigned:{bucket_name}:{object_name}", None, ttl=1)
            
            self._stats["deletes"] += 1
            logger.info(f"Deleted {object_name} from bucket {bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete {bucket_name}/{object_name}: {e}")
            return False

    async def list_objects(
        self,
        bucket_name: str,
        prefix: str | None = None,
        recursive: bool = True,
        include_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        """
        List objects in a bucket with optional metadata.
        
        Returns:
            List of object dictionaries with name, size, last_modified, etc.
        """
        try:
            func = self.client.list_objects
            args = (bucket_name, prefix, recursive)
            
            loop = asyncio.get_event_loop()
            objects = await loop.run_in_executor(self.executor, func, *args)
            
            result = []
            for obj in objects:
                obj_dict = {
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                    "etag": obj.etag,
                }
                if include_metadata:
                    try:
                        stats = await self.get_stats(bucket_name, obj.object_name)
                        obj_dict["content_type"] = getattr(stats, "content_type", None)
                        obj_dict["metadata"] = getattr(stats, "metadata", None)
                    except Exception:
                        pass
                result.append(obj_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list objects in {bucket_name}: {e}")
            return []

    @retry_minio_operation
    async def get_stats(self, bucket_name: str, object_name: str) -> Any:
        """Get stats for an object in MinIO storage."""
        cache_key = f"stats:{bucket_name}:{object_name}"
        cached = await self._get_from_cache(cache_key)
        if cached:
            # Return a simple object-like structure
            class Stats:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
            return Stats(cached)
        
        try:
            func = self.client.stat_object
            args = (bucket_name, object_name)
            
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(self.executor, func, *args)
            
            # Cache stats
            stats_dict = {
                "size": stats.size,
                "etag": stats.etag,
                "content_type": stats.content_type,
                "last_modified": stats.last_modified.isoformat() if stats.last_modified else None,
                "metadata": dict(stats.metadata) if stats.metadata else {},
            }
            await self._set_cache(cache_key, stats_dict, ttl=300)  # Cache for 5 minutes
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for {bucket_name}/{object_name}: {e}")
            raise

    async def ensure_bucket_exists(self, bucket_name: str) -> bool:
        """Ensure bucket exists, create if it doesn't."""
        try:
            func = self.client.bucket_exists
            args = (bucket_name,)
            
            loop = asyncio.get_event_loop()
            exists = await loop.run_in_executor(self.executor, func, *args)
            
            if not exists:
                make_bucket_func = self.client.make_bucket
                await loop.run_in_executor(self.executor, make_bucket_func, bucket_name)
                logger.info(f"Created bucket: {bucket_name}")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure bucket exists {bucket_name}: {e}")
            return False

    async def get_presigned_url(
        self,
        bucket_name: str,
        object_name: str,
        expires: int = 604800,
        use_cache: bool = True,
    ) -> str | None:
        """
        Get presigned URL with caching support.
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            expires: Expiration time in seconds (default 7 days)
            use_cache: Whether to use cached URL if available
        
        Returns:
            Presigned URL string or None if failed
        """
        cache_key = f"presigned:{bucket_name}:{object_name}:{expires}"
        
        if use_cache:
            cached_url = await self._get_from_cache(cache_key)
            if cached_url:
                logger.debug(f"Returning cached presigned URL for {object_name}")
                return cached_url
        
        try:
            expires_duration = datetime.timedelta(seconds=expires)
            func = self.client.presigned_get_object
            args = (bucket_name, object_name, expires_duration)
            
            loop = asyncio.get_event_loop()
            url = await loop.run_in_executor(self.executor, func, *args)
            
            # Cache URL with TTL slightly less than expiration
            cache_ttl = min(expires - 60, self.cache_ttl)  # Cache for expires - 1 minute or default TTL
            await self._set_cache(cache_key, url, ttl=cache_ttl)
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {bucket_name}/{object_name}: {e}")
            return None

    def get_public_url(self, bucket_name: str, object_name: str) -> str:
        """Get public URL for an object."""
        protocol = "https" if self.client._base_url.scheme == "https" else "http"
        return f"{protocol}://{self.endpoint}/{bucket_name}/{object_name}"

    async def get_new_presigned_url(self, presigned_url: str) -> str | None:
        """Generate a new presigned URL from an existing one."""
        try:
            bucket_name, object_name = self.parse_s3_url(presigned_url)
            new_url = await self.get_presigned_url(bucket_name, object_name, use_cache=False)
            
            if new_url:
                logger.info(f"Generated new presigned URL for {object_name} in bucket {bucket_name}")
            
            return new_url
            
        except Exception as e:
            logger.warning(f"Failed to generate new presigned URL: {e}")
            return None

    def parse_s3_url(self, url: str) -> tuple[str, str]:
        """Extract bucket name and object name from S3 URL."""
        parsed_url = urlparse(url)
        bucket_name = parsed_url.hostname.split(".")[0]
        object_name = parsed_url.path.lstrip("/").split("?")[0]
        return bucket_name, object_name

    # Batch operations
    async def batch_upload(
        self,
        bucket_name: str,
        files: list[tuple[str, str]],  # List of (file_path, object_name) tuples
        max_concurrent: int = 5,
    ) -> dict[str, bool]:
        """
        Upload multiple files concurrently.
        
        Args:
            bucket_name: Name of the bucket
            files: List of (file_path, object_name) tuples
            max_concurrent: Maximum concurrent uploads
        
        Returns:
            Dictionary mapping object_name to success status
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}

        async def upload_with_semaphore(file_path: str, object_name: str):
            async with semaphore:
                success = await self.upload_file(bucket_name, object_name, file_path)
                results[object_name] = success

        tasks = [upload_with_semaphore(fp, on) for fp, on in files]
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def batch_download(
        self,
        bucket_name: str,
        files: list[tuple[str, str]],  # List of (object_name, file_path) tuples
        max_concurrent: int = 5,
    ) -> dict[str, bool]:
        """
        Download multiple files concurrently.
        
        Args:
            bucket_name: Name of the bucket
            files: List of (object_name, file_path) tuples
            max_concurrent: Maximum concurrent downloads
        
        Returns:
            Dictionary mapping object_name to success status
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}

        async def download_with_semaphore(object_name: str, file_path: str):
            async with semaphore:
                success = await self.download_file(bucket_name, object_name, file_path)
                results[object_name] = success

        tasks = [download_with_semaphore(on, fp) for on, fp in files]
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def batch_delete(
        self,
        bucket_name: str,
        object_names: list[str],
        max_concurrent: int = 10,
    ) -> dict[str, bool]:
        """
        Delete multiple files concurrently.
        
        Args:
            bucket_name: Name of the bucket
            object_names: List of object names to delete
            max_concurrent: Maximum concurrent deletes
        
        Returns:
            Dictionary mapping object_name to success status
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}

        async def delete_with_semaphore(object_name: str):
            async with semaphore:
                success = await self.delete_file(bucket_name, object_name)
                results[object_name] = success

        tasks = [delete_with_semaphore(on) for on in object_names]
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def get_transfer_progress(self, object_name: str) -> TransferProgress | None:
        """Get current transfer progress for an object."""
        return self._transfer_progress.get(object_name)

    def get_stats_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            **self._stats,
            "active_transfers": len(self._transfer_progress),
            "cache_hit_rate": (
                self._stats["cache_hits"] / (self._stats["cache_hits"] + self._stats["cache_misses"])
                if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "uploads": 0,
            "downloads": 0,
            "deletes": 0,
            "total_bytes_uploaded": 0,
            "total_bytes_downloaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)


# Environment variables
S3_COMPATIBLE_ENDPOINT = os.getenv("S3_COMPATIBLE_ENDPOINT", "localhost:9000")
S3_COMPATIBLE_ACCESS_KEY = os.getenv("S3_COMPATIBLE_ACCESS_KEY", "minioadmin")
S3_COMPATIBLE_SECRET_KEY = os.getenv("S3_COMPATIBLE_SECRET_KEY", "minioadmin")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
MINIO_CACHE_TTL = int(os.getenv("MINIO_CACHE_TTL", "3600"))
MINIO_ENABLE_INTEGRITY_CHECK = os.getenv("MINIO_ENABLE_INTEGRITY_CHECK", "true").lower() == "true"

# Global client instance (will be initialized with Redis if available)
async_minio_client: AsyncMinioClient | None = None


def get_minio_client(redis_client: Any = None) -> AsyncMinioClient:
    """
    Get or create the global MinIO client instance.
    Can be called with Redis client to enable caching.
    """
    global async_minio_client
    
    if async_minio_client is None:
        async_minio_client = AsyncMinioClient(
            endpoint=S3_COMPATIBLE_ENDPOINT,
            access_key=S3_COMPATIBLE_ACCESS_KEY,
            secret_key=S3_COMPATIBLE_SECRET_KEY,
            region=S3_REGION,
            redis_client=redis_client,
            cache_ttl=MINIO_CACHE_TTL,
            enable_integrity_check=MINIO_ENABLE_INTEGRITY_CHECK,
        )
    elif redis_client and not async_minio_client.redis_client:
        # Update existing client with Redis if not already set
        async_minio_client.redis_client = redis_client
    
    return async_minio_client


# Initialize default client (without Redis - can be updated later)
async_minio_client = AsyncMinioClient(
    endpoint=S3_COMPATIBLE_ENDPOINT,
    access_key=S3_COMPATIBLE_ACCESS_KEY,
    secret_key=S3_COMPATIBLE_SECRET_KEY,
    region=S3_REGION,
    cache_ttl=MINIO_CACHE_TTL,
    enable_integrity_check=MINIO_ENABLE_INTEGRITY_CHECK,
)
