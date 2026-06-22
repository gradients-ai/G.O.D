"""
Storage Analytics and Monitoring Module

Provides comprehensive storage analytics, monitoring, and lifecycle management
for MinIO/S3 storage operations.
"""

import asyncio
import datetime
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any

from validator.utils.logging import get_logger
from validator.utils.minio import AsyncMinioClient


logger = get_logger(__name__)


@dataclass
class StorageMetrics:
    """Storage usage metrics for a bucket or prefix."""
    bucket_name: str
    prefix: str | None = None
    total_objects: int = 0
    total_size_bytes: int = 0
    total_size_mb: float = 0.0
    total_size_gb: float = 0.0
    average_object_size: float = 0.0
    oldest_object: datetime.datetime | None = None
    newest_object: datetime.datetime | None = None
    objects_by_type: dict[str, int] = None
    size_by_type: dict[str, int] = None

    def __post_init__(self):
        if self.objects_by_type is None:
            self.objects_by_type = defaultdict(int)
        if self.size_by_type is None:
            self.size_by_type = defaultdict(int)
        
        if self.total_size_bytes > 0:
            self.total_size_mb = self.total_size_bytes / (1024 * 1024)
            self.total_size_gb = self.total_size_bytes / (1024 * 1024 * 1024)
        
        if self.total_objects > 0:
            self.average_object_size = self.total_size_bytes / self.total_objects


@dataclass
class AccessPattern:
    """Access pattern analysis for storage objects."""
    object_name: str
    access_count: int = 0
    last_accessed: datetime.datetime | None = None
    first_accessed: datetime.datetime | None = None
    is_hot: bool = False  # Frequently accessed
    is_cold: bool = False  # Rarely accessed
    access_frequency: float = 0.0  # Accesses per day


@dataclass
class StorageCostEstimate:
    """Estimated storage costs."""
    storage_cost_per_gb_month: float = 0.023  # Default S3 standard pricing
    total_storage_gb: float = 0.0
    monthly_cost: float = 0.0
    annual_cost: float = 0.0
    transfer_cost_per_gb: float = 0.09  # Default egress pricing
    estimated_transfer_gb: float = 0.0
    transfer_cost: float = 0.0


class StorageAnalytics:
    """
    Comprehensive storage analytics and monitoring system.
    
    Provides:
    - Storage usage tracking
    - Cost estimation
    - Access pattern analysis
    - Performance metrics
    - Lifecycle management recommendations
    """

    def __init__(
        self,
        minio_client: AsyncMinioClient,
        redis_client: Any = None,
        cache_ttl: int = 3600,
    ):
        self.minio_client = minio_client
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        self._access_patterns: dict[str, AccessPattern] = {}
        self._metrics_cache: dict[str, StorageMetrics] = {}
        self._last_scan_time: dict[str, datetime.datetime] = {}

    async def _get_from_cache(self, key: str) -> Any | None:
        """Get value from Redis cache if available."""
        if not self.redis_client:
            return None
        try:
            import json
            cached = await self.redis_client.get(key)
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None

    async def _set_cache(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in Redis cache."""
        if not self.redis_client:
            return
        try:
            import json
            ttl = ttl or self.cache_ttl
            await self.redis_client.set(key, json.dumps(value, default=str), ex=ttl)
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")

    async def get_storage_metrics(
        self,
        bucket_name: str,
        prefix: str | None = None,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> StorageMetrics:
        """
        Calculate comprehensive storage metrics for a bucket or prefix.
        
        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix to filter objects
            use_cache: Whether to use cached results
            force_refresh: Force refresh even if cache exists
        
        Returns:
            StorageMetrics object with detailed statistics
        """
        cache_key = f"storage_metrics:{bucket_name}:{prefix or ''}"
        
        if use_cache and not force_refresh:
            cached = await self._get_from_cache(cache_key)
            if cached:
                # Reconstruct from dict
                metrics = StorageMetrics(**cached)
                if metrics.objects_by_type is None:
                    metrics.objects_by_type = defaultdict(int)
                if metrics.size_by_type is None:
                    metrics.size_by_type = defaultdict(int)
                return metrics

        logger.info(f"Calculating storage metrics for {bucket_name}/{prefix or ''}")

        objects = await self.minio_client.list_objects(
            bucket_name,
            prefix=prefix,
            recursive=True,
            include_metadata=False,
        )

        metrics = StorageMetrics(bucket_name=bucket_name, prefix=prefix)
        metrics.total_objects = len(objects)

        oldest_time = None
        newest_time = None

        for obj in objects:
            size = obj.get("size", 0)
            metrics.total_size_bytes += size

            # Track by file extension
            object_name = obj.get("object_name", "")
            ext = os.path.splitext(object_name)[1].lower() or "no_extension"
            metrics.objects_by_type[ext] += 1
            metrics.size_by_type[ext] += size

            # Track oldest/newest
            last_modified_str = obj.get("last_modified")
            if last_modified_str:
                try:
                    last_modified = datetime.datetime.fromisoformat(last_modified_str.replace("Z", "+00:00"))
                    if oldest_time is None or last_modified < oldest_time:
                        oldest_time = last_modified
                        metrics.oldest_object = last_modified
                    if newest_time is None or last_modified > newest_time:
                        newest_time = last_modified
                        metrics.newest_object = last_modified
                except Exception:
                    pass

        # Calculate derived metrics
        if metrics.total_size_bytes > 0:
            metrics.total_size_mb = metrics.total_size_bytes / (1024 * 1024)
            metrics.total_size_gb = metrics.total_size_bytes / (1024 * 1024 * 1024)

        if metrics.total_objects > 0:
            metrics.average_object_size = metrics.total_size_bytes / metrics.total_objects

        # Cache results
        await self._set_cache(cache_key, asdict(metrics), ttl=self.cache_ttl)
        self._metrics_cache[cache_key] = metrics
        self._last_scan_time[cache_key] = datetime.datetime.now()

        logger.info(
            f"Storage metrics for {bucket_name}/{prefix or ''}: "
            f"{metrics.total_objects} objects, {metrics.total_size_gb:.2f} GB"
        )

        return metrics

    async def get_cost_estimate(
        self,
        bucket_name: str,
        prefix: str | None = None,
        storage_cost_per_gb_month: float = 0.023,
        transfer_cost_per_gb: float = 0.09,
        estimated_transfer_gb: float = 0.0,
    ) -> StorageCostEstimate:
        """
        Estimate storage costs for a bucket or prefix.
        
        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix to filter objects
            storage_cost_per_gb_month: Cost per GB per month for storage
            transfer_cost_per_gb: Cost per GB for data transfer
            estimated_transfer_gb: Estimated monthly transfer in GB
        
        Returns:
            StorageCostEstimate with cost breakdown
        """
        metrics = await self.get_storage_metrics(bucket_name, prefix)

        estimate = StorageCostEstimate(
            storage_cost_per_gb_month=storage_cost_per_gb_month,
            total_storage_gb=metrics.total_size_gb,
            transfer_cost_per_gb=transfer_cost_per_gb,
            estimated_transfer_gb=estimated_transfer_gb,
        )

        estimate.monthly_cost = estimate.total_storage_gb * storage_cost_per_gb_month
        estimate.annual_cost = estimate.monthly_cost * 12
        estimate.transfer_cost = estimate.estimated_transfer_gb * transfer_cost_per_gb

        return estimate

    async def analyze_access_patterns(
        self,
        bucket_name: str,
        prefix: str | None = None,
        hot_threshold_days: int = 7,
        cold_threshold_days: int = 90,
    ) -> dict[str, AccessPattern]:
        """
        Analyze access patterns for objects to identify hot/cold data.
        
        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix to filter objects
            hot_threshold_days: Objects accessed within this many days are "hot"
            cold_threshold_days: Objects not accessed for this many days are "cold"
        
        Returns:
            Dictionary mapping object_name to AccessPattern
        """
        objects = await self.minio_client.list_objects(
            bucket_name,
            prefix=prefix,
            recursive=True,
            include_metadata=False,
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        patterns = {}

        for obj in objects:
            object_name = obj.get("object_name", "")
            last_modified_str = obj.get("last_modified")

            pattern = AccessPattern(object_name=object_name)

            if last_modified_str:
                try:
                    last_modified = datetime.datetime.fromisoformat(
                        last_modified_str.replace("Z", "+00:00")
                    )
                    pattern.last_accessed = last_modified
                    pattern.first_accessed = last_modified  # Approximate

                    days_since_access = (now - last_modified).days
                    pattern.is_hot = days_since_access <= hot_threshold_days
                    pattern.is_cold = days_since_access >= cold_threshold_days

                    # Estimate access frequency (simplified - assumes last_modified = last_access)
                    if days_since_access > 0:
                        pattern.access_frequency = 1.0 / days_since_access
                    else:
                        pattern.access_frequency = 1.0

                except Exception:
                    pass

            patterns[object_name] = pattern

        # Store patterns
        for obj_name, pattern in patterns.items():
            self._access_patterns[obj_name] = pattern

        return patterns

    async def get_lifecycle_recommendations(
        self,
        bucket_name: str,
        prefix: str | None = None,
        cold_threshold_days: int = 90,
        archive_threshold_days: int = 180,
    ) -> dict[str, Any]:
        """
        Generate lifecycle management recommendations.
        
        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix to filter objects
            cold_threshold_days: Days before recommending cold storage
            archive_threshold_days: Days before recommending archival
        
        Returns:
            Dictionary with recommendations
        """
        metrics = await self.get_storage_metrics(bucket_name, prefix)
        patterns = await self.analyze_access_patterns(
            bucket_name,
            prefix,
            cold_threshold_days=cold_threshold_days,
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        recommendations = {
            "total_objects": metrics.total_objects,
            "total_size_gb": metrics.total_size_gb,
            "objects_to_archive": [],
            "objects_to_delete": [],
            "objects_for_cold_storage": [],
            "estimated_savings_gb": 0.0,
            "estimated_savings_cost": 0.0,
        }

        for obj_name, pattern in patterns.items():
            if pattern.last_accessed:
                days_since_access = (now - pattern.last_accessed).days

                # Get object size
                obj_size = 0
                for obj in await self.minio_client.list_objects(bucket_name, prefix=obj_name):
                    if obj.get("object_name") == obj_name:
                        obj_size = obj.get("size", 0)
                        break

                if days_since_access >= archive_threshold_days:
                    recommendations["objects_to_archive"].append({
                        "object_name": obj_name,
                        "size_bytes": obj_size,
                        "days_since_access": days_since_access,
                    })
                    recommendations["estimated_savings_gb"] += obj_size / (1024 * 1024 * 1024)
                elif days_since_access >= cold_threshold_days:
                    recommendations["objects_for_cold_storage"].append({
                        "object_name": obj_name,
                        "size_bytes": obj_size,
                        "days_since_access": days_since_access,
                    })

        # Estimate cost savings (assuming 50% cost reduction for cold storage, 80% for archive)
        recommendations["estimated_savings_cost"] = (
            recommendations["estimated_savings_gb"] * 0.023 * 0.8  # Archive savings
        )

        return recommendations

    async def get_performance_metrics(
        self,
        time_window_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get performance metrics from MinIO client statistics.
        
        Args:
            time_window_hours: Time window for metrics
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self.minio_client.get_stats_summary()

        return {
            "uploads": stats.get("uploads", 0),
            "downloads": stats.get("downloads", 0),
            "deletes": stats.get("deletes", 0),
            "total_bytes_uploaded": stats.get("total_bytes_uploaded", 0),
            "total_bytes_downloaded": stats.get("total_bytes_downloaded", 0),
            "total_bytes_uploaded_gb": stats.get("total_bytes_uploaded", 0) / (1024 * 1024 * 1024),
            "total_bytes_downloaded_gb": stats.get("total_bytes_downloaded", 0) / (1024 * 1024 * 1024),
            "cache_hits": stats.get("cache_hits", 0),
            "cache_misses": stats.get("cache_misses", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", 0.0),
            "active_transfers": stats.get("active_transfers", 0),
        }

    async def get_storage_health(
        self,
        bucket_name: str,
    ) -> dict[str, Any]:
        """
        Get overall storage health status.
        
        Args:
            bucket_name: Name of the bucket
        
        Returns:
            Dictionary with health status and recommendations
        """
        try:
            # Check if bucket exists
            bucket_exists = await self.minio_client.ensure_bucket_exists(bucket_name)
            
            if not bucket_exists:
                return {
                    "status": "unhealthy",
                    "bucket_exists": False,
                    "issues": ["Bucket does not exist and could not be created"],
                }

            metrics = await self.get_storage_metrics(bucket_name)
            performance = await self.get_performance_metrics()

            health = {
                "status": "healthy",
                "bucket_exists": True,
                "metrics": {
                    "total_objects": metrics.total_objects,
                    "total_size_gb": metrics.total_size_gb,
                },
                "performance": performance,
                "issues": [],
                "recommendations": [],
            }

            # Check for potential issues
            if metrics.total_size_gb > 1000:  # > 1TB
                health["recommendations"].append(
                    "Consider implementing lifecycle policies for large storage"
                )

            if performance.get("cache_hit_rate", 0) < 0.5:
                health["recommendations"].append(
                    "Cache hit rate is low - consider increasing cache TTL"
                )

            if metrics.average_object_size > 100 * 1024 * 1024:  # > 100MB average
                health["recommendations"].append(
                    "Large average object size - consider compression or chunking"
                )

            return health

        except Exception as e:
            logger.error(f"Error checking storage health: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def cleanup_old_objects(
        self,
        bucket_name: str,
        prefix: str | None = None,
        older_than_days: int = 90,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """
        Clean up objects older than specified days.
        
        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix to filter objects
            older_than_days: Delete objects older than this many days
            dry_run: If True, only report what would be deleted
        
        Returns:
            Dictionary with cleanup results
        """
        objects = await self.minio_client.list_objects(
            bucket_name,
            prefix=prefix,
            recursive=True,
            include_metadata=False,
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff_date = now - datetime.timedelta(days=older_than_days)

        objects_to_delete = []
        total_size = 0

        for obj in objects:
            last_modified_str = obj.get("last_modified")
            if last_modified_str:
                try:
                    last_modified = datetime.datetime.fromisoformat(
                        last_modified_str.replace("Z", "+00:00")
                    )
                    if last_modified < cutoff_date:
                        object_name = obj.get("object_name", "")
                        size = obj.get("size", 0)
                        objects_to_delete.append(object_name)
                        total_size += size
                except Exception:
                    pass

        result = {
            "dry_run": dry_run,
            "objects_found": len(objects_to_delete),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "deleted_objects": [],
            "failed_deletes": [],
        }

        if not dry_run and objects_to_delete:
            delete_results = await self.minio_client.batch_delete(
                bucket_name,
                objects_to_delete,
            )
            result["deleted_objects"] = [
                obj for obj, success in delete_results.items() if success
            ]
            result["failed_deletes"] = [
                obj for obj, success in delete_results.items() if not success
            ]

        return result

