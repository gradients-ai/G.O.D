"""Constants for prepared tournament image builds. Environment toggles are read at call time (see config.py)."""

from core.models.image_models import ImageModelType


ENABLED_ENV = "TOURNAMENT_IMAGE_BUILDS_ENABLED"

WORKER_INTERVAL_SECONDS = 15
# A build attempt holds its lease while the builder reports progress; a builder that disappears (trainer restart,
# network loss) loses the lease and the attempt is retried elsewhere.
LEASE_SECONDS = 10 * 60
# Hard ceiling on one attempt regardless of trainer reports (trainer kills at 60 min / 10 min without output).
ATTEMPT_HARD_TIMEOUT_SECONDS = 90 * 60
BUILDER_HTTP_TIMEOUT_SECONDS = 20.0
DEFAULT_TRAINER_PORT = 8001
GLOBAL_MAX_CONCURRENT_BUILDS = 48

# Trainings wait for their image this long after the build row was created, then fall back to building on the
# trainer (today's behaviour). Measured cold builds take 5-10 min, with retries well under this.
MAX_WAIT_FOR_IMAGE_SECONDS = 2 * 60 * 60

# Max attempts per failure class of the latest attempt. Deterministic failures (missing Dockerfile, bad repo/token)
# are not retried; a failed build never fails the miner, training builds on the trainer instead.
MAX_ATTEMPTS_BY_FAILURE_CLASS = {
    "infrastructure": 4,
    "cancelled": 4,
    "lease_expired": 4,
    "stalled": 2,
    "timeout": 2,
    "build": 2,
    "dockerfile": 1,
    "clone": 1,
}
RETRY_BACKOFF_SECONDS = (60, 180, 600)

# Image tournaments build the toolkit image up front; the FLUX Dockerfile is only built if a FLUX task appears.
FLUX_MODEL_TYPES = frozenset({ImageModelType.FLUX})

# Prune completed tournaments' images from R2 on Fridays (after the Friday 14:00 UTC tournament deadline)
# through the weekend; blobs no repository references are garbage-collected once they are this old.
PRUNE_START_WEEKDAY = 4
PRUNE_START_HOUR_UTC = 15
PRUNE_CHECK_INTERVAL_SECONDS = 60 * 60
ORPHAN_BLOB_MIN_AGE_HOURS = 24

# Shadeform fallback builder (used only when no trainer can take queued builds).
SHADEFORM_API_URL = "https://api.shadeform.ai/v1"
SHADEFORM_INSTANCE_NAME_PREFIX = "god-image-builder"
SHADEFORM_TAG = "god-image-builder"
SHADEFORM_FALLBACK_AFTER_SECONDS = 10 * 60
SHADEFORM_IDLE_SECONDS = 15 * 60
SHADEFORM_MAX_LIFETIME_HOURS = 8
SHADEFORM_MAX_SPEND_USD = "40.00"
SHADEFORM_BOOT_TIMEOUT_SECONDS = 30 * 60
SHADEFORM_MIN_VCPUS = 32
SHADEFORM_MIN_STORAGE_GB = 1000
SHADEFORM_MAX_HOURLY_PRICE_CENTS = 400
