# Validator Infrastructure

Clients and adapters for external systems used by the validator.

## Contents

- `cache.py`: Redis/cache helpers.
- `cache_policy.py`: cache keys and TTL policy.
- `comfy_gateway.py`: ComfyUI gateway integration.
- `content_service.py`: Gradients/content-service API calls.
- `fal_client.py`: Fal client integration.
- `llm.py`: LLM client helpers.
- `minio_client.py`: MinIO/S3-compatible storage client. Configured by `S3_COMPATIBLE_ENDPOINT`, `S3_COMPATIBLE_ACCESS_KEY`, `S3_COMPATIBLE_SECRET_KEY`, `S3_REGION` and `S3_SECURE` (default `true`, HTTPS; `false` for a plain-HTTP local MinIO), all read at import. `parse_s3_url` reads both virtual-host (`bucket.host/key`) and path-style (`host/bucket/key`) URLs.
- `models.py`: validator infrastructure helper schemas.
- `retries.py`: retry decorators and retry policy.
- `service_constants.py`: service endpoint constants.
- `substrate.py`: Substrate/Bittensor helpers.
- `__init__.py`: package marker.
