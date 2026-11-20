# Validator infrastructure

Clients for the external systems the validator uses, and the repository review and duplicate detection that call them.

## Contents

| Path | Purpose |
| --- | --- |
| `cache.py` | Local Hugging Face cache and temp file cleanup. |
| `cache_policy.py` | Cache size limits, lookup windows and the cleanup interval. |
| `challenger_code_review.py` | Code review of a round 2 entrant's repository. |
| `challenger_code_review_config.json` | Configuration of that review. |
| `challenger_dependency_review.py` | Collects an entrant's third-party dependencies for the code review. |
| `content_service.py` | Signed requests to the content and inference services. |
| `dstack_client.py` | Async client for the dstack HTTP API, used by training and evaluation. |
| `fal_client.py` | Fal image generation client. |
| `llm.py` | Chat completion helpers, including reasoning removal. |
| `minio_client.py` | S3-compatible storage client. `parse_s3_url` reads virtual-host and path-style URLs. |
| `models.py` | Chat message models. |
| `organic_dstack.py` | Chain-free dstack helpers for organic training. |
| `repo_dedup.py` | Detects functionally equivalent tournament submissions. |
| `repo_dedup_config.json` | Configuration of the duplicate detection. |
| `retries.py` | Retry decorators. |
| `service_constants.py` | Service endpoints, bucket names and shared field names. |
| `substrate.py` | Substrate query helper. |

## See also

- [Fair play and review](../../docs/miner.md#fair-play-and-review): the review and duplicate rules as a miner sees them.
- [Setup](../../docs/developer.md#setup): installation and env files.
