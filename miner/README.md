# Miner

FastAPI service a tournament miner runs. It tells validators which training repository and commit to run for each tournament type.

## Contents

| Path | Purpose |
| --- | --- |
| `asgi.py` | FastAPI application factory and server entrypoint. |
| `endpoints/` | Miner API routes. |

## See also

- [Entering a tournament](../docs/miner.md#entering-a-tournament): what a miner serves and how an entry is charged.
- [Running a miner locally](../docs/developer.md#running-a-miner-locally): how to start the service.
