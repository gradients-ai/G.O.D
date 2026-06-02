# Miner

FastAPI service exposed by tournament miners. The service tells validators which training repository and commit to run for each tournament type.

## Contents

- `asgi.py`: FastAPI application factory and server entrypoint.
- `endpoints/`: miner API route modules.
- `__init__.py`: package marker.

Run locally with `task miner`. Tournament participation details live in `docs/miner.md`.
