# Subnet API

Public, read-only API for subnet 56: auditing, tournaments and performance. For validator operators who publish it.

It forwards requests to the validator API on the same machine, keeps only allowlisted fields and caches each successful response for 30 seconds. While the validator answers, public traffic costs it at most one query per route and parameter set every 30 seconds. Errors are not cached.

## Contents

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI app factory and the public routes. |
| `docs.py` | The API reference page at `/docs`. |
| `models.py` | Response models. They define the allowlisted fields. |
| `settings.py` | Settings read from the environment. |
| `validator_client.py` | Cached HTTP client for the validator API. |

## Routes

| Route | Returns |
| --- | --- |
| `GET /health` | Liveness check. |
| `GET /auditing/tasks`, `/auditing/tasks/hotkey/{hotkey}`, `/auditing/tasks/{task_id}` | Task audit data. |
| `GET /auditing/scores-url` | Location of the latest published scores. |
| `GET /tournament/fees`, `/tournament/balance/{coldkey}` | Entry fees and a coldkey's balance. |
| `GET /tournament/latest/details`, `/tournament/{tournament_id}/details` | Tournament details. |
| `GET /v1/performance/latest-tournament-weights`, `/weight-projection`, `/weight-projection-static`, `/last-boss-battle` | Weights and performance. |

## Run it on the validator

Add a port to `.vali.env`:

```bash
SUBNET_API_PORT=9100
```

`ops/validator_ops/start_validator.sh` then starts it under pm2 as `subnet_api` whenever the validator starts or auto-updates. It reuses `VALIDATOR_PORT` and `FRONTEND_API_KEY` from `.vali.env`. It listens on `0.0.0.0`, so the port is reachable on the machine's public IP.

To start or restart it without restarting the validator:

```bash
task subnet-api
curl -s localhost:${SUBNET_API_PORT}/health
```

Open the firewall for that port if needed, then point your domain's DNS A/AAAA records at the validator's public IP.

Optional settings, also read from the environment:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SUBNET_API_CACHE_SECONDS` | `30` | How long responses are cached |
| `SUBNET_API_REQUEST_TIMEOUT` | `30` | Seconds to wait for the validator |
| `SUBNET_API_SLOW_REQUEST_TIMEOUT` | `120` | Same, for the tournament details and performance routes |
| `SUBNET_API_CORS_ORIGINS` | none | Comma-separated browser origins allowed to call it |
| `SUBNET_API_PUBLIC_URL` | `https://api.sn56.io` | Base URL shown in the code samples on `/docs` |

The API reference is served at `/docs`, from `/openapi.json`.

Only `SUBNET_API_PORT` is published. Keep `VALIDATOR_PORT` firewalled.
