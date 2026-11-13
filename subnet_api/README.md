# Subnet API

Public, read-only API for subnet 56: auditing, tournaments and performance. For validator operators who publish it.

It forwards requests to the validator API on the same machine, keeps only allowlisted fields and caches each successful response for 30 seconds. While the validator answers, public traffic costs it at most one query per route and parameter set every 30 seconds. Errors are not cached.

## Contents

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI app factory and the public routes. |
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

`ops/validator_ops/start_validator.sh` then starts it under pm2 as `subnet_api` whenever the validator starts or auto-updates. It reuses `VALIDATOR_PORT` and `FRONTEND_API_KEY` from `.vali.env`. It listens on `127.0.0.1` only, so the Cloudflare tunnel below is its only public entry point.

To start it without restarting the validator:

```bash
set -a; . ./.vali.env; set +a
VALIDATOR_URL="http://127.0.0.1:${VALIDATOR_PORT}" VALIDATOR_API_KEY="${FRONTEND_API_KEY}" \
  pm2 start "uvicorn --factory subnet_api.app:app --host 127.0.0.1 --port ${SUBNET_API_PORT} --no-server-header" \
  --name subnet_api
curl -s localhost:${SUBNET_API_PORT}/health
```

Optional settings, also read from the environment:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SUBNET_API_CACHE_SECONDS` | `30` | How long responses are cached |
| `SUBNET_API_REQUEST_TIMEOUT` | `30` | Seconds to wait for the validator |
| `SUBNET_API_SLOW_REQUEST_TIMEOUT` | `120` | Same, for the tournament details and performance routes |
| `SUBNET_API_CORS_ORIGINS` | none | Comma-separated browser origins allowed to call it |

## Publishing it through a Cloudflare tunnel

1. **Put the domain on Cloudflare.** In the Cloudflare dashboard, *Add a domain* `<your-domain>`, then set the two nameservers it shows at your registrar. Wait until the domain shows *Active*.
2. **Install cloudflared on the validator** (Ubuntu/Debian):

   ```bash
   sudo mkdir -p --mode=0755 /usr/share/keyrings
   curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
     | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
   echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' \
     | sudo tee /etc/apt/sources.list.d/cloudflared.list
   sudo apt-get update && sudo apt-get install -y cloudflared
   ```

3. **Create the tunnel.** *Zero Trust → Networks → Tunnels → Create a tunnel → Cloudflared*, give it a name. Copy the `sudo cloudflared service install <token>` command it shows and run it on the validator. It installs a systemd service that reconnects after reboots. The tunnel then shows *Healthy*.
4. **Route the domain.** In the tunnel's *Public hostnames*, add hostname `<your-domain>` (no subdomain) with service type `HTTP` and URL `localhost:9100` (your `SUBNET_API_PORT`). Cloudflare creates the DNS record.
5. **Check it:** `curl https://<your-domain>/health` returns `{"status":"ok"}`.

The tunnel only makes outbound connections, so no inbound port is opened on the validator. Optionally, add a Cloudflare rate-limiting rule for the domain to cap requests per client IP.
