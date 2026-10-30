# Subnet API (sn56.io)

Public, read-only API for subnet 56: auditing, tournaments and performance. It
forwards to the validator API on the same machine, keeps only allowlisted fields,
and caches responses for 30 seconds, so public traffic costs the validator at most
one query per route and parameters every 30 seconds.

## Run it on the validator

Add a port to `.vali.env`:

```bash
SUBNET_API_PORT=9100
```

`ops/validator_ops/start_validator.sh` then starts it under pm2 as `subnet_api`
whenever the validator starts or auto-updates. It reuses `VALIDATOR_PORT` and
`FRONTEND_API_KEY` from `.vali.env`, and listens on `127.0.0.1` only: the
Cloudflare tunnel below is its only public entry point.

To start it now without restarting the validator:

```bash
set -a; . ./.vali.env; set +a
VALIDATOR_URL="http://127.0.0.1:${VALIDATOR_PORT}" VALIDATOR_API_KEY="${FRONTEND_API_KEY}" \
  pm2 start "uvicorn --factory subnet_api.app:app --host 127.0.0.1 --port ${SUBNET_API_PORT} --no-server-header" \
  --name subnet_api
curl -s localhost:${SUBNET_API_PORT}/health
```

Optional settings, also read from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `SUBNET_API_CACHE_SECONDS` | `30` | How long responses are cached |
| `SUBNET_API_REQUEST_TIMEOUT` | `30` | Seconds to wait for the validator |
| `SUBNET_API_SLOW_REQUEST_TIMEOUT` | `120` | Same, for the heavy performance routes |
| `SUBNET_API_CORS_ORIGINS` | none | Comma-separated browser origins allowed to call it |
| `SUBNET_API_PUBLIC_URL` | `https://api.sn56.io` | Base URL shown in the code samples on `/docs` |

The API reference is served at `/docs`, from `/openapi.json`.

## Publish it on sn56.io with Cloudflare

1. **Put the domain on Cloudflare.** In the Cloudflare dashboard, *Add a domain*
   `sn56.io`, then set the two nameservers it shows at your registrar. Wait until
   the domain shows *Active*.
2. **Install cloudflared on the validator** (Ubuntu/Debian):

   ```bash
   sudo mkdir -p --mode=0755 /usr/share/keyrings
   curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
     | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
   echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' \
     | sudo tee /etc/apt/sources.list.d/cloudflared.list
   sudo apt-get update && sudo apt-get install -y cloudflared
   ```

3. **Create the tunnel.** *Zero Trust → Networks → Tunnels → Create a tunnel →
   Cloudflared*, name it `sn56`. Copy the `sudo cloudflared service install <token>`
   command it shows and run it on the validator. It installs a systemd service
   that reconnects on its own after reboots; the tunnel turns *Healthy*.
4. **Route the domain.** In the tunnel's *Public hostnames*, add hostname
   `sn56.io` (no subdomain) with service type `HTTP` and URL
   `localhost:9100` (your `SUBNET_API_PORT`). Cloudflare creates the DNS record.
5. **Check it:** `curl https://sn56.io/health` returns `{"status":"ok"}`.

The tunnel only makes outbound connections, so no inbound port is opened on the
validator. Optionally, add a Cloudflare rate-limiting rule for `sn56.io` to cap
requests per client IP.
