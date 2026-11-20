# Subnet API (sn56.io)

Public, read-only API for subnet 56: auditing, tournaments and performance. It
forwards to the validator API on the same machine, keeps only allowlisted fields,
and caches responses for 30 seconds, so public traffic costs the validator at most
one query per route and parameters every 30 seconds.

## Run it on the validator

Add to `.vali.env`:

```bash
SUBNET_API_PORT=9100
SUBNET_API_DOMAIN=api.sn56.io   # optional: serve it over HTTPS through Caddy
```

`task validator` (also run by the auto-updater) then:

1. Runs `ops/validator_ops/setup_caddy.sh` when `SUBNET_API_DOMAIN` is set. It
   installs Caddy from the official apt repository if missing, writes
   `/etc/caddy/Caddyfile` to proxy `SUBNET_API_DOMAIN` to `127.0.0.1:SUBNET_API_PORT`
   (backing up any different existing file to `Caddyfile.bak.<timestamp>`), and
   enables or reloads the `caddy` systemd service. Caddy gets and renews the
   Let's Encrypt certificate itself. It only uses sudo when something changes, and
   a failure here logs a warning without blocking the validator restart.
2. Starts the API under pm2 as `subnet_api` from `ops/validator_ops/start_validator.sh`,
   reusing `VALIDATOR_PORT` and `FRONTEND_API_KEY` from `.vali.env`. With
   `SUBNET_API_DOMAIN` set it listens on `127.0.0.1` only; without it, on `0.0.0.0`
   so the port is reachable on the public IP directly.

For HTTPS, point the domain's DNS A/AAAA records at the validator and open ports
80 and 443 (Let's Encrypt validates over them). Then:

```bash
curl -s localhost:${SUBNET_API_PORT}/health
curl -s https://${SUBNET_API_DOMAIN}/health
journalctl -u caddy -f   # certificate / proxy logs
```

To restart only the API: `pm2 restart subnet_api`.

Optional settings, also read from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `SUBNET_API_DOMAIN` | none | Domain Caddy serves over HTTPS; also binds the API to localhost |
| `SUBNET_API_CACHE_SECONDS` | `30` | How long responses are cached |
| `SUBNET_API_REQUEST_TIMEOUT` | `30` | Seconds to wait for the validator |
| `SUBNET_API_SLOW_REQUEST_TIMEOUT` | `120` | Same, for the heavy performance routes |
| `SUBNET_API_CORS_ORIGINS` | none | Comma-separated browser origins allowed to call it |
| `SUBNET_API_PUBLIC_URL` | `https://api.sn56.io` | Base URL shown in the code samples on `/docs` |

The API reference is served at `/docs`, from `/openapi.json`.

Keep `VALIDATOR_PORT` firewalled / bound as today. Publish only 80/443 (with Caddy)
or `SUBNET_API_PORT` (without it).
