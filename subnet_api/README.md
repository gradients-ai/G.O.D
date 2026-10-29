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
`FRONTEND_API_KEY` from `.vali.env`, and listens on `0.0.0.0` so the port is
reachable on the machine's public IP.

To start (or restart) it without restarting the validator:

```bash
task subnet-api
curl -s localhost:${SUBNET_API_PORT}/health
```

Open the firewall for that port if needed, then point `sn56.io` (or any host) at
the validator's public IP. DNS A/AAAA records expose that IP.

Optional settings, also read from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `SUBNET_API_CACHE_SECONDS` | `30` | How long responses are cached |
| `SUBNET_API_REQUEST_TIMEOUT` | `30` | Seconds to wait for the validator |
| `SUBNET_API_SLOW_REQUEST_TIMEOUT` | `120` | Same, for the heavy performance routes |
| `SUBNET_API_CORS_ORIGINS` | none | Comma-separated browser origins allowed to call it |

Keep `VALIDATOR_PORT` firewalled / bound as today — only publish `SUBNET_API_PORT`.
