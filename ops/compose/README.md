# Compose

Docker Compose stacks for validator services, local development, observability and trainer log shipping.

## Contents

| Path | Purpose |
| --- | --- |
| `docker-compose.yml` | Validator dependencies: Postgres migrations (`dbmate`) and Redis. |
| `docker-compose.dev.yml` | Development additions: Redis, Redis Commander and Adminer. |
| `docker-compose.observability.yml` | Validator observability: Grafana, Loki, Prometheus, Tempo, the OpenTelemetry collector and Vector. |
| `docker-compose.observability-server.yml` | Training observability server: Grafana, Loki and Prometheus for trainer logs. |
| `docker-compose.trainer-server.yml` | Trainer host log shipping: Vector and node exporter. |

## See also

- [Running a validator](../../docs/developer.md#running-a-validator): how the validator services are started.
