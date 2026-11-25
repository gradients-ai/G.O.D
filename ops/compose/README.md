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

Prometheus (`9090`), the OpenTelemetry collector (`4317`) and the trainer's Vector ports (`8686`, `8688`, `9090`) are published on `127.0.0.1` only. Prometheus runs without `--web.enable-lifecycle`: restart the container to load a changed config. The validator Grafana needs `GRAFANA_PASSWORD` in `.vali.env`.

## See also

- [Running a validator](../../docs/developer.md#running-a-validator): how the validator services are started.
