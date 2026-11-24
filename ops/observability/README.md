# Observability

Grafana, Loki, Prometheus, Tempo, Vector and nginx configuration for validator and trainer observability.

## Contents

| Path | Purpose |
| --- | --- |
| `nginx/` | Reverse proxy configuration for the observability services. |
| `vector/` | Vector log shipping configuration. |
| `grafana-dashboards.yaml`, `grafana-datasource.yaml` | Grafana provisioning for the validator stack. |
| `grafana-training-dashboards.yaml`, `grafana-training-datasource.yaml` | Grafana provisioning for the training stack. |
| `grafana-validator-general.json`, `grafana-validator-errors.json` | Validator dashboards. |
| `grafana-tournament.json`, `grafana-tournament-costs.json` | Tournament and tournament cost dashboards. |
| `grafana-training-dashboard.json` | Training log dashboard. |
| `loki-config.yaml`, `loki-training-config.yaml` | Loki configuration for validator and trainer logs. |
| `prometheus-config.yaml`, `prometheus-training-config.yaml` | Prometheus configuration for validator and training metrics. |
| `otel-config.yaml` | OpenTelemetry collector configuration. |
| `tempo-config.yaml` | Tempo tracing configuration. |

## Training dashboard queries

Log panels in `grafana-training-dashboard.json` select the single stream `{job="docker-training-containers"}` and filter by `task_id`, `hotkey`, `model` and `task_type` inside the query. Keep panels to that pattern: high-cardinality stream matchers and `label_values(...)` variables are slow on long ranges.

## See also

- [Developer guide](../../docs/developer.md): setup, configuration and running the services.
