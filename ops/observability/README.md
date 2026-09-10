# Observability

Grafana, Loki, Prometheus, Tempo, Vector, and nginx configuration for validator and trainer observability.

## Contents

- `nginx/`: nginx reverse-proxy configuration for observability services.
- `vector/`: Vector log shipping configuration.
- `grafana-*.yaml`: Grafana provisioning configuration.
- `grafana-*.json`: Grafana dashboards.
- `loki-config.yaml`: Loki config for validator logs.
- `loki-training-config.yaml`: Loki config for trainer logs.
- `otel-config.yaml`: OpenTelemetry collector config.
- `prometheus-config.yaml`: Prometheus config for validator metrics.
- `prometheus-training-config.yaml`: Prometheus config for training metrics.
- `tempo-config.yaml`: Tempo tracing config.

## Training Grafana

The training dashboard (`grafana-training-dashboard.json`) avoids Loki `label_values(...)` variables and high-cardinality stream matchers such as `task_id=~".*"`. Those queries hang on this Loki for ranges longer than a few hours, so Grafana sits on a spinner.

It defaults to `now-2d` because trainers often stop shipping for a day and a 6-hour window looks empty. Log panels query `{job="docker-training-containers"}` only; Line filter is optional extra LogQL (for example `|~ "loss"` or `| task_id =~ "uuid"`). Metric panels stay on the last 6 hours.
