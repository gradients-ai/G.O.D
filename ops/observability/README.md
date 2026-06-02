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
