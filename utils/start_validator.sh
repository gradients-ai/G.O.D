#!/bin/bash

# Read port from .vali.env
VALIDATOR_PORT=$(grep VALIDATOR_PORT .vali.env | cut -d '=' -f2)

# Delete old validator services
pm2 delete validator || true
pm2 delete validator_api || true
pm2 delete validator_cycle || true
pm2 delete weight_setter || true
pm2 delete tournament_orchestrator || true
pm2 delete tournament_cycle || true
pm2 delete dstack_orchestrator || true

# Load variables from .vali.env
set -a # Automatically export all variables
. .vali.env
set +a # Stop automatic export

# Start the validator service using opentelemetry-instrument with combined env vars

OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:4317" \
OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true" \
OTEL_PYTHON_LOG_CORRELATION="true" \
pm2 start \
    "opentelemetry-instrument \
    --logs_exporter otlp \
    --traces_exporter none \
    --metrics_exporter otlp \
    --service_name validator \
    uvicorn \
    --factory validator.asgi:factory \
    --host 0.0.0.0 \
    --port ${VALIDATOR_PORT} \
    --env-file .vali.env" \
    --name validator_api

# Start the validator_cycle service using opentelemetry-instrument
OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:4317" \
OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true" \
OTEL_PYTHON_LOG_CORRELATION="true" \
pm2 start \
    "opentelemetry-instrument \
    --logs_exporter otlp \
    --traces_exporter none \
    --metrics_exporter otlp \
    --service_name validator_cycle \
    python -u -m validator.cycle.main" \
    --name validator_cycle

pm2 start \
    "python -m validator.core.weight_setting" \
    --name weight_setter

# Start the tournament orchestrator service using opentelemetry-instrument
OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:4317" \
OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true" \
OTEL_PYTHON_LOG_CORRELATION="true" \
pm2 start \
    "opentelemetry-instrument \
    --logs_exporter otlp \
    --traces_exporter none \
    --metrics_exporter otlp \
    --service_name tournament_orchestrator \
    python -u -m validator.tournament.orchestrator" \
    --name tournament_orchestrator

# start the tournament cycle 
OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:4317" \
OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true" \
OTEL_PYTHON_LOG_CORRELATION="true" \
pm2 start \
    "opentelemetry-instrument \
    --logs_exporter otlp \
    --traces_exporter none \
    --metrics_exporter otlp \
    --service_name tournament_cycle \
    python -u -m validator.tournament.cycle" \
    --name tournament_cycle

# start the dstack orchestrator (separate from tournament_cycle for easier log viewing)
OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:4317" \
OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true" \
OTEL_PYTHON_LOG_CORRELATION="true" \
pm2 start \
    "opentelemetry-instrument \
    --logs_exporter otlp \
    --traces_exporter none \
    --metrics_exporter otlp \
    --service_name dstack_orchestrator \
    python -u -m validator.tournament.dstack_orchestrator" \
    --name dstack_orchestrator

# Start Grafana and observability stack
echo "Starting Grafana and observability stack..."
docker compose --env-file .vali.env -f docker-compose.observability.yml up -d grafana
