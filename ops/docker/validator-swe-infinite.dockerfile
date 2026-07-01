# Image for validator/evaluation/evaluators/swe.py.
# Serves a candidate model with SGLang and calls an external Affinetes
# SWE Infinite environment server.
# Build manually:
#   docker build -f ops/docker/validator-swe-infinite.dockerfile -t gradientsio/env-eval-swe-infinite:basilica .

FROM lmsysorg/sglang:latest

WORKDIR /app

COPY pyproject.toml README.md ./
COPY core core
COPY miner miner
COPY ops ops
COPY trainer trainer
COPY validator validator

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed .

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    peft==0.18.1 accelerate==1.6.0

RUN apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*

COPY . /app

ENV SGLANG_PORT=30000
ENV SGLANG_BASE_URL=http://127.0.0.1:30000
ENV SGLANG_HEALTH_PATH=/v1/models
