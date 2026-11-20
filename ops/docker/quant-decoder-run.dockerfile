# Quant decoder sandbox, run phase: fixed package set, no compiler, no model weights.
# Every version is pinned so no decoder logic can live in a package outside the counted decoder bytes.
FROM python:3.11.10-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends zstd xz-utils && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy==2.1.3 \
    safetensors==0.4.5 \
    zstandard==0.23.0 && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

WORKDIR /tmp
