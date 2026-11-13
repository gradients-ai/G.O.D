# Quant decoder sandbox, build phase: the run image plus a C/C++ toolchain.
# Only decoder/ source is mounted in this phase, so artifact bytes can never be compiled.
FROM gradientsio/quant-decoder-run:pinned

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ make libc6-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
