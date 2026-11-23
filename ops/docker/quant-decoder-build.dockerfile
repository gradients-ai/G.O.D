# Quant decoder sandbox, build phase: the run image plus a C/C++ toolchain.
# Only decoder/ source is mounted in this phase, so artifact bytes can never be compiled.
# CI builds it on the run image from the same commit; a local build needs the run image built first.
FROM gradientsio/quant-decoder-run:pinned

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ make libc6-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
