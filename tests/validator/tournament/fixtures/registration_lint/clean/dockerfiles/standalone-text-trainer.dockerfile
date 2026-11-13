# A build that passes every rule: hashes, SHA-pinned git, a digest-pinned base, checksummed fetches.
ARG FIBER_SHA=0123456789abcdef0123456789abcdef01234567
FROM python:3.11-slim AS tools
FROM ghcr.io/someone/cuda-base@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

RUN apt-get update && apt-get install -y wget curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY scripts/requirements.txt scripts/install_extras.sh ./
COPY --from=tools /usr/local/bin/python3 /usr/local/bin/python3-tools
ADD --checksum=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb https://example.com/tokenizer.tar.gz /workspace/

RUN pip install --no-deps --require-hashes -r requirements.txt \
    && pip install "fiber @ git+https://github.com/rayonlabs/fiber.git@${FIBER_SHA}" \
    && pip install "https://example.com/wheels/kernel-1.0-py3-none-any.whl#sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" \
    && pip install -e .

RUN curl -fsSL -o /tmp/tool.tgz https://example.com/tool.tgz \
    && echo "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb  /tmp/tool.tgz" | sha256sum -c -

RUN git clone https://github.com/example/kernels.git /opt/kernels \
    && git -C /opt/kernels checkout 0123456789abcdef0123456789abcdef01234567

RUN bash install_extras.sh

COPY scripts /workspace/scripts
ENTRYPOINT ["./scripts/entrypoint.sh"]
