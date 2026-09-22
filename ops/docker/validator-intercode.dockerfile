# Image for validator/evaluation/evaluators/intercode.py — runs the InterCode-Bash
# NL2Bash benchmark against a candidate HF model served locally via SGLang.
# Build manually:
#   docker build -f ops/docker/validator-intercode.dockerfile -t gradientsio/env-eval-intercode:basilica .

# ── Stage 1: build the per-fs filesystem snapshots from princeton-nlp/intercode.
FROM ubuntu:22.04 AS intercode_fs

ARG INTERCODE_COMMIT=c3e46d827cfc9d4c704ec078f7abf9f41e3191d8

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash python3 psmisc bsdmainutils cron imagemagick dnsutils git tree \
    net-tools iputils-ping coreutils curl cpio jq ca-certificates \
    findutils gawk grep sed acl attr && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/princeton-nlp/intercode /opt/intercode && \
    cd /opt/intercode && git checkout ${INTERCODE_COMMIT}

COPY ops/docker/intercode_build_fs.sh /opt/intercode-build/build_fs.sh
RUN chmod +x /opt/intercode-build/build_fs.sh && /opt/intercode-build/build_fs.sh


# ── Stage 2: runtime image (SGLang + python deps + baked-in snapshots/data).
FROM lmsysorg/sglang:v0.5.14

WORKDIR /app

COPY pyproject.toml README.md ./
COPY core core
COPY miner miner
COPY ops ops
COPY trainer trainer
COPY validator validator

# Install the gradients package itself so eval_intercode.py can import
# `core.*` and `validator.*`. Keep --upgrade-strategy=only-if-needed so we
# don't churn the SGLang base image's pinned deps.
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed .

# Extra deps the intercode eval needs:
#   peft, accelerate — for LoRA merging (same as validator-env.dockerfile)
#   openai           — talk to the local SGLang OpenAI-compatible endpoint
#   scikit-learn     — TF-IDF answer-similarity in the reward function
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    peft==0.18.1 accelerate==1.6.0 openai scikit-learn

# Bash + standard unix tools required by the NL2Bash gold commands.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnuma1 bash coreutils findutils gawk grep sed file bsdmainutils \
    util-linux jq psmisc imagemagick dnsutils tree net-tools \
    iputils-ping cpio curl git acl attr && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

# Bring in the pre-built fs snapshots and the NL2Bash dataset JSONs.
COPY --from=intercode_fs /intercode_fs /intercode_fs
COPY --from=intercode_fs /opt/intercode/data/nl2bash /intercode_data

ENV SGLANG_PORT=30000
ENV SGLANG_BASE_URL=http://127.0.0.1:30000
ENV SGLANG_HEALTH_PATH=/v1/models
ENV INTERCODE_FS_ROOT=/intercode_fs
ENV INTERCODE_DATA_ROOT=/intercode_data
