# Image for validator/evaluation/evaluators/swe.py.
# Serves a candidate model with SGLang and calls an external Affinetes
# SWE Infinite environment server.
# Build manually:
#   docker build -f ops/docker/validator-swe-infinite.dockerfile -t gradientsio/env-eval-swe-infinite:basilica .

FROM lmsysorg/sglang:v0.5.14

WORKDIR /app

COPY pyproject.toml README.md ./
COPY core core
COPY miner miner
COPY ops ops
COPY trainer trainer
COPY validator validator

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed .

# Keep the Python schema layers paired explicitly. Installing the validator's
# FastAPI-era dependency set over the SGLang image can otherwise leave a newer
# pydantic package calling an older pydantic-core API.
#
# SGLang v0.5.14 ships transformers 5.8.1, whose PEFT integration imports
# _maybe_shard_state_dict_for_tp. PEFT 0.18.1 does not provide that helper.
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    pydantic==2.12.5 pydantic-core==2.41.5 \
    peft==0.19.1 accelerate==1.6.0

RUN apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*

# Catch model-server and LoRA-merge dependency skew while building the image,
# before a Basilica deployment spends GPU time discovering it.
RUN python3 -c "import inspect; import sglang.launch_server; import validator.evaluation.evaluators.swe; from peft.utils.save_and_load import _maybe_shard_state_dict_for_tp; from pydantic import TypeAdapter; from pydantic_core import core_schema; from typing_extensions import TypedDict; assert 'cls_name' in inspect.signature(core_schema.typed_dict_schema).parameters; assert TypeAdapter(TypedDict('Smoke', {'value': int})).validate_python({'value': '1'}) == {'value': 1}"

COPY . /app

ENV SGLANG_PORT=30000
ENV SGLANG_BASE_URL=http://127.0.0.1:30000
ENV SGLANG_HEALTH_PATH=/v1/models
