# Image for validator/evaluation/evaluators/swe.py.
# Serves a candidate model with SGLang and calls an external Affinetes
# SWE Infinite environment server.
# Build manually:
#   docker build -f ops/docker/validator-swe-infinite.dockerfile -t gradientsio/env-eval-swe-infinite:basilica .

FROM lmsysorg/sglang:v0.5.14-cu129

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
    peft==0.19.1 accelerate==1.6.0 "mistral-common>=1.8.6"

RUN pip install --no-cache-dir --upgrade \
    openai==2.6.1 pydantic==2.12.5 pydantic-core==2.41.5

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    flash-linear-attention==0.5.1

RUN apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*

# Catch model-server and LoRA-merge dependency skew while building the image,
# before a Basilica deployment spends GPU time discovering it.
RUN python3 -c "import inspect; import openai; import validator.evaluation.evaluators.swe; from core.pvp.sglang_server import _register_olmo_tool_parser; from peft.utils.save_and_load import _maybe_shard_state_dict_for_tp; from pydantic import TypeAdapter; from pydantic_core import core_schema; from transformers import Gemma4ForCausalLM, Gemma4ForConditionalGeneration, GraniteForCausalLM, Lfm2ForCausalLM, Ministral3ForCausalLM, Mistral3ForConditionalGeneration, NemotronHForCausalLM, Olmo3ForCausalLM, OlmoHybridForCausalLM, Qwen3_5ForCausalLM; from transformers.utils.import_utils import is_flash_linear_attention_available; from typing_extensions import TypedDict; _register_olmo_tool_parser(); assert openai.__version__ == '2.6.1'; assert is_flash_linear_attention_available(); assert 'cls_name' in inspect.signature(core_schema.typed_dict_schema).parameters; assert TypeAdapter(TypedDict('Smoke', {'value': int})).validate_python({'value': '1'}) == {'value': 1}"

COPY . /app

ENV SGLANG_PORT=30000
ENV SGLANG_BASE_URL=http://127.0.0.1:30000
ENV SGLANG_HEALTH_PATH=/v1/models
