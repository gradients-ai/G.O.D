FROM lmsysorg/sglang:v0.5.14-cu129

WORKDIR /app

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    open_spiel \
    pydantic \
    pyyaml \
    aiohttp \
    huggingface_hub \
    tenacity \
    basilica-sdk \
    docker \
    git+https://github.com/besimray/fiber.git@v2.6.0 \
    peft==0.19.1 accelerate==1.6.0 "mistral-common>=1.8.6"
# peft + accelerate: continuation-base reconstruction merges the previous-round
# adapter in-container. Baked in, not installed at runtime: transformers caches
# accelerate availability after import. Pins match validator-env.dockerfile.

RUN pip install --no-cache-dir --upgrade \
    openai==2.6.1 pydantic==2.12.5 pydantic-core==2.41.5

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    flash-linear-attention==0.5.1

RUN apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python3 -c "import inspect; import openai; from core.pvp.sglang_server import _register_olmo_tool_parser; from peft.utils.save_and_load import _maybe_shard_state_dict_for_tp; from pydantic import TypeAdapter; from pydantic_core import core_schema; from transformers import Gemma4ForCausalLM, Gemma4ForConditionalGeneration, GraniteForCausalLM, Lfm2ForCausalLM, Ministral3ForCausalLM, Mistral3ForConditionalGeneration, NemotronHForCausalLM, Olmo3ForCausalLM, OlmoHybridForCausalLM, Qwen3_5ForCausalLM; from transformers.utils.import_utils import is_flash_linear_attention_available; from typing_extensions import TypedDict; _register_olmo_tool_parser(); assert openai.__version__ == '2.6.1'; assert is_flash_linear_attention_available(); assert 'cls_name' in inspect.signature(core_schema.typed_dict_schema).parameters; assert TypeAdapter(TypedDict('Smoke', {'value': int})).validate_python({'value': '1'}) == {'value': 1}"

ENV PVP_EVAL_CONFIG=""
ENV EVAL_LOG_LEVEL="INFO"

ENTRYPOINT ["python", "-m", "validator.evaluation.pvp"]
