FROM lmsysorg/sglang:v0.5.14-cu129

WORKDIR /app

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    datasketch aiohttp python-dotenv textstat "open-spiel==1.6.13" openai docker \
    "mistral-common>=1.8.6" peft==0.19.1 accelerate==1.6.0 \
    "git+https://github.com/besimray/fiber.git@v2.6.0"

# Fiber's declared Pydantic cap and SGLang's current schema API cannot be
# resolved in one transaction. Install the application graph first, then keep
# the Pydantic Python/core pair aligned for SGLang at runtime.
RUN pip install --no-cache-dir --upgrade \
    openai==2.6.1 pydantic==2.12.5 pydantic-core==2.41.5

# OLMo Hybrid's Transformers implementation has a correct torch fallback, but
# tournament baselines need the fused recurrent kernels to finish on budget.
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    flash-linear-attention==0.5.1

COPY trainer/model_prep/ trainer/model_prep/
COPY core/ core/

# Fail the image build if SGLang's Transformers-v5 stack cannot resolve any of
# the round-one architecture families or PEFT's v5 integration helper.
RUN python3 -c "import inspect; import openai; from core.pvp.sglang_server import _register_olmo_tool_parser; from peft.utils.save_and_load import _maybe_shard_state_dict_for_tp; from pydantic import TypeAdapter; from pydantic_core import core_schema; from transformers import Gemma4ForCausalLM, Gemma4ForConditionalGeneration, GraniteForCausalLM, Lfm2ForCausalLM, Ministral3ForCausalLM, Mistral3ForConditionalGeneration, NemotronHForCausalLM, Olmo3ForCausalLM, OlmoHybridForCausalLM, Qwen3_5ForCausalLM; from transformers.utils.import_utils import is_flash_linear_attention_available; from typing_extensions import TypedDict; _register_olmo_tool_parser(); assert openai.__version__ == '2.6.1'; assert is_flash_linear_attention_available(); assert 'cls_name' in inspect.signature(core_schema.typed_dict_schema).parameters; assert TypeAdapter(TypedDict('Smoke', {'value': int})).validate_python({'value': '1'}) == {'value': 1}"

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer/model_prep/entrypoint.py"]
