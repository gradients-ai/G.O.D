FROM lmsysorg/sglang:latest

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
    peft==0.18.1 accelerate==1.6.0
# peft + accelerate: continuation-base reconstruction merges the previous-round
# adapter in-container (materialize_base_model -> _merge_base_and_lora, which uses
# device_map and so requires accelerate). Baked in, not installed at runtime: the
# runtime fallback in _merge_base_and_lora can't help once transformers is already
# imported (it caches accelerate-availability at import time). Pins match
# validator-env.dockerfile.

RUN apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*

COPY . /app

ENV PVP_EVAL_CONFIG=""
ENV EVAL_LOG_LEVEL="INFO"

ENTRYPOINT ["python", "-m", "validator.evaluation.pvp"]
