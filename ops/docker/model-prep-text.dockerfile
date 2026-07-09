# Text-task model-prep: instruct / dpo / grpo / chat, including continuous-SFT custom-arch (quasar).
# On the transformers-v5 axolotl base so the quasar v5 arch loads. Deliberately NO sglang: it is an
# unused, heavy dependency for text tasks (sglang is only invoked by ENVIRONMENT-task baseline stats,
# which run in the separate model-prep image, ops/docker/model-prep-env.dockerfile). Recent sglang
# (>= v0.5.10) pins transformers v5 too, so the historical v4-conflict reason no longer applies.
FROM axolotlai/axolotl:main-20260701-py3.11-cu128-2.9.1

WORKDIR /app

# axolotl base ships a uv-managed venv at /workspace/axolotl-venv with no `pip` on PATH.
# transformers/datasets/peft come from the base (v5 set) — do NOT pin them here.
RUN uv pip install --python /workspace/axolotl-venv/bin/python --no-cache \
    "git+https://github.com/besimray/fiber.git@v2.6.0" \
    docker datasketch aiohttp python-dotenv textstat

# causal-conv1d (quasar hybrid-arch load-time dep) must build against the base torch ABI, not an
# isolated newer torch (see ops/docker/validator.dockerfile). flash-linear-attention is in the base.
RUN TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX" uv pip install --python /workspace/axolotl-venv/bin/python \
    --no-cache --no-build-isolation causal-conv1d==1.6.2.post1

COPY trainer/model_prep/ trainer/model_prep/
COPY core/ core/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer/model_prep/entrypoint.py"]
