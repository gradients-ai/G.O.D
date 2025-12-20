FROM axolotlai/axolotl:main-py3.11-cu124-2.5.1

# Install core dependencies from pyproject.toml
RUN pip install mlflow huggingface_hub aiohttp pydantic requests toml \
    "fiber @ git+https://github.com/besimray/fiber.git@v2.6.0" \
    fastapi uvicorn httpx loguru python-dotenv \
    scipy numpy datasets tenacity minio \
    transformers pandas==2.2.3 tiktoken==0.8.0 sentencepiece==0.2.0 peft Pillow==11.1.0 PyYAML \
    requests huggingface_hub textstat==0.7.7 langcheck detoxify

WORKDIR /workspace/axolotl
RUN mkdir -p /workspace/axolotl/configs \
    /workspace/axolotl/outputs \
    /workspace/axolotl/data \
    /workspace/input_data

    
COPY core /workspace/core
COPY miner /workspace/miner
COPY trainer /workspace/trainer
COPY scripts /workspace/scripts
COPY core/config/base.yml /workspace/axolotl/base.yml
COPY core/config/base_grpo.yml /workspace/axolotl/base_grpo.yml

RUN python3 /workspace/core/manual_reward_funcs.py

RUN chmod +x /workspace/scripts/run_text_trainer.sh
RUN chmod +x /workspace/scripts/text_trainer.py

ENTRYPOINT ["/workspace/scripts/run_text_trainer.sh"]
