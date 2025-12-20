FROM diagonalge/kohya_latest:latest

# Install git (required for pip installations from git repositories)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install core dependencies from pyproject.toml
RUN pip install aiohttp pydantic requests toml \
    "fiber @ git+https://github.com/besimray/fiber.git@v2.6.0" \
    fastapi uvicorn httpx loguru python-dotenv \
    scipy numpy datasets tenacity minio huggingface_hub \
    transformers==4.46.2 pandas==2.2.3 tiktoken==0.8.0 sentencepiece==0.2.0 peft Pillow==11.1.0 PyYAML \
    requests huggingface_hub

RUN mkdir -p /dataset/configs \
    /dataset/outputs \
    /dataset/images \
    /workspace/scripts \
    /workspace/core

COPY core /workspace/core
COPY miner /workspace/miner
COPY trainer /workspace/trainer
COPY scripts /workspace/scripts

RUN chmod +x /workspace/scripts/run_image_trainer.sh
RUN chmod +x /workspace/scripts/image_trainer.py

ENTRYPOINT ["/workspace/scripts/run_image_trainer.sh"]