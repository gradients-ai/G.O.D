FROM diagonalge/ai-toolkit:latest

RUN pip install --no-cache-dir \
    mlflow \
    aiohttp \
    requests \
    fastapi \
    uvicorn \
    httpx \
    loguru \
    scipy \
    numpy \
    pandas==2.2.3 \
    tiktoken==0.8.0 \
    Pillow==11.1.0 \
    "fiber @ git+https://github.com/rayonlabs/fiber.git@2.4.0"

RUN mkdir -p /dataset/configs \
    /dataset/outputs \
    /dataset/images \
    /workspace/ops/trainer_ops \
    /workspace/core

COPY core /workspace/core
COPY miner /workspace/miner
COPY trainer /workspace/trainer
COPY ops/trainer_ops /workspace/ops/trainer_ops

RUN chmod +x /workspace/ops/trainer_ops/run_image_trainer.sh

ENTRYPOINT ["/workspace/ops/trainer_ops/run_image_trainer.sh"]
