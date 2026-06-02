FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y git curl git-lfs && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

WORKDIR /app

RUN pip install --no-cache-dir \
    huggingface_hub \
    wandb 

COPY trainer/ trainer/
COPY core/ core/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer/containers/uploader.py"]