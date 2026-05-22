FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    huggingface_hub aiohttp pydantic transformers python-dotenv \
    peft==0.17.1 safetensors \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY trainer/ trainer/
COPY core/ core/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer/utils/trainer_downloader.py"]
