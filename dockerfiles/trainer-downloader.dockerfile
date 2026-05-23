FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    huggingface_hub aiohttp pydantic python-dotenv safetensors \
    torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers==4.57.1 peft==0.17.1

COPY trainer/ trainer/
COPY core/ core/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer/utils/trainer_downloader.py"]
