FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    huggingface_hub aiohttp pydantic python-dotenv safetensors \
    torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers==5.12.1 peft==0.19.1
# v5 stack (matches the axolotl images); downloader.py sanitizes the merged output for v4 miner consumers.

COPY trainer/ trainer/
COPY core/ core/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer/containers/downloader.py"]
