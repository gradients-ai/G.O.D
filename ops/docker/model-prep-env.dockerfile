FROM winglian/axolotl:main-20251113

WORKDIR /app

RUN TORCH_VER=$(python -c "import torch; print(torch.__version__)") && \
    pip install --no-cache-dir "sglang==0.5.5.post3" "torch==${TORCH_VER}" datasketch aiohttp python-dotenv textstat \
    "open-spiel==1.6.13" openai

RUN pip install --no-cache-dir "git+https://github.com/besimray/fiber.git@v2.6.0" docker

COPY trainer/model_prep/ trainer/model_prep/
COPY core/ core/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer/model_prep/entrypoint.py"]
