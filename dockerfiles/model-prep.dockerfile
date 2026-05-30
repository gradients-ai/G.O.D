FROM winglian/axolotl:main-20251113

WORKDIR /app

RUN TORCH_VER=$(python -c "import torch; print(torch.__version__)") && \
    pip install --no-cache-dir "sglang[srt]" "torch==${TORCH_VER}" datasketch aiohttp python-dotenv textstat

COPY trainer/model_prep/ trainer/model_prep/
COPY core/ core/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer/model_prep/entrypoint.py"]
