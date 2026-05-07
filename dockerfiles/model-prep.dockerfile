FROM axolotlai/axolotl:main-py3.11-cu124-2.5.1

WORKDIR /app

RUN pip install --no-cache-dir datasketch aiohttp sglang python-dotenv

COPY trainer/model_prep/ trainer/model_prep/
COPY core/ core/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer/model_prep/entrypoint.py"]
