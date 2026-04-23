FROM phoenixbeaudry/game:mcts-api AS mcts_runtime

FROM lmsysorg/sglang:latest

WORKDIR /app

COPY --from=mcts_runtime /app /opt/mcts
ENV PYTHONPATH="/opt/mcts:/app"

COPY pyproject.toml README.md ./
RUN mkdir -p src && touch src/__init__.py

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed .

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed -r /opt/mcts/requirements.txt

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed affinetes==0.1.0 peft==0.18.1 accelerate==1.6.0

COPY --from=mcts_runtime /usr/local/lib/python3.12/site-packages/affinetes /usr/local/lib/python3.12/dist-packages/affinetes
COPY --from=mcts_runtime /usr/local/lib/python3.12/site-packages/affinetes-0.1.0.dist-info /usr/local/lib/python3.12/dist-packages/affinetes-0.1.0.dist-info

RUN apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*

COPY . /app
COPY --from=mcts_runtime /app/env.py /app/env.py

ENV SGLANG_PORT=30000
ENV SGLANG_BASE_URL=http://127.0.0.1:30000
ENV SGLANG_HEALTH_PATH=/v1/models
ENV ENV_SERVER_BASE_URL=http://127.0.0.1:8001
ENV ENV_SERVER_HEALTH_PATH=/health
ENV ENV_SERVER_CMD="python -m uvicorn _affinetes.server:app --host 0.0.0.0 --port 8001 --workers 1 --loop asyncio"