FROM phoenixbeaudry/game:mcts-api AS mcts_runtime

FROM lmsysorg/sglang:v0.5.14

WORKDIR /app

COPY --from=mcts_runtime /app /opt/mcts
ENV PYTHONPATH="/opt/mcts:/app"

COPY pyproject.toml README.md ./
COPY core core
COPY miner miner
COPY ops ops
COPY trainer trainer
COPY validator validator

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed .

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed -r /opt/mcts/requirements.txt

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    git+https://github.com/PhoenixBeaudry/affinetes-gradients.git@feat/mcts-api \
    peft==0.18.1 accelerate==1.6.0

RUN apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*

COPY . /app
COPY --from=mcts_runtime /app/env.py /app/env.py

ENV SGLANG_PORT=30000
ENV SGLANG_BASE_URL=http://127.0.0.1:30000
ENV SGLANG_HEALTH_PATH=/v1/models
ENV ENV_SERVER_BASE_URL=http://127.0.0.1:8001
ENV ENV_SERVER_HEALTH_PATH=/health
ENV ENV_SERVER_CMD="python -m uvicorn _affinetes.server:app --host 0.0.0.0 --port 8001 --workers 1 --loop asyncio"
