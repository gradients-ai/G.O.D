FROM diagonalge/mcts-api:latest AS mcts_runtime

FROM lmsysorg/sglang:latest

WORKDIR /app

# Bring in MCTS environment server runtime from the original image.
COPY --from=mcts_runtime /app /opt/mcts
ENV PYTHONPATH="/opt/mcts:/app"

# Copy package metadata files so dependency install layer is cached when only code changes.
COPY pyproject.toml README.md ./
RUN mkdir -p src && touch src/__init__.py

# SGLang is provided by the base image.
# Then install base pyproject dependencies (no optional extras).
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed .

# Install MCTS env-server dependencies expected by /opt/mcts/env.py (e.g. open-spiel).
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed -r /opt/mcts/requirements.txt
# Install affinetes dependencies, then overlay the exact affinetes package from mcts image
# so env-server imports match the standalone working image.
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed affinetes==0.1.0
COPY --from=mcts_runtime /usr/local/lib/python3.12/site-packages/affinetes /usr/local/lib/python3.12/dist-packages/affinetes
COPY --from=mcts_runtime /usr/local/lib/python3.12/site-packages/affinetes-0.1.0.dist-info /usr/local/lib/python3.12/dist-packages/affinetes-0.1.0.dist-info

# libnuma1 required by SGLang
RUN apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*

# Add validator code
COPY . /app
# mcts env-server expects user env at /app/env.py
COPY --from=mcts_runtime /app/env.py /app/env.py

# Shared runtime defaults used by validator.evaluation.eval_environment.
ENV SGLANG_PORT=30000
ENV SGLANG_BASE_URL=http://127.0.0.1:30000
ENV SGLANG_HEALTH_PATH=/v1/models
ENV ENV_SERVER_BASE_URL=http://127.0.0.1:8001
ENV ENV_SERVER_HEALTH_PATH=/health
ENV ENV_SERVER_CMD="python -m uvicorn _affinetes.server:app --host 0.0.0.0 --port 8001 --workers 1 --loop asyncio"

# Basilica runner will execute generated source and invoke:
# python -m validator.evaluation.eval_environment
