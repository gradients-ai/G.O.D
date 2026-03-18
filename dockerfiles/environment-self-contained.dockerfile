FROM diagonalge/mcts-api:latest

WORKDIR /app

# Install SGLang runtime into the env-server image so both services
# can run inside the same Basilica deployment container.
RUN pip install --no-cache-dir "sglang[all]"

# Add validator code so `python -m validator.evaluation.eval_environment` is available.
COPY . /app

# Shared runtime defaults used by validator.evaluation.eval_environment.
ENV SGLANG_PORT=30000
ENV SGLANG_BASE_URL=http://127.0.0.1:30000
ENV SGLANG_HEALTH_PATH=/v1/models
ENV ENV_SERVER_BASE_URL=http://127.0.0.1:8001
ENV ENV_SERVER_HEALTH_PATH=/health
ENV ENV_SERVER_CMD="python -m uvicorn _affinetes.server:app --host 0.0.0.0 --port 8001 --workers 1"

# Basilica runner will execute generated source and invoke:
# python -m validator.evaluation.eval_environment
