FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi>=0.110.3,<0.113.0" \
    "httpx==0.27.0" \
    "loguru==0.7.2" \
    "pydantic>=2.9,<3" \
    "uvicorn>=0.30.5,<0.32.0"

COPY subnet_api/ subnet_api/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 app
USER app

CMD ["sh", "-c", "exec uvicorn subnet_api.app:app --factory --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
